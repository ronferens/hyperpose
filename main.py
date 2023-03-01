import pandas as pd
import torch
import numpy as np
import logging
from util import utils
import time
from datasets.CameraPoseDataset import CameraPoseDataset
from models.pose_losses import CameraPoseLoss
from models.pose_regressors import get_model
from os.path import join
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="cambridge_train")
def main(cfg) -> None:

    # Initiate logger and output folder for the experiment
    log_path = utils.init_logger(outpath=cfg.inputs.output_path)
    utils.save_config_to_output_dir(log_path, cfg)

    # Record execution details
    logging.info("Start {} with {}".format(cfg.inputs.model_name, cfg.inputs.mode))
    if cfg.inputs.experiment is not None:
        logging.info("Experiment details: {}".format(cfg.inputs.experiment))
    logging.info("Using dataset: {}".format(cfg.inputs.dataset_path))
    logging.info("Using labels file: {}".format(cfg.inputs.labels_file))

    # Init Tensorboard and set experiment timestamp
    writer = SummaryWriter()

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(numpy_seed)

    # Create the model
    model_config = OmegaConf.to_container(cfg[cfg.inputs.model_name])
    model = get_model(cfg.inputs.model_name, cfg.inputs.backbone_path, model_config)
    model = torch.nn.DataParallel(model, device_ids=cfg.general.devices_id)
    model.cuda()

    # Load the checkpoint if needed
    if cfg.inputs.checkpoint_path:
        model.load_state_dict(torch.load(cfg.inputs.checkpoint_path))
        logging.info("Initializing from checkpoint: {}".format(cfg.inputs.checkpoint_path))

    if cfg.inputs.mode == 'train':
        # Set to train mode
        model.train()

        # Freeze parts of the model if indicated
        freeze = cfg[cfg.inputs.model_name].freeze
        freeze_exclude_phrase = cfg[cfg.inputs.model_name].freeze_exclude_phrase
        if isinstance(freeze_exclude_phrase, str):
            freeze_exclude_phrase = [freeze_exclude_phrase]
        if freeze:
            for name, parameter in model.named_parameters():
                freeze_param = True
                for phrase in freeze_exclude_phrase:
                    if phrase in name:
                        freeze_param = False
                        break
                if freeze_param:
                    parameter.requires_grad_(False)

        # Set the loss
        loss_config = OmegaConf.to_container(cfg[cfg.inputs.model_name].loss)
        pose_loss = CameraPoseLoss(loss_config).cuda()

        # Set the optimizer and scheduler
        params = list(model.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, params),
                                  lr=cfg[cfg.inputs.model_name].lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=cfg[cfg.inputs.model_name].lr_scheduler_step_size,
                                                    gamma=cfg[cfg.inputs.model_name].lr_scheduler_gamma)

        # Set the dataset and data loader
        transform = utils.train_transforms.get('baseline')
        dataset = CameraPoseDataset(cfg.inputs.dataset_path, cfg.inputs.labels_file, transform)
        loader_params = {'batch_size': cfg.general.batch_size,
                         'shuffle': True,
                         'num_workers': cfg.general.n_workers}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # Get training details
        n_freq_print = cfg.general.n_freq_print
        n_freq_checkpoint = cfg.general.n_freq_checkpoint
        n_epochs = cfg.general.n_epochs
        start_save_epoch = cfg.general.start_save_epoch

        # Train
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        for epoch in range(n_epochs):

            # Resetting temporal loss used for logging
            running_loss = 0.0
            n_samples = 0

            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.cuda()
                gt_pose = minibatch.get('pose').to(dtype=torch.float32)
                batch_size = gt_pose.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                if freeze:  # For TransPoseNet
                    model.eval()
                    with torch.no_grad():
                        transformers_res = model.module.forward_transformers(minibatch)
                    model.train()

                # Zero the gradients
                optim.zero_grad()

                # Forward pass to estimate the pose
                if freeze:
                    res = model.module.forward_heads(transformers_res)
                else:
                    res = model(minibatch)

                est_pose = res.get('pose')
                # Pose loss
                criterion = pose_loss(est_pose, gt_pose)

                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    posit_err, orient_err = utils.pose_err(est_pose.detach(), gt_pose.detach())
                    logging.info("[Batch-{}/Epoch-{}] running camera pose loss: {:.3f}, "
                                 "camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                        batch_idx + 1, epoch + 1, (running_loss / n_samples),
                        posit_err.mean().item(),
                        orient_err.mean().item()))
                    writer.add_scalar("Loss/train", (running_loss / n_samples), n_total_samples)
                    writer.add_scalar("Pose/translation_train", posit_err.mean().item(), n_total_samples)
                    writer.add_scalar("Pose/orientation_train", orient_err.mean().item(), n_total_samples)
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch >= start_save_epoch:
                torch.save(model.state_dict(),
                           join(log_path, f'{utils.get_stamp_from_log()}_checkpoint-{epoch}.pth'))

            # Scheduler update
            scheduler.step()
            writer.add_scalar("Loss/lr", scheduler.get_lr()[0], epoch)

        logging.info('Training completed')
        torch.save(model.state_dict(), join(log_path, f'{utils.get_stamp_from_log()}_final.pth'.format(epoch)))
        writer.flush()
        writer.close()

        # Plot the loss function
        loss_fig_path = join(log_path, f'{utils.get_stamp_from_log()}_loss_fig.png')
        utils.plot_loss_func(sample_count, loss_vals, loss_fig_path)

    else:  # Test
        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        transform = utils.test_transforms.get('baseline')
        dataset = CameraPoseDataset(cfg.inputs.dataset_path, cfg.inputs.labels_file, transform)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': cfg.general.n_workers}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        stats = np.zeros((len(dataloader.dataset), 3))

        if cfg.general.save_hyper_weights_to_file:
            hyperparams = np.zeros((len(dataloader.dataset), 387 + 2052))  # For saving the output layer's weights

        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.cuda()

                gt_pose = minibatch.get('pose').to(dtype=torch.float32)

                # Forward pass to predict the pose
                tic = time.time()
                res = model(minibatch)
                est_pose = res.get('pose')
                w_t = res.get('w_t')
                w_rot = res.get('w_rot')
                toc = time.time()

                # Evaluate error
                posit_err, orient_err = utils.pose_err(est_pose, gt_pose)

                # Collect statistics
                stats[i, 0] = posit_err.item()
                stats[i, 1] = orient_err.item()
                stats[i, 2] = (toc - tic) * 1000.0

                logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                    stats[i, 0], stats[i, 1], stats[i, 2]))

                # Save hypernetwork's output (weights)
                if cfg.general.save_hyper_weights_to_file:
                    hyperparams[i, :] = np.concatenate((w_t.data.cpu(), w_rot.data.cpu()), axis=1).reshape(-1)

        # Record overall statistics
        logging.info("Performance of {} on {}".format(cfg.inputs.checkpoint_path, cfg.inputs.labels_file))
        logging.info(
            "Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]), np.nanmedian(stats[:, 1])))
        logging.info("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))

        if cfg.general.save_hyper_weights_to_file:
            hyperparams_data = pd.DataFrame(hyperparams)
            hyperparams_data.to_csv('hypernetwork_weights_w_in.csv')


if __name__ == "__main__":
    main()
