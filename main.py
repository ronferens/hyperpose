import os
import torch
import logging
from util import utils
from datasets.CameraPoseDataset import CameraPoseDataset
from models.pose_regressors import get_model, load_model_from_checkpoint
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import OmegaConf
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


@hydra.main(version_base=None, config_path="config", config_name="cambridge_train")
def main(cfg) -> None:

    # Initiate logger and output folder for the experiment
    log_path = utils.init_logger(outpath=cfg.inputs.output_path)
    utils.save_config_to_output_dir(log_path, cfg)

    # --------------------
    # Data and CBs
    # --------------------
    if cfg.inputs.mode == 'train':
        # Setting the train and validation datasets
        transform = utils.train_transforms.get('baseline')
        dataset = CameraPoseDataset(cfg.inputs.dataset_path, cfg.inputs.trainset_path, transform)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.general.batch_size, shuffle=True)
        random_sampler = torch.utils.data.RandomSampler(dataset, num_samples=train_dataloader.dataset.dataset_size // 10)
        val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.general.batch_size, sampler=random_sampler)
        logging.info("Using training labels file: {}".format(cfg.inputs.trainset_path))

        # Model checkpoint saving callbacks
        save_format = f"{cfg.inputs.model_name}_{utils.get_stamp_from_log()}_checkpoint-" + "epoch_{epoch}"
        callbacks = []
        best_ckpt_callback = ModelCheckpoint(dirpath=os.path.join(log_path, cfg.general.ckpt_dir_name, 'best'),
                                             auto_insert_metric_name=False,
                                             save_top_k=5,
                                             monitor='loss/val',
                                             filename=save_format,
                                             verbose=True)
        callbacks.append(best_ckpt_callback)
        latest_ckpt_callback = ModelCheckpoint(dirpath=os.path.join(log_path, cfg.general.ckpt_dir_name, 'last'),
                                               auto_insert_metric_name=False,
                                               save_last=True,
                                               filename=save_format,
                                               verbose=True)
        callbacks.append(latest_ckpt_callback)
        if cfg.general.n_freq_checkpoint:
            continuous_ckpt_callback = ModelCheckpoint(dirpath=os.path.join(log_path, cfg.general.ckpt_dir_name),
                                                       save_top_k=-1,
                                                       auto_insert_metric_name=False,
                                                       every_n_epochs=cfg.general.n_freq_checkpoint,
                                                       filename=save_format,
                                                       verbose=True)
            callbacks.append(continuous_ckpt_callback)

    # Setting the test datasets
    transform = utils.test_transforms.get('baseline')
    dataset = CameraPoseDataset(cfg.inputs.dataset_path, cfg.inputs.testset_path, transform)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    logging.info("Using testing labels file: {}".format(cfg.inputs.testset_path))

    # --------------------
    # Model
    # --------------------
    # Load the checkpoint in case specified
    if cfg.inputs.checkpoint_path is not None:
        model = load_model_from_checkpoint(cfg.inputs.model_name, cfg.inputs.checkpoint_path)
        logging.info("Initializing from checkpoint: {}".format(cfg.inputs.checkpoint_path))
    else:
        if cfg.inputs.mode == 'test':
            raise Exception('In test mode must supply a valid checkpoint')
        model_config = OmegaConf.to_container(cfg[cfg.inputs.model_name])
        model = get_model(cfg.inputs.model_name, model_config)

    # Record execution details
    logging.info("Start {} with {}".format(cfg.inputs.model_name, cfg.inputs.mode))
    logging.info("Using dataset: {}".format(cfg.inputs.dataset_path))

    # --------------------
    # Trainer
    # --------------------
    # Set the seeds and the device
    pl.seed_everything()
    trainer = None

    logger = TensorBoardLogger('tb_logs', name='pose_framework_' + utils.get_stamp_from_log(), default_hp_metric=False)

    if cfg.inputs.mode == 'train':
        # Training the model
        trainer = pl.Trainer(max_epochs=cfg.general.n_epochs,
                             accelerator='auto',
                             devices='auto',
                             strategy='ddp_find_unused_parameters_true',
                             logger=logger,
                             callbacks=callbacks)

        trainer.fit(model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader)

    # Testing the model
    if trainer is None or trainer.global_rank == 0:
        tester = pl.Trainer(accelerator='auto', devices=1, logger=logger)
        tester.test(model, test_dataloader)


if __name__ == "__main__":
    main()
