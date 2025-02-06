"""
Entry point training and testing multi-scene transformer
"""
import argparse
import torch
import pandas as pd
import numpy as np
import json
import logging
from util import utils
import time
from datasets.CameraPoseDataset import CameraPoseDataset
from models.pose_regressors import get_model
from os.path import splitext, join
import plotly
import plotly.graph_objects as go


def test_scene(args, config, model, verbose=False):
    model.eval()

    # Set the dataset and data loader
    transform = utils.test_transforms.get('baseline')
    dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform, False)
    loader_params = {'batch_size': 1,
                     'shuffle': False,
                     'num_workers': config.get('n_workers')}
    dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

    stats = np.zeros((len(dataloader.dataset), 3))

    with torch.no_grad():
        for i, minibatch in enumerate(dataloader, 0):
            _ = minibatch.pop('img_path', None)[0]

            for k, v in minibatch.items():
                minibatch[k] = v.to(device)

            gt_pose = minibatch.get('pose').to(dtype=torch.float32)

            # Forward pass to predict the pose
            tic = time.time()
            est_pose, x_embs = model(minibatch.get('img'))
            toc = time.time()

            # Evaluate error
            posit_err, orient_err = utils.pose_err(est_pose, gt_pose)

            # Collect statistics
            stats[i, 0] = posit_err.item()
            stats[i, 1] = orient_err.item()
            stats[i, 2] = (toc - tic) * 1000

    # Record overall statistics
    logging.info("\tMedian pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]),
                                                                      np.nanmedian(stats[:, 1])))
    logging.info("\tMean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))
    return stats


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model_name",
                            help="name of model to create (e.g. posenet, hyperpose")
    arg_parser.add_argument("mode", help="train or eval")
    arg_parser.add_argument("backbone_path", help="path to backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("labels_file", help="path to a file mapping images to their poses")
    arg_parser.add_argument("config_file", help="path to configuration file", default="7scenes-config.json")
    arg_parser.add_argument("checkpoint_path",
                            help="path to a pre-trained model (should match the model indicated in model_name")
    arg_parser.add_argument("--output_path", help="path to save the experiment's output")
    arg_parser.add_argument("--ckpt_start_index", help="indicating the first checkpoint to test", default=0, type=int)

    args = arg_parser.parse_args()

    log_path = utils.init_logger(outpath=args.output_path)

    # Record execution details
    logging.info("Start {} with {}".format(args.model_name, args.mode))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    model_params = config[args.model_name]
    general_params = config['general']
    config = {**model_params, **general_params}
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Create the model
    model, model_dir = get_model(args.model_name, args.backbone_path, config)
    model.to(device)

    ckpt_list = utils.get_checkpoint_list(args.checkpoint_path, args.ckpt_start_index)
    models_to_check = sorted(ckpt_list['ckpts'].keys())
    if 'final' in ckpt_list:
        models_to_check.append('final')
    batch_eval_results = []

    num_models = len(models_to_check)
    logging.info(f'Number of model to evaluate : {num_models}')
    for model_index, ckpt in enumerate(models_to_check):

        if 'final' in str(ckpt):
            model_path = ckpt_list['final']
        else:
            model_path = ckpt_list['ckpts'][ckpt]

        model.load_state_dict(torch.load(model_path, map_location=device_id))
        logging.info(f'{model_index + 1}/{num_models}) Initializing from checkpoint: {model_path}')

        stats = test_scene(args, config, model, model_path)

        # Saving the models' evaluation results
        checkpoint_name = splitext(model_path.split('_')[-1])[0]
        batch_eval_results.append([checkpoint_name,
                                   model_path,
                                   np.nanmedian(stats[:, 0]),
                                   np.nanmedian(stats[:, 1])])

    # Saving the results
    col_chk_pnt = 'checkpoint'
    col_model = 'Model'
    col_pos_err = 'Median Position Error [m]'
    col_orient_err = 'Median Orientation Error[deg]'
    batch_eval_results = pd.DataFrame(batch_eval_results, columns=[col_chk_pnt,
                                                                   col_model,
                                                                   col_pos_err,
                                                                   col_orient_err])

    batch_eval_results.to_csv(join(args.checkpoint_path, f'{checkpoint_name}_batch_eval.csv'))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=batch_eval_results[col_chk_pnt],
                             y=batch_eval_results[col_pos_err],
                             mode='lines+markers',
                             name=col_pos_err))
    fig.add_trace(go.Scatter(x=batch_eval_results[col_chk_pnt],
                             y=batch_eval_results[col_orient_err],
                             mode='lines+markers',
                             name=col_orient_err))
    fig.update_layout(
        title="Batch model evaluation: {}".format(args.model_name.capitalize()),
        xaxis_title="Model",
        yaxis_title="Position and Orientation Errors",
        legend_title="Error Type",
        font=dict(family="Courier New, monospace")
    )

    # Plotting and saving the figure
    plotly.offline.plot(fig, filename=join(args.checkpoint_path, f'{checkpoint_name}_batch_eval.html'))