import torch
import logging
from util import utils
from datasets.CameraPoseDataset import CameraPoseDataset
from models.pose_regressors import load_model_from_checkpoint
from torch.utils.tensorboard import SummaryWriter
import hydra
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import pandas as pd
from os.path import join, basename
import plotly
import plotly.graph_objects as go


@hydra.main(version_base=None, config_path="config", config_name="test")
def main(cfg) -> None:

    # Initiate logger and output folder for the experiment
    log_path = utils.init_logger(outpath=cfg.inputs.output_path)
    utils.save_config_to_output_dir(log_path, cfg)

    # --------------------
    # Data
    # --------------------
    transform = utils.test_transforms.get('baseline')
    dataset = CameraPoseDataset(cfg.inputs.dataset_path, cfg.inputs.testset_path, transform)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    logging.info("Using testing labels file: {}".format(cfg.inputs.testset_path))

    # --------------------
    # Model
    # --------------------
    # Load the checkpoint in case specified
    if cfg.inputs.models_path is None:
        raise Exception('Must specify models\' path')

    checkpoints = utils.get_checkpoint_list(cfg.inputs.models_path)
    models_to_check = sorted(checkpoints['ckpts'].keys())
    if 'last' in checkpoints:
        models_to_check.append('last')
    batch_eval_results = []

    for ckpt in models_to_check:
        if 'last' in str(ckpt):
            model_path = checkpoints['last']
        else:
            model_path = checkpoints['ckpts'][ckpt]

        model = load_model_from_checkpoint(cfg.inputs.model_name, model_path)
        model_name = basename(model_path)
        logging.info("Initializing from checkpoint: {}".format(model_name))

        # --------------------
        # Trainer
        # --------------------
        # Set the seeds and the device
        pl.seed_everything()

        logger = TensorBoardLogger('tb_logs', name='pose_framework_' + utils.get_stamp_from_log(), default_hp_metric=False)

        tester = pl.Trainer(accelerator='auto', devices=1, logger=logger)
        ckpt_res = tester.test(model, test_dataloader)
        batch_eval_results.append([basename(model_path).split('.')[0],
                                   model_path,
                                   ckpt_res[0]['pose/trans_err/test'],
                                   ckpt_res[0]['pose/orient_err/test']])

    # Saving the results
    col_chk_pnt = 'checkpoint'
    col_model = 'Model'
    col_trans_err = 'Median Translation Error [m]'
    col_orient_err = 'Median Orientation Error[deg]'
    batch_eval_results = pd.DataFrame(batch_eval_results, columns=[col_chk_pnt,
                                                                   col_model,
                                                                   col_trans_err,
                                                                   col_orient_err])

    results_file_prefix = cfg.inputs.models_path.split('/')[-1]
    batch_eval_results.to_csv(join(cfg.inputs.models_path, f'{results_file_prefix}_batch_eval.csv'))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=batch_eval_results[col_chk_pnt],
                             y=batch_eval_results[col_trans_err],
                             mode='lines+markers',
                             name=col_trans_err))
    fig.add_trace(go.Scatter(x=batch_eval_results[col_chk_pnt],
                             y=batch_eval_results[col_orient_err],
                             mode='lines+markers',
                             name=col_orient_err))
    fig.update_layout(
        title="Batch model evaluation: {}".format(cfg.inputs.model_name.capitalize()),
        xaxis_title="Model",
        yaxis_title="Position and Orientation Errors",
        legend_title="Error Type",
        font=dict(family="Courier New, monospace")
    )

    # Plotting and saving the figure
    plotly.offline.plot(fig, filename=join(cfg.inputs.models_path, f'{results_file_prefix}_batch_eval.html'))


if __name__ == "__main__":
    main()
