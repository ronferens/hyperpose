import torch
import lightning as pl
from models.pose_losses import CameraPoseLoss
from util import utils
import logging


class BasePoseLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters()

        self._cfg = config
        self._backbone = torch.load(self._cfg.get('backbone_path'))
        self._pose_loss = CameraPoseLoss(self._cfg.get('loss')).cuda()
        self._train_step_outputs = self._reset_accumulated_outputs()
        self._validation_step_outputs = self._reset_accumulated_outputs()
        self._test_step_outputs = self._reset_accumulated_outputs()

    @staticmethod
    def _reset_accumulated_outputs():
        return {'loss': [], 'posit_err': [], 'orient_err': []}

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        input, pose_gt = batch.get('img'), batch.get('pose')
        output = self.forward(input)
        loss = self._pose_loss(output, pose_gt)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, pose_gt = batch.get('img'), batch.get('pose')
        output = self.forward(input)
        loss = self._pose_loss(output, pose_gt)

        # Evaluating the model over the selected validation set
        posit_err, orient_err = utils.pose_err(output.double(), pose_gt)
        self._validation_step_outputs['loss'].append(loss)
        self._validation_step_outputs['posit_err'].append(posit_err)
        self._validation_step_outputs['orient_err'].append(orient_err)

        self.log("val/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        epoch_val_loss = torch.cat(self._validation_step_outputs['loss'], dim=0)
        epoch_val_loss = self.all_gather(epoch_val_loss)
        epoch_val_loss = epoch_val_loss.view(1, -1)

        epoch_val_pose_err = torch.cat(self._validation_step_outputs['posit_err'], dim=0)
        epoch_val_pose_err = self.all_gather(epoch_val_pose_err)
        epoch_val_pose_err = epoch_val_pose_err.view(self.trainer.world_size * epoch_val_pose_err.shape[1])

        epoch_val_orient_err = torch.cat(self._validation_step_outputs['orient_err'], dim=0)
        epoch_val_orient_err = self.all_gather(epoch_val_orient_err)
        epoch_val_orient_err = epoch_val_orient_err.view(self.trainer.world_size * epoch_val_orient_err.shape[1])

        if self.trainer.global_rank == 0:
            logging.info("[Epoch-{}] validation camera pose loss={:.3f}, "
                         "Mean camera pose error: {:.2f}[m], {:.2f}[deg]".format(self.current_epoch,
                                                                                 torch.mean(epoch_val_loss),
                                                                                 torch.mean(epoch_val_pose_err, dim=0),
                                                                                 torch.mean(epoch_val_orient_err, dim=0)
                                                                                 ))

        self._validation_step_outputs = self._reset_accumulated_outputs()

    def test_step(self, batch, batch_idx):
        input, pose_gt = batch.get('img'), batch.get('pose')
        output = self.forward(input)
        loss = self._pose_loss(output, pose_gt)

        # Evaluating the model over the test set
        posit_err, orient_err = utils.pose_err(output.double(), pose_gt)
        self._test_step_outputs['loss'].append(loss)
        self._test_step_outputs['posit_err'].append(posit_err)
        self._test_step_outputs['orient_err'].append(orient_err)

        self.log("test/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def on_test_epoch_end(self) -> None:
        epoch_test_loss = torch.cat(self._test_step_outputs['loss'], dim=0)
        epoch_test_loss = self.all_gather(epoch_test_loss)
        epoch_test_loss = epoch_test_loss.view(1, -1)

        epoch_test_pose_err = torch.cat(self._test_step_outputs['posit_err'], dim=0)
        epoch_test_pose_err = self.all_gather(epoch_test_pose_err)
        epoch_test_pose_err = epoch_test_pose_err.view(self.trainer.world_size * epoch_test_pose_err.shape[1])

        epoch_test_orient_err = torch.cat(self._test_step_outputs['orient_err'], dim=0)
        epoch_test_orient_err = self.all_gather(epoch_test_orient_err)
        epoch_test_orient_err = epoch_test_orient_err.view(self.trainer.world_size * epoch_test_orient_err.shape[1])

        if self.trainer.global_rank == 0:
            logging.info("Test results: camera pose loss={:.3f}, "
                         "Mean camera pose error: {:.2f}[m], {:.2f}[deg]".format(torch.mean(epoch_test_loss),
                                                                                 torch.mean(epoch_test_pose_err, dim=0),
                                                                                 torch.mean(epoch_test_orient_err, dim=0)
                                                                                 ))

        self._test_step_outputs = self._reset_accumulated_outputs()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self._cfg.get('lr'))

        # Setting learning-rate scheduler - Reduce On Plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode="min",
                                                               factor=self._cfg.get('lr_scheduler_gamma'),
                                                               patience=self._cfg.get('lr_scheduler_patience'),
                                                               min_lr=self._cfg.get('min_lr'))

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}
