from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.regression import MeanSquaredError, SpearmanCorrCoef, R2Score

from lightning.pytorch.loggers.wandb import WandbLogger

import numpy as np


class RegressionLLMExplorationMultiHead(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        model_data_key_translate: Dict[str, str],
        compile: bool,
        exploration: Any = None,
        plddt_loss_weight: float = 1.0,
        target: str = 'dockq', 
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # TODO Not sure if this should rather be in the data config
        self.model_data_key_translate = model_data_key_translate

        # loss function
        self.criterion = torch.nn.MSELoss()
        self.plddt_loss_weight = plddt_loss_weight

        # metric objects for calculating and averaging accuracy across batches
        self.train_mse = MeanSquaredError()

        self.val_mse = MeanSquaredError()
        self.val_scorr = SpearmanCorrCoef()
        self.val_r2 = R2Score()

        self.val_ptm_loss = MeanMetric()
        self.val_ptm_mse = MeanSquaredError()
        self.val_ptm_scorr = SpearmanCorrCoef()
        self.val_ptm_r2 = R2Score()
        
        self.val_plddt_loss = MeanMetric()
        self.val_plddt_mse = MeanSquaredError()
        self.val_plddt_scorr = SpearmanCorrCoef()
        self.val_plddt_r2 = R2Score()


        self.test_mse = MeanSquaredError()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_mse_best = MinMetric()
        
        # the target to learn
        self.target = target

        self.exploration = exploration
        # self.exploration = "masking"
        # self.exploration = "probabilistic_masking"

    def get_transform(self, data_cfg):
        if hasattr(self.net, "get_transform"):
            return self.net.get_transform(data_cfg)

        return None

    def get_collate_fn(self, data_cfg):
        if hasattr(self.net, "get_collate_fn"):
            return self.net.get_collate_fn(data_cfg)

        return None

    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        # rename dict keys to match model input names
        if self.model_data_key_translate is not None:
            for from_key, to_key in self.model_data_key_translate.items():
                if from_key in x:
                    x[to_key] = x.pop(from_key)
        
        # import pdb; pdb.set_trace()
        return self.net(**x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_mse.reset()
        self.val_mse_best.reset()

    def model_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        if self.exploration is not None:
            ptm = batch['ptm'].to(torch.float32)
            plddt = batch['plddt'].to(torch.float32)
            tgt_y = torch.stack([ptm, plddt], dim=1)

            multihead_output = self.forward(batch)
            # num_heads x batch_size x num_features (eg 5 x 32 x 2)
            tgt_pred = torch.stack([output for output in multihead_output["seq_vectorized"]])

        if self.exploration == "masking":
            chosen_head = np.random.randint(0, self.net.num_output_heads)
            tgt_pred_chosen = tgt_pred[chosen_head, ...]

        elif self.exploration == "probabilistic_masking":
            tgt_pred_chosen = tgt_pred.clone()
            mask = torch.rand(self.net.num_output_heads).to(tgt_pred.device)
            mask /= mask.sum()
            tgt_pred_chosen = torch.zeros_like(tgt_pred)
            for m in range(self.net.num_output_heads): 
                tgt_pred_chosen[m] =  mask[m]*tgt_pred[m]
            
            tgt_pred_chosen = tgt_pred_chosen.sum(axis=0)

        elif self.exploration == "bootstrap": # here we choose different head for different data-points
            bsz = tgt_y.shape[0]
            chosen_heads = torch.randint(0, self.net.num_output_heads, (bsz,)) # # Generate mask
            row_indices = torch.arange(0, bsz, dtype=torch.int)
            tgt_pred_chosen = tgt_pred[chosen_heads, row_indices]
            
        
        if self.exploration is not None:
            # Axis 0 is ptm, axis 1 is plddt; weighted additive MSE loss
            loss = self.criterion(tgt_pred_chosen[:, 0], tgt_y[:, 0]) + \
                    self.plddt_loss_weight * self.criterion(tgt_pred_chosen[:, 1], tgt_y[:, 1])

        
        else:  #default, for scalar version
            tgt_y = batch[self.target].to(torch.float32)
            out_batch = self.forward(batch)
            tgt_pred = out_batch["seq_vectorized"]
            
            loss = self.criterion(tgt_pred, tgt_y)

        # return loss, tgt_pred, tgt_y
        return loss, tgt_pred, tgt_y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        preds_ptm, targets_ptm = preds[..., 0], targets[..., 0]
        if len(preds_ptm.size()) ==  2: # For multi-head average over the predictions
            preds_ptm = preds_ptm.mean(axis=0)
        
        preds_plddt, targets_plddt = preds[..., 1], targets[..., 1]
        if len(preds_plddt.size()) ==  2: # For multi-head average over the predictions
            preds_plddt = preds_plddt.mean(axis=0)

        # update and log metrics
        self.train_loss(loss)
        self.train_mse(preds_ptm, targets_ptm)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/ptm_mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        # self.val_mse(preds, targets)
        # self.val_scorr(preds, targets)
        # self.val_r2(preds, targets)
        # self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/scorr", self.val_scorr, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True)

        preds_ptm, targets_ptm = preds[..., 0], targets[..., 0]
        if len(preds_ptm.size()) ==  2: # For multi-head average over the predictions
            preds_ptm = preds_ptm.mean(axis=0)

        self.val_ptm_mse(preds_ptm, targets_ptm)
        self.val_ptm_scorr(preds_ptm, targets_ptm)
        self.val_ptm_r2(preds_ptm, targets_ptm)
        self.log("val/ptm_mse", self.val_ptm_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ptm_scorr", self.val_ptm_scorr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ptm_r2", self.val_ptm_r2, on_step=False, on_epoch=True, prog_bar=True)

        preds_plddt, targets_plddt = preds[..., 1], targets[..., 1]
        if len(preds_plddt.size()) ==  2: # For multi-head average over the predictions
            preds_plddt = preds_plddt.mean(axis=0)

        self.val_plddt_mse(preds_plddt, targets_plddt)
        self.val_plddt_scorr(preds_plddt, targets_plddt)
        self.val_plddt_r2(preds_plddt, targets_plddt)
        self.log("val/plddt_mse", self.val_plddt_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/plddt_scorr", self.val_plddt_scorr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/plddt_r2", self.val_plddt_r2, on_step=False, on_epoch=True, prog_bar=True)

        self.val_mse = self.val_ptm_mse + self.val_plddt_mse
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True,
                    metric_attribute='val_mse')
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_mse.compute()  # get current val acc
        self.val_mse_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/mse_best", self.val_mse_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_mse(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            print(f"Using {self.hparams.scheduler} as scheduler")
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = RegressionLLMv0(None, None, None, None)
