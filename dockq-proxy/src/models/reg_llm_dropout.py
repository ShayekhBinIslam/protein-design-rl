from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.regression import MeanSquaredError, SpearmanCorrCoef

from lightning.pytorch.loggers.wandb import WandbLogger


class RegressionLLMBayesian(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        model_data_key_translate: Dict[str, str],
        compile: bool,
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

        # metric objects for calculating and averaging accuracy across batches
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.val_scorr = SpearmanCorrCoef()
        self.test_mse = MeanSquaredError()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_mse_best = MinMetric()
        
        # the target to learn
        self.target = target

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

        return self.net(**x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_mse.reset()
        self.val_mse_best.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch['ptm'] = batch['ptm'].to(torch.float32)
        batch['plddt'] = batch['plddt'].to(torch.float32)
        num_uncertainty_steps = 5
        outputs = []
        for _ in range(num_uncertainty_steps):
            # tgt_y = batch[self.target]
            out_batch = self.forward(batch)
            tgt_pred = out_batch["seq_scalar"]
            outputs.append(tgt_pred)
        
        return tgt_pred, outputs
        

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
        batch['ptm'] = batch['ptm'].to(torch.float32)
        batch['plddt'] = batch['plddt'].to(torch.float32)
        tgt_y = batch[self.target]
        out_batch = self.forward(batch)
        tgt_pred = out_batch["seq_scalar"]
        # num_uncertainty_steps = 5
        outputs = []
        # for _ in range(num_uncertainty_steps):
        #     tgt_y = batch[self.target]
        #     out_batch = self.forward(batch)
        #     tgt_pred = out_batch["seq_scalar"]
        #     outputs.append(tgt_pred)
        
        loss = self.criterion(tgt_pred, tgt_y)
        # return loss, tgt_pred, tgt_y
        return loss, tgt_pred, tgt_y, outputs

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, outputs = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_mse(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=True)

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
        loss, preds, targets, outputs = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_mse(preds, targets)
        self.val_scorr(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/scorr", self.val_scorr, on_step=False, on_epoch=True, prog_bar=True)

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
        loss, preds, targets, outputs = self.model_step(batch)

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