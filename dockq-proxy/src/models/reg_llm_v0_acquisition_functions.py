from typing import Any, Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.regression import MeanSquaredError, SpearmanCorrCoef, R2Score

from lightning.pytorch.loggers.wandb import WandbLogger

from .acquisition_functions.bmdal.features import Features
from .acquisition_functions.bmdal.feature_data import TensorFeatureData
from .acquisition_functions.bmdal.selection import (
    MaxDetSelectionMethod,
    MaxDiagSelectionMethod,
    RandomSelectionMethod,
)
from .acquisition_functions.bmdal.feature_maps import (
    IdentityFeatureMap,
    LaplaceKernelFeatureMap,
    ReLUNNGPFeatureMap,
)

# Grad based approaches
from .acquisition_functions.bmdal.layer_features import (
    LayerGradientComputation,
    LinearGradientComputation,
    create_grad_feature_map,
)

from operator import itemgetter 

class RegressionLLMv0Acquisition(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        model_data_key_translate: Dict[str, str],
        compile: bool,
        target: str = 'dockq', 
        base_kernel = 'nngp',
        selection_method = "maxdet",
        acquisition_samples_save_path=None,
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
        self.val_r2 = R2Score()
        self.test_mse = MeanSquaredError()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_mse_best = MinMetric()
        
        # the target to learn
        self.target = target

        # AL
        self.train_features = None
        self.base_kernel = base_kernel
        self.selection_method = selection_method
        # self.acquisition_samples_save_path = acquisition_samples_save_path
        # if self.acquisition_samples_save_path is None:
        #     self.acquisition_samples_save_path = './samples.csv'
        
        # self.df = pd.DataFrame({'sequence': []})
        

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
        
    
    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     pass

    def select(self, train_features, pool_features, batch_to_query=5):
        # config = None
        # base_kernel = "linear"
        # pool_feats = out_batch['feat_seq']
        # pool_embedding = self.forward(shuffled(batch)) 
        ### something like contrastive/retrieval
        n_features = train_features.shape[-1]

        if self.base_kernel == 'linear':
            feature_map = IdentityFeatureMap(n_features=n_features)
        elif self.base_kernel == 'laplace':
            laplace_scale = 1.0
            feature_map = LaplaceKernelFeatureMap(scale=laplace_scale)
        elif self.base_kernel == 'nngp':
            n_nngp_layers = 3 # TODO: not sure about this hyper-param
            weight_gain = 0.25
            sigma_b = 0.0
            feature_map = ReLUNNGPFeatureMap(
                n_layers=n_nngp_layers,
                sigma_w_sq=weight_gain**2,
                sigma_b_sq=sigma_b**2,
            )
        # elif self.base_kernel == 'll':
        else:
            raise NotImplementedError()

        # grad_dict = {nn.Linear: LinearGradientComputation}
        # grad_layers = []
        # import pdb; pdb.set_trace()
        # for layer in self.net.modules():
        #     if isinstance(layer, LayerGradientComputation):
        #         grad_layers.append(layer)
        #     elif type(layer) in grad_dict:
        #         grad_layers.append(grad_dict[type(layer)](layer))
        # feature_map = create_grad_feature_map(
        #     self.net, grad_layers, 
        #     # use_float64=use_float64
        # )

        if self.selection_method == 'maxdet':
            train_feature_data = TensorFeatureData(train_features)
            pool_feature_data = TensorFeatureData(pool_features)
            train_features = Features(feature_map, train_feature_data)
            pool_features = Features(feature_map, pool_feature_data)
            noise_sigma = 0.01
            noise_sigma = 0.0
            alg = MaxDetSelectionMethod(
                pool_features, train_features,
                noise_sigma=noise_sigma,
                sel_with_train=True, 
                # We are allowing both old and new batches ?
            )
        elif self.selection_method in ['maxdiag', 'random']:
            # Merge pool and train, like sel with train
            pool_features = torch.vstack([pool_features, train_features])
            # print(pool_features.shape)
            pool_feature_data = TensorFeatureData(pool_features)
            pool_features = Features(feature_map, pool_feature_data)
            if self.selection_method == 'maxdiag':
                alg = MaxDiagSelectionMethod(pool_features)
            else:
                alg = RandomSelectionMethod(pool_features)
        else:
            raise NotImplementedError()

        acquisition_idx = alg.select(batch_to_query)
        acquisition_idx = acquisition_idx.tolist()

        # print(acquisition_idx)
        # import pdb; pdb.set_trace()

        # acquisition_samples = list(itemgetter(*acquisition_idx)(batch['sequence']))
        # self.df = pd.concat(
        #     [self.df,  pd.DataFrame({'sequence': acquisition_samples})], ignore_index=True)

        return acquisition_idx
        

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
        tgt_y = batch[self.target].to(torch.float32)
        out_batch = self.forward(batch)
        tgt_pred = out_batch["seq_vectorized"]
        # import pdb; pdb.set_trace()
        loss = self.criterion(tgt_pred, tgt_y)

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

        # update and log metrics
        self.train_loss(loss)
        self.train_mse(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        # self.df.to_csv(self.acquisition_samples_save_path, index=None)
        # print("len df", len(self.df))
        # self.df = pd.DataFrame({'sequence': []})
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
        self.val_mse(preds, targets)
        self.val_scorr(preds, targets)
        self.val_r2(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/scorr", self.val_scorr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True)

        self.log(f"val/{self.target}_mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"val/{self.target}_scorr", self.val_scorr, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"val/{self.target}_r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True)

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
    # _ = RegressionLLMv0(None, None, None, None)
    pass