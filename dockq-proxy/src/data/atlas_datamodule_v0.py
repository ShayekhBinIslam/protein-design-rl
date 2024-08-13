from typing import Any, Dict, Optional, Callable

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

# switch from gs to s3
import boto3
from urllib.parse import urlparse

class AtlasDataset(Dataset):
    """Load dataset from aggregated aalpha submission data ."""

    def __init__(self, cfg, df=None, transform=None, drop_na=True, split: Optional[str] = None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # TODO implement some local caching method in order to not download
        #  the dataset every time.
        # df = pd.read_csv(cfg.dataset_path)

        if df is None:
            if cfg.dataset_path.startswith("s3://"):
                parsed_url = urlparse(cfg.dataset_path)
                bucket_name = parsed_url.netloc
                file_path = parsed_url.path.lstrip('/')
                
                s3 = boto3.client('s3')
                obj = s3.get_object(Bucket=bucket_name, Key=file_path)
                df = pd.read_csv(obj['Body'])
            else: 
                df = pd.read_csv(cfg.dataset_path)
                
        if split is not None:
            df = df[df["split"] == split]

        if drop_na:
            df = df.dropna()
            # plddt: 1319 null values of 70k
        self._df = df
        self._transform = transform
        # import pdb; pdb.set_trace()


    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rows = self._df.iloc[idx]
        
        sample = {
            "sequence": rows["sequence"],
            "ptm": rows.get("ptm", -1),
            "plddt": rows.get("plddt", -1),
        }

        if self._transform:
            sample = self._transform(sample)

        return sample


class AtlasV0(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        dataset_cfg: Optional[Dict[str, Any]] = None,
        transform: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
    ) -> None:
        """
        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self._transform = transform
        self._collate_fn = collate_fn

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self._dataset_cfg = dataset_cfg

    # @property
    # def num_classes(self) -> int:
    #     """Get the number of classes.

    #     :return: The number of MNIST classes (10).
    #     """
    #     return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices"
                    f" ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            train_set = AtlasDataset(self._dataset_cfg, transform=self.transforms, split="train")
            test_set = AtlasDataset(self._dataset_cfg, transform=self.transforms, split="test")
            validation = AtlasDataset(self._dataset_cfg, transform=self.transforms, split="validation")
            self.data_train, self.data_val, self.data_test = train_set, test_set, validation

    @property
    def transforms(self):
        return self._transform

    @transforms.setter
    def transforms(self, transform):
        self._transform = transform

    @property
    def collate_fn(self):
        return self._collate_fn

    @collate_fn.setter
    def collate_fn(self, collate_fn):
        self._collate_fn = collate_fn

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = AtlasV0()
