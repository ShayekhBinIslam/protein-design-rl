from functools import partial
import contextlib
from typing import List, Tuple

import torch
from torch import nn, Tensor

from torch.utils.data.dataloader import default_collate

from src.utils.load_esm_model import load_model_and_alphabet

# Let's try our loader, so we can deactivate pretrained weights
load_fn = load_model_and_alphabet

esm_registry = {
    "esm2_8M": partial(load_fn, "esm2_t6_8M_UR50D_500K"),
    "esm2_8M_270K": partial(load_fn, "esm2_t6_8M_UR50D"),
    "esm2_35M": partial(load_fn, "esm2_t12_35M_UR50D_500K"),
    "esm2_35M_270K": partial(load_fn, "esm2_t12_35M_UR50D"),
    "esm2_150M": partial(load_fn, "esm2_t30_150M_UR50D_500K"),
    "esm2_150M_270K": partial(load_fn, "esm2_t30_150M_UR50D_270K"),
    "esm2_650M": partial(load_fn, "esm2_t33_650M_UR50D"),
    "esm2_650M_270K": partial(load_fn, "esm2_t33_650M_270K_UR50D"),
    "esm2_3B": partial(load_fn, "esm2_t36_3B_UR50D"),
    "esm2_3B_270K": partial(load_fn, "esm2_t36_3B_UR50D_500K"),
    "esm2_15B": partial(load_fn, "esm2_t48_15B_UR50D"),
}


@contextlib.contextmanager
def null_context():
    """
    A null context manager.
    """
    yield


# Conditional context manager
def get_grad_context(freeze):
    """
    Returns torch.no_grad() if freeze is True, otherwise an empty context.
    """
    return torch.no_grad() if freeze else null_context()


class SimpleESMV0(nn.Module):

    def __init__(
            self,
            esm_type: str = "esm2_8M",
            pretrained: bool = True,
            scalar_linear_layers: int = 2,
            freeze_esm: bool = False,
            representation_layer: int = -1,
            hsize: int = 256,
    ):
        super().__init__()

        # Load ESM model
        self.esm, self.esm_alphabet = esm_registry.get(esm_type)(pretrained=pretrained)
        if freeze_esm:
            self.esm.eval()
            self.esm.requires_grad_(False)
        self._freeze_esm = freeze_esm

        # Configure the represenation layer we want to use as features for the
        #   per sequence scalar predictor
        if representation_layer < 0:
            representation_layer = len(self.esm.layers) + representation_layer

        self._repr_layer = [representation_layer]

        # Determine size of representation feature
        test = self.run_esm([("test", "GG")], repr_layer=representation_layer)
        token_representations = test["representations"][representation_layer]
        r_out_size = token_representations.shape[-1]

        # Initialize scalar predictor
        s_modules = []
        feat_size = r_out_size
        for _ in range(scalar_linear_layers - 1):
            s_modules.append(nn.Linear(feat_size, hsize))
            s_modules.append(nn.ReLU())
            feat_size = hsize
        s_modules.append(nn.Linear(feat_size, 1))
        self._scalar_mlp = nn.Sequential(*s_modules)

    def run_esm(self, data: List[Tuple[str, str]], repr_layer: int):
        """ Run ESM on a list of sequences.
        data example:
            [("label", "protein_string"),
        """
        assert repr_layer in range(len(self.esm.layers) + 1), "Invalid representation layer index"

        batch_converter = self.esm_alphabet.get_batch_converter()

        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != self.esm_alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = self.esm(batch_tokens, repr_layers=[repr_layer])

        return results



    def get_collate_fn(self, data_cfg):
        # TODO Refactor this!
        #   Would be great if we could use multiple collate_fn in sequence
        #   run the dataset collate -> model collate
        # Data comes as a list of dictionaries.

        batch_converter = self.esm_alphabet.get_batch_converter()
        chain_linker = "G" * 25

        def _collate_fn(batch):
            # Collate with default method info from the dict
            data_out = {}
            for kk, vv in batch[0].items():
                data_out[kk] = default_collate([b[kk] for b in batch])

            data = []
            for ix, x in enumerate(batch):
                sequence = x["sequence"]
                sequence = sequence.replace(":", chain_linker)
                # TODO Is there anything else we need to prepare for
                #  the esm model to work with multimers?
                data.append((f"{ix}", sequence))

            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_lens = (batch_tokens != self.esm_alphabet.padding_idx).sum(1)
            data_out.update(
                {
                    "batch_labels": batch_labels,
                    "batch_strs": batch_strs,
                    "batch_tokens": batch_tokens,
                    "batch_lens": batch_lens,
                }
            )

            # Transform to specific datatype (e.g. from float64 to float32)
            # data_out["dockq"] = data_out["dockq"].float()

            return data_out
        return _collate_fn

    def forward(
        self,
        batch_labels, batch_strs, batch_tokens, batch_lens,
        repr_layers=None,
        return_contacts=False,
        **kwargs
    ):
        """ """
        repr_layers = self._repr_layer if repr_layers is None else repr_layers

        with get_grad_context(self._freeze_esm):
            results = self.esm(
                batch_tokens,
                repr_layers=repr_layers,
                return_contacts=return_contacts
            )

        token_representations = results["representations"][repr_layers[-1]]

        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
        sequence_representations = torch.stack(sequence_representations)

        results["sequence_representations"] = sequence_representations
        results["seq_vectorized"] = self._scalar_mlp(sequence_representations).squeeze(-1)

        return results