import torch
from functools import partial

from torch.utils.data.dataloader import default_collate
from esm.esmfold.v1.misc import batch_encode_sequences

from openfold.np import residue_constants

# TODO we could move to a more complex encoding wit all the tokens like in the esm model
#  but for now we just implement something fast to test  prediction on last token

ALL_SPECIAL_TOKENS = ['<eos>', '<unk>', '<pad>', '<cls>', '<mask>']
ALL_TOKENS = dict(residue_constants.restype_order_with_x)
ALL_TOKENS.update({
    k: residue_constants.unk_restype_index + 1 + v for k, v in
    zip(ALL_SPECIAL_TOKENS, range(len(ALL_SPECIAL_TOKENS)))
})


def collate_dockp_info(
        batch,
        seq_id="sequences",
        residue_index_offset=512,
        chain_linker="G" * 25,
        add_ending_token=False,
):
    """Collate function for dockq-p dataset."""

    # Collate with default method info from the dict
    data = {}
    for kk, vv in batch[0].items():
        data[kk] = default_collate([b[kk] for b in batch])

    # Transform sequences to tensors
    sequences = [b[seq_id] for b in batch]

    if add_ending_token:
        sequences = [seq + "X" for seq in sequences]

    # TODO: check how slow this is? Maybe worth moving some processing on the dataset loader size
    #   where we have multiprocessing (at least the encode_sequence part)
    #   which is done sequentially
    aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
        sequences, residue_index_offset, chain_linker
    )

    if add_ending_token:
        # Change aatype for last non-masked token to <eos> token
        aatype[torch.arange(len(aatype)), mask.sum(1) - 1] = ALL_TOKENS["<eos>"]

    data.update({
        "aatype": aatype,
        "mask": mask,
        "linker_mask": linker_mask,
        "chain_index": chain_index,
        "residx": _residx,
    })

    # Transform to specific datatype (e.g. from float64 to float32)
    # data["dockq"] = data["dockq"].float()

    return data


def default_collate_fn(**kwargs):
    seq_id = kwargs.get("seq_id", "sequence")
    residue_index_offset = kwargs.get("residue_index_offset", 512)
    chain_linker = kwargs.get("chain_linker", "G" * 25)
    add_ending_token = kwargs.get("add_ending_token", False)

    collate_method = partial(
        collate_dockp_info,
        seq_id=seq_id,
        residue_index_offset=residue_index_offset,
        chain_linker=chain_linker,
        add_ending_token=add_ending_token
    )
    return collate_method