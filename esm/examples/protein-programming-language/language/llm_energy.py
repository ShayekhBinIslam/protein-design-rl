from pathlib import Path
import torch.nn.functional as F
import torch
import ray
import sys

from language.lm_design.lm_design import Designer
from language.sequence import ALL_RESIDUE_TYPES, RESIDUE_TYPES_WITHOUT_CYSTEINE

_LOSS_RESULT_IDX = 0

class ESMRewardModelWrapper(Designer):
    def __init__(self):
        self.allowed_AA = ALL_RESIDUE_TYPES

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_models()
        self._init_no_target(20)

@ray.remote(num_gpus=1)
class ESMLLMEnergyCalculator:
    def __init__(self):
        self.esm_reward_calculator = ESMRewardModelWrapper()

        self.ngram_orders = list(range(3))
        self.all_esm_toks = self.esm_reward_calculator.vocab.all_toks
        self.esm_vocab_char_to_idx = self.esm_reward_calculator.vocab.tok_to_idx

    def _encode(
        self,
        sequence: str,
        include_eos: bool
    ) -> 'TensorType["seq_len", "num_aa_types", int]':
        def convert(token):
            return self.esm_vocab_char_to_idx[token]

        sep_idx = sequence.index(':')
        int_idx_list = [
            convert(tkn)
            for tkn in sequence[:sep_idx]
        ]

        if include_eos:
            int_idx_list.append(convert('<eos>'))

        int_esm_encoded_seqs = torch.tensor(int_idx_list).to(
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        return F.one_hot(
            int_esm_encoded_seqs,
            len(self.all_esm_toks)
        ).unsqueeze(0).float()

    def get_nll_energy_term(self, sequence: str) -> float:
        return self.esm_reward_calculator.calc_total_loss(
            x=self._encode(sequence, include_eos=True),
            mask=None,
            LM_w=1.0,
            struct_w=False,
            ngram_w=False,
            ngram_orders=[]
        )[_LOSS_RESULT_IDX].item()

    def get_ngram_energy_term(self, sequence: str) -> float:
        return self.esm_reward_calculator.calc_total_loss(
            x=self._encode(sequence, include_eos=False),
            mask=None,
            LM_w=False,
            struct_w=False,
            ngram_w=1.0,
            ngram_orders=self.ngram_orders
        )[_LOSS_RESULT_IDX].item()
