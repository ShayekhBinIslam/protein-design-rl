"""Defines the BertGFPBrightness landscape."""
import os

import numpy as np
import requests
import tape
import torch
# import design_bench

import flexs
import flexs.utils.sequence_utils as s_utils

def print_mem():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved

    print()
    print('Total mem: %d' % t)
    print('Reserved mem: %d' % r)
    print('Allocated mem: %d' % a)
    print('Free mem: %d' % f)
    print()

    if ((t - f) / t) > .99:
        import pdb; pdb.set_trace()

class BertGFPBrightness(flexs.Landscape):
    r"""
    Green fluorescent protein (GFP) brightness landscape.

    The oracle used in this lanscape is the transformer model
    from TAPE (https://github.com/songlab-cal/tape).

    To create the transformer model used here, run the command:

        ```tape-train transformer fluorescence --from_pretrained bert-base \
                                               --batch_size 128 \
                                               --gradient_accumulation_steps 10 \
                                               --data_dir .```

    Note that the output of this landscape is not normalized to be between 0 and 1.

    Attributes:
        gfp_wt_sequence (str): Wild-type sequence for jellyfish
            green fluorescence protein.
        starts (dict): A dictionary of starting sequences at different edit distances
            from wild-type with different difficulties of optimization.

    """

    gfp_wt_sequence = (
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVT"
        "TLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIE"
        "LKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNT"
        "PIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
    )

    starts = {
        "ed_10_wt": "MSKGEVLFTGVVPILVEMDGDVNGHKFSVSGEGEGDATYGKLTTKFTCTTGKLPVPWPTKVTTLSYRVQCFSRYPDVMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVQFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNIKRDCMVLLEFVTAAGITHGMDELY",  # noqa: E501
        "ed_18_wt": "MSKGEHLFTGVVPILVELDGDVNGKKFSVSGEGQGDATYGKLTLKFICTTAKVHVPWCTLVTTLSYGVQCFSRYPDHMKQHDFFKGAMPEGYVQERTIFFKDIGNYKLRAEVKFEGDTLVNRIELKGIDFKEDGNIHGHKLEYNYNSQNVYIMASKQKNGIKVNFKIRLNIEDGSVQLAEHYQVNTPIGDFPVLLPDNHKLSAQSADSKDPNEKRDHMHLLEFVTAVGITHGMDELYK",  # noqa: E501
        "ed_31_wt": "MSKGEELFSGVQPILVELDGCVNGHKFSVSGEGEIDATYGKLTLKFICTTWKLPMPWPCLVTFGSYGVQCFSRYRDHPKQHDFFKSAVPEGYVQERTIFMKDDLLYKTRAEVKFEGLTLVNRIELKGKDFKEDGNILGHKLEYNYNSHCVYPMADWNKNWIKVNSKIRLPIEDGSVILADHYQQNTPIGDQPVLLPENHYLSTQSALSKDPEEKGDLMVLLEFVTAAGITHGMDELYK",  # noqa: E501
    }

    def __init__(self):
        """
        Create GFP landscape.

        Downloads model into `./fluorescence-model` if not already cached there.
        If interrupted during download, may have to delete this folder and try again.
        """
        super().__init__(name="GFP")

        self.task = design_bench.make('GFP-Transformer-v0')

        self.char_map = {
            char: idx
            for idx, char in enumerate(s_utils.AAS)
        }

        self.inv_char_map = {
            idx: char
            for char, idx in self.char_map.items()
        }

        seq_x = self._cat_encode_to_seq(self.task.x)
        self.x, self.y = seq_x, self.task.y.flatten()

    def _cat_encode_to_seq(self, x):
        return np.array([
            ''.join([self.inv_char_map[val] for val in seq_cats])
            for seq_cats in x
        ])

    def get_dataset(self):
        return self

    def get_full_dataset(self):
        return self.x, self.y

    def add(self, batch):
        samples, scores = batch

        self.x = np.concatenate((self.x, samples), axis=0)
        self.y = np.concatenate((self.y, scores), axis=0)

    def _encode(self, sequences):
        return np.array([
            [self.char_map[char] for char in seq]
            for seq in sequences
        ])

    @property
    def length(self):
        return len(self.x[0])

    @property
    def vocab_size(self):
        return len(s_utils.DB_AAS)

    @property
    def is_dynamic_length(self):
        return False

    def _fitness_function(self, sequences):
        return self.task.predict(self._encode(sequences)).flatten()

#class BertGFPBrightness(flexs.Landscape):
#    r"""
#    Green fluorescent protein (GFP) brightness landscape.
#
#    The oracle used in this lanscape is the transformer model
#    from TAPE (https://github.com/songlab-cal/tape).
#
#    To create the transformer model used here, run the command:
#
#        ```tape-train transformer fluorescence --from_pretrained bert-base \
#                                               --batch_size 128 \
#                                               --gradient_accumulation_steps 10 \
#                                               --data_dir .```
#
#    Note that the output of this landscape is not normalized to be between 0 and 1.
#
#    Attributes:
#        gfp_wt_sequence (str): Wild-type sequence for jellyfish
#            green fluorescence protein.
#        starts (dict): A dictionary of starting sequences at different edit distances
#            from wild-type with different difficulties of optimization.
#
#    """
#
#    gfp_wt_sequence = (
#        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVT"
#        "TLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIE"
#        "LKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNT"
#        "PIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
#    )
#
#    starts = {
#        "ed_10_wt": "MSKGEVLFTGVVPILVEMDGDVNGHKFSVSGEGEGDATYGKLTTKFTCTTGKLPVPWPTKVTTLSYRVQCFSRYPDVMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVQFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNIKRDCMVLLEFVTAAGITHGMDELYK",  # noqa: E501
#        "ed_18_wt": "MSKGEHLFTGVVPILVELDGDVNGKKFSVSGEGQGDATYGKLTLKFICTTAKVHVPWCTLVTTLSYGVQCFSRYPDHMKQHDFFKGAMPEGYVQERTIFFKDIGNYKLRAEVKFEGDTLVNRIELKGIDFKEDGNIHGHKLEYNYNSQNVYIMASKQKNGIKVNFKIRLNIEDGSVQLAEHYQVNTPIGDFPVLLPDNHKLSAQSADSKDPNEKRDHMHLLEFVTAVGITHGMDELYK",  # noqa: E501
#        "ed_31_wt": "MSKGEELFSGVQPILVELDGCVNGHKFSVSGEGEIDATYGKLTLKFICTTWKLPMPWPCLVTFGSYGVQCFSRYRDHPKQHDFFKSAVPEGYVQERTIFMKDDLLYKTRAEVKFEGLTLVNRIELKGKDFKEDGNILGHKLEYNYNSHCVYPMADWNKNWIKVNSKIRLPIEDGSVILADHYQQNTPIGDQPVLLPENHYLSTQSALSKDPEEKGDLMVLLEFVTAAGITHGMDELYK",  # noqa: E501
#    }
#
#    def __init__(self):
#        """
#        Create GFP landscape.
#
#        Downloads model into `./fluorescence-model` if not already cached there.
#        If interrupted during download, may have to delete this folder and try again.
#        """
#        super().__init__(name="GFP")
#
#        # Download GFP model weights and config info
#        if not os.path.exists("fluorescence-model"):
#            os.mkdir("fluorescence-model")
#
#            # URL for BERT GFP fluorescence model
#            gfp_model_path = "https://fluorescence-model.s3.amazonaws.com/fluorescence_transformer_20-05-25-03-49-06_184764/"  # noqa: E501
#            for file_name in [
#                "args.json",
#                "checkpoint.bin",
#                "config.json",
#                "pytorch_model.bin",
#            ]:
#                print("Downloading", file_name)
#                response = requests.get(gfp_model_path + file_name)
#                with open(f"fluorescence-model/{file_name}", "wb") as f:
#                    f.write(response.content)
#
#        self.tokenizer = tape.TAPETokenizer(vocab="iupac")
#
#        self.device = "cuda" if torch.cuda.is_available() else "cpu"
#        self.model = tape.ProteinBertForValuePrediction.from_pretrained(
#            "fluorescence-model"
#        ).to(self.device)
#
#    def _fitness_function(self, sequences):
#        sequences = np.array(sequences)
#        scores = []
#
#        # Score sequences in batches of size 32
#        for subset in np.array_split(sequences, max(1, len(sequences) // 32)):
#            encoded_seqs = torch.tensor(
#                np.array([self.tokenizer.encode(seq) for seq in subset])
#            ).to(self.device)
#
#            scores.append(
#                self.model(encoded_seqs)[0].detach().cpu().numpy().astype(float).reshape(-1)
#            )
#
#        return np.concatenate(scores)
