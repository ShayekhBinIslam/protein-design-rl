import esm
import torch
import ray

@ray.remote(num_gpus=1.0)
class LLMContactPredictor:
        def __init__(self) -> None:
            self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            model.to(self.device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            self.model = model
            self.batch_converter = alphabet.get_batch_converter()
        def get_contacts(self, sequence,linker_length):
            # add linker
            first_length = sequence.find(":")
            complex_length = len(sequence)-1
            sequence_with_liker = sequence[:first_length] + "G" * linker_length + sequence[1+first_length:]
            # predict llm contacts
            _, _, batch_tokens = self.batch_converter([(0,sequence_with_liker)])
            batch_tokens = batch_tokens.to(self.device)
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
            contacts = results["contacts"][0,:,:]
            # average across diagonal to make symmetric
            contacts = (contacts + contacts.transpose (0, 1)) / 2
            # mask linker in the contact map
            mask = torch.ones([complex_length + linker_length, complex_length + linker_length],dtype=torch.bool)
            mask[first_length:first_length+linker_length,:]=0
            mask[:,first_length:first_length+linker_length]=0
            contacts = contacts[mask].reshape([complex_length,complex_length])
            return contacts.to("cpu")