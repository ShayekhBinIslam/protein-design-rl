import os
from pathlib import Path

import hydra
import omegaconf

import logging
from async_multi_gpu import GPUProcesses, Infer
# Turn off cuda device logging
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.ERROR)

import flexs
from src.data.atlas_datamodule_v0 import AtlasDataset
import numpy as np
import pandas as pd
from esm.esmfold.v1.misc import batch_encode_sequences, collate_dense_tensors

from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, Subset, DataLoader
from termcolor import colored


def get_proxy(save_path, checkpoint_name):
	os.environ['SAVE_ROOT'] = save_path
	save_path = Path(os.environ['SAVE_ROOT'])
	trainer_config_path = save_path/"config.yaml"
	chekcpoint_path = save_path/f'checkpoints/{checkpoint_name}'
	print(f"trainer_config_path: {trainer_config_path}")
	print(f"chekcpoint_path: {chekcpoint_path}")

	cfg = omegaconf.OmegaConf.load(trainer_config_path)
	model = hydra.utils.instantiate(cfg.model)
	model = type(model).load_from_checkpoint(chekcpoint_path)
	cfg.trainer.enable_progress_bar=False
	trainer = hydra.utils.instantiate(cfg.trainer)
	datamodule = hydra.utils.instantiate(cfg.data)
	return cfg, datamodule, model, trainer

def get_proxy_predictions(sequences, cfg, datamodule, model, trainer):
	if isinstance(sequences, str):
		sequences = [sequences]
		
	df = pd.DataFrame({'sequence': sequences})
	datamodule.data_test = AtlasDataset(cfg=cfg, df=df)
	proxy_results = trainer.predict(model=model, dataloaders=datamodule.test_dataloader())
	return proxy_results


class PTMProxy(flexs.Model):
	def __init__(self, checkpoint_dir, checkpoint_name='epoch_040.ckpt'):
		self.cfg, self.datamodule, self.model, self.trainer = get_proxy(
			checkpoint_dir,
			checkpoint_name,
		)
		self.hparam_tune = False
		super().__init__('PTMProxy')
	
	def _train(self, one_hot_sequences, scores):
		tqdm.write(colored('Not training proxy', 'red'))
		return

	def _fitness_function(self, sequences: str):
		proxy_results, _ = self.evaluate(sequences)
		return proxy_results, None
	
	def evaluate(self, sequences):
		proxy_results = get_proxy_predictions(sequences, self.cfg, self.datamodule, self.model, self.trainer)
		proxy_results = np.concatenate([result['seq_vectorized'].numpy() for result in proxy_results])
		return proxy_results, None


class PTMMCDropoutProxy():
	def __init__(self):
		self.cfg, self.datamodule, self.model, self.trainer = get_proxy(
			'/efs/users/riashat_islam_341e494/proxy_atlas/reg_llm_ptm_mcdropout',
			'epoch_013.ckpt',
		)

	def evaluate(self, sequences):
		proxy_results = get_proxy_predictions(sequences, self.cfg, self.datamodule, self.model, self.trainer)
		proxy_results = np.concatenate([result[0].numpy() for result in proxy_results])
		return proxy_results


class PLDDTMCDropoutProxy():
	def __init__(self):
		self.cfg, self.datamodule, self.model, self.trainer = get_proxy(
			'/efs/users/riashat_islam_341e494/proxy_atlas/reg_llm_plddt_mcdropout',
			'epoch_045.ckpt',
		)

	def evaluate(self, sequences):
		proxy_results = get_proxy_predictions(sequences, self.cfg, self.datamodule, self.model, self.trainer)
		proxy_results = np.concatenate([result[0].numpy() for result in proxy_results])
		return proxy_results


class Oracle(flexs.Model):
	def __init__(self, esm_model, rank, batch_size, use_multi_gpu, bfloat16):
		self.rank = rank
		self.device = torch.device(f"cuda:{rank}")
		self.esm_model = esm_model.eval()

		if bfloat16:
			self.esm_model = self.esm_model.bfloat16()
		
		self.name = 'PTM Oracle'
		self.batch_size = batch_size
		self.hparam_tune = False
		self.use_multi_gpu = use_multi_gpu
		
		if use_multi_gpu:
			self.gpu_processes = GPUProcesses(world_size=torch.cuda.device_count())
			self.gpu_processes.start(self.esm_model)
		
		else:
			self.esm_model = self.esm_model.to(self.device)
			self.inference = Infer(self.esm_model, self.rank, ddp=False)

	def close(self):
		if self.use_multi_gpu:
			self.gpu_processes.end()

	def _train(self, one_hot_sequences, scores):
		# batch_idx is never used in the `training_step` method
		print('Not finetuning proxy')
		return

	def _fitness_function(self, sequences):
		return self.evaluate(sequences)

	def preprocess(self, sequences, residx=None, residue_index_offset=512, chain_linker = "G" * 25):
		if isinstance(sequences, str):
			sequences = [sequences] * torch.cuda.device_count()

		aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
			sequences, residue_index_offset, chain_linker
		)
		if residx is None:
			residx = _residx
		elif not isinstance(residx, torch.Tensor):
			residx = collate_dense_tensors(residx)

		return aatype, mask, residx, linker_mask, chain_index

	def evaluate(self, sequences):
		aatype, mask, residx, linker_mask, chain_index = self.preprocess(sequences)
		dataset = TensorDataset(aatype, mask, residx, linker_mask, chain_index, torch.arange(len(aatype)))

		if self.use_multi_gpu:
			indices = np.array_split(np.arange(len(dataset)), self.gpu_processes.world_size)
			subsets = [Subset(dataset, idx) for idx in indices]

			assert self.gpu_processes.ended is False
			for data_subset, input_queue in zip(subsets, self.gpu_processes.input_queues):
				input_queue.put(data_subset)
			
			parallel_ptm_scores, parallel_plddt_scores, parallel_idxs = [], [], []
			for output_queue in self.gpu_processes.output_queues:
				parallel_ptm_score_k, parallel_plddt_score_k, parallel_idx_k = output_queue.get()
				
				parallel_ptm_scores.append(parallel_ptm_score_k)
				parallel_plddt_scores.append(parallel_plddt_score_k)
				parallel_idxs.append(parallel_idx_k)

			parallel_ptm_scores = torch.cat(parallel_ptm_scores)
			parallel_plddt_scores = torch.cat(parallel_plddt_scores)
			parallel_idxs = torch.cat(parallel_idxs)

			assert (parallel_idxs == torch.arange(len(parallel_idxs))).all()
			return parallel_ptm_scores.numpy(), parallel_plddt_scores.numpy()

		else:
			dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
			ptm_scores, plddt_scores, idxs = self.inference.run(dataloader)
			return ptm_scores.cpu().numpy(), plddt_scores.cpu().numpy()



if __name__ == '__main__':
	proxy = PTMProxy()
	proxy.evaluate(['ACAC'])