from utils.sampling import Oracle
import pandas as pd
import os
os.environ['TORCH_HOME'] = '/efs/users/riashat_islam_341e494/proxy_seqs/torchhome'


basepath = '/efs/users/riashat_islam_341e494/proxy_seqs/'
# score_file = 'dynappo_trial_run100_flexslog.csv'
# score_file = 'dynappo_trial_run100_mq10k_flexslog.csv'
# score_file = 'ptm_maxdiag_laplace.csv'
# score_file = 'ptm_random_nngp.csv'
score_file = 'mcmc_exp5_63chains.csv'

# skiprows=1
skiprows=0
esm_batch_size = 16

ppo_seqs_file = basepath + score_file
dynappo_sequences = pd.read_csv(filepath_or_buffer=ppo_seqs_file, skiprows=skiprows, index_col=False)
oracle = Oracle(esm_batch_size)
# import ipdb; ipdb.set_trace()
seqs = dynappo_sequences['sequence']

ptms, plddts = oracle.evaluate(seqs)
dynappo_sequences['ptm'] = ptms
dynappo_sequences['plddt'] = plddts
# print(dynappo_sequences)
score_fname = score_file.split(".")[0]
dynappo_sequences.to_csv(f"{basepath}{score_fname}_esmfoldv1_score.csv")

# Diff checker: https://www.editpad.org/tool/diff-checker
