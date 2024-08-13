import pandas as pd
import matplotlib.pyplot as plt
import time


basepath = '/efs/users/riashat_islam_341e494/proxy_seqs/'
# score_file = 'dynappo_trial_run100_flexslog_esmfoldv1_score.csv'
score_file = 'dynappo_trial_run100_mq10k_flexslog_esmfoldv1_score.csv'

ppo_seqs_file = basepath + score_file
df = pd.read_csv(ppo_seqs_file)
df['proxy_ptm'] = df['true_score']


fname = score_file.split('.')[0]
timestr = time.strftime("%Y%m%d-%H%M%S")
timestr += '_' + fname


# plt.clf()
# df.boxplot('ptm', 'round')
# plt.savefig(f'./plots/{timestr}_ptm.jpg')

# plt.clf()
# df.boxplot('plddt', 'round')
# plt.savefig(f'./plots/{timestr}_plddt.jpg')

# plt.clf()
# df.boxplot('true_score', 'round')
# plt.savefig(f'./plots/{timestr}_proxy_ptm.jpg')


# Create a figure and axis object
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Boxplot for 'ptm'
df.boxplot('ptm', 'round', ax=axes[0])
axes[0].set_title('Boxplot for true ptm')

# Boxplot for 'plddt'
df.boxplot('plddt', 'round', ax=axes[1])
axes[1].set_title('Boxplot for true plddt')

# Boxplot for 'true_score'
df.boxplot('proxy_ptm', 'round', ax=axes[2])
axes[2].set_title('Boxplot for proxy ptm')

# Adjust layout to prevent clipping of titles
plt.tight_layout()

# Save the figure
plt.savefig(f'./plots/{timestr}_boxplots.jpg')

# cd /efs/users/riashat_islam_341e494/proxy_seqs
'''
import pandas as pd
df = pd.read_csv('ptm_random_nngp_esmfoldv1_score.csv')
df.boxplot('ptm', 'round')
plt.savefig('al.jpg'j)
plt.savefig('al.jpg')
import matplotlib.pyplot as plt
plt.savefig('al.jpg')
plt.clf()
df['ptm'].groupby('round')
df.groupby('round').mean()
df.groupby('round').float.mean()
df.groupby('round')['ptm'].mean()
df.groupby('round')['ptm'].mean().plot()
plt.savefig('al_mean.jpg')
df.groupby('round')['ptm'].max().plot()
plt.savefig('al_max.jpg')
df.groupby('round')['plddt'].mean().plot()
df.groupby('round')['plddt'].max().plot()
plt.savefig('/home/ray/default/al_random_nngp_plddt_mean_max.jpg')
df.plot('round', 'ptm')
plt.savefig('/home/ray/default/mcmc_ptm.jpg')

'''

