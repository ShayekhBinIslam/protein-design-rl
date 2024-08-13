import pandas as pd
import matplotlib.pyplot as plt
import time


basepath = '/efs/users/riashat_islam_341e494/proxy_seqs/'
# score_file = 'dynappo_trial_run100_flexslog_esmfoldv1_score.csv'
rand = 'ptm_random_nngp_esmfoldv1_score.csv'
maxdiag = 'ptm_maxdiag_laplace_esmfoldv1_score.csv'
# mcmc = 'mcmc_exp5_esmfoldv1_score.csv'

df_rand = pd.read_csv(basepath + rand)
df_maxdiag = pd.read_csv(basepath + maxdiag)
# df_mcmc = pd.read_csv(basepath + mcmc)
# feature='ptm'

def plot_compare(feature='ptm'):
    plt.clf()
    df_rand.groupby('round')[feature].mean().plot(label='Random (mean)')
    df_rand.groupby('round')[feature].max().plot(label='Random (max)')

    df_maxdiag.groupby('round')[feature].mean().plot(label='MaxDiag (mean)')
    df_maxdiag.groupby('round')[feature].max().plot(label='MaxDiag (max)')

    # df_mcmc.plot('round', 'ptm')

    plt.legend()
    plt.title(feature)
    plt.savefig(f'/home/ray/default/al_random_vs_maxdet_nngp_{feature}_mean_max.jpg')

plot_compare('plddt')
plot_compare('ptm')



