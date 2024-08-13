#!/bin/bash
# for target in CD3e TrkA MDM2 PD1 PDL1 SARSCoV2
#for target in TrkA MDM2
#for target in MDM2
#for target in TrkA
for target in CD3e TrkA PD1 PDL1 MDM2
do
  for plddt in 5 10
  do
    for ptm in 5 10
    do
      for iptm in 5
      do
        for tmax in 5
        do
          python main.py \
            target_name=${target} \
            program=vhh_binder \
            experiment=hparam_sweep \
            name=${target}_vhh \
            chain.T_max=${tmax} \
            wandb.tags=["long","v6","high_plddt"] \
            program.energy_function_weights=[${ptm},${plddt},1.0,${iptm},${iptm},10.0] \
            chain.num_chains=20 &
          sleep 1000
        done
      done
    done
  done
done
