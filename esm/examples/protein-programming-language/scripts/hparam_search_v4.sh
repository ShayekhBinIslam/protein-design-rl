#!/bin/bash
# for target in CD3e TrkA MDM2 PD1 PDL1 SARSCoV2
#for target in TrkA MDM2
#for target in MDM2
for target in CD3e TrkA PD1 PDL1
do
  for iptm in 5 10
  do
    for tmax in 1 10
    do
      python main.py \
        target_name=${target} \
        program=vhh_binder \
        experiment=hparam_sweep \
        name=${target}_vhh \
        chain.T_max=${tmax} \
        wandb.tags=["hparam","v2"] \
        program.energy_function_weights=[1.0,1.0,1.0,${iptm},${iptm},0.0] \
        chain.num_chains=5 &
      sleep 5
    done
  done
done
