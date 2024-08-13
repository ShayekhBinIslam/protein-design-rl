#!/bin/bash
# Remove SARSCoV2 for now as it does not work with esmfold
# for target in CD3e TrkA MDM2 PD1 PDL1 SARSCoV2
for target in CD3e TrkA MDM2 PD1 PDL1
do
python main.py \
  target_name=${target} \
  program=vhh_binder \
  name=${target}_vhh \
  chain.num_chains=5 &
done
