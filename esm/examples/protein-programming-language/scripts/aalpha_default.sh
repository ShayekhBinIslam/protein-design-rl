#!/bin/bash
# for target in CD3e TrkA MDM2 PD1 PDL1 SARSCoV2
for target in CD3e TrkA MDM2 PD1 PDL1
do
python main.py \
  target_name=${target} \
  program=vhh_binder_with_hotspots2 \
  name=${target}_vhh_hotspot \
  chain.num_chains=50 &
done
