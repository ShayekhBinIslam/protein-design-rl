target=MDM2
python main.py -m \
  target_name=${target} \
  debug=default \
  program=vhh_binder \
  name=${target}_vhh \
  chain.T_max=1.0,10.0 \
  chain.num_chains=1
