name: "pablo_proxy"
compute_config: "esm-min-cost-v4-zonea" # You may specify `compute_config_id` or `cloud` instead
cluster_env: pablo-proxy:8 # You may specify `build_id` instead
runtime_env:
  working_dir: "/home/ray/default/dockq-proxy"
entrypoint: "micromamba run -p /home/ray/micromamba/envs/proxy python src/train.py trainer=gpu experiment=best"
max_retries: 0