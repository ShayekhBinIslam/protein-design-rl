import subprocess
import sys
import time

import ray
import argparse


def run_wrapper(sys_args) -> None:
    # run(conf)
    # cmd = "ls -l"
    # result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    # print(result.stdout)
    # run(conf)
    cmd = "sh update_inference_env.sh"
    result = subprocess.run(cmd, shell=True)
    print(result.stdout)
    print(f"Python path: {sys.executable}")

    run_args = " ".join(sys_args)
    cmd_exp = (
        f"micromamba run -p /home/ray/micromamba/envs/proxy python src/train.py {run_args}"
    )
    print(f"Running command {cmd_exp}")
    result = subprocess.run(cmd_exp, shell=True)

    print(f"Experiment finished with return {result.returncode}")
    return None


def get_args():
    parser = argparse.ArgumentParser(description='Launch')
    parser.add_argument('--lnum_gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--lconda_env', type=str, default='base', help='Conda environment')
    parser.add_argument('--lloop', action="store_true", help='Try running in loop')

    args, unknown = parser.parse_known_args()
    return args, unknown


def run():
    args, unknown = get_args()

    remote_cfg = dict(
        num_gpus=args.lnum_gpus,
        runtime_env={
            "conda": args.lconda_env,
        },
    )

    # Do some setup for restarting experiments
    loop_iter = 0
    while args.lloop or loop_iter == 0:
        start_time = time.time()
        print(f"Starting run remote! start time {start_time}")
        run_remote = ray.remote(**remote_cfg)(run_wrapper)
        exit = ray.get(run_remote.remote(unknown))
        end_time = time.time()
        print(f"Exiting XXX after {end_time - start_time} seconds")
        loop_iter += 1

        if exit == 0:
            break

    print("Process succeeded!")


if __name__ == "__main__":
    run()
