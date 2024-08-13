#!/bin/bash
: '
This script is designed to generate and submit jobs on Anyscale.

Usage:
./script_name <job-name> <input-file>

Arguments:
<job-name>       : Name of the job. This also corresponds to the name of the directory where job YAML files will be stored.
<input-file>     : Path to an input template YAML file. This file should contain placeholders (like {{entrypoint}} and {{job_name}}) which will be replaced by actual values.
'

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <job-name> <input-file>"
    exit 1
fi

job_name="$1"
input_file="$2"

# Generate folder for job yamls with that name
# Check if the folder already exists
if [ -d "$job_name" ]; then
    echo "Error: Folder '$job_name' already exists!"
    exit 1
else
    mkdir "$job_name"
    echo "Folder '$job_name' created successfully."
fi

# Here we actually have the grid & py commands

# Initialize an array to keep track of created files
declare -a created_files

counter=0

#for target in PD1; do
for target in CD3e TrkA PD1 PDL1; do
  for plddt in 5; do
    for ptm in 5; do
      for iptm in 5; do
        for tmax in 5; do
#          for interface in 0; do
          for interface in 0 1 2 5; do
# TODO DO NOT INTEND this lines of code
replacement_string=$(cat <<- EOM
python main.py \
target_name=${target} \
program=vhh_binder \
experiment=hparam_sweep \
name=${target}_vhh \
chain.T_max=${tmax} \
chain.num_chains=30 \
program.energy_function_weights=[${ptm},${plddt},1.0,${iptm},${iptm},${interface}] \
wandb.tags=['long','v6','high_plddt']
EOM
)

      let counter+=1

      output_file="$job_name/${input_file}_$counter.yaml"

      # Use sed to replace {{change}} with the provided string
#      sed "s/{{entrypoint}}/$replacement_string/g" "$input_file" > "$output_file"
      sed -e "s/{{entrypoint}}/$replacement_string/g" -e "s/{{job_name}}/$job_name/g" "$input_file" > "$output_file"

      echo "New yaml job @ $output_file for cmd $replacement_string"

      # Add the newly created file to the array
      created_files+=("$output_file")
          done
        done
      done
    done
  done
done


# Prompt the user for confirmation
read -p "Do you want to proceed with starting anyscale jobs? (y/n) " -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Action canceled by user."
    exit 1
fi

# Initialize the counter and get the total count of files
counter=0
count=${#created_files[@]}

# Now, iterate over the created files and run your command on each
for yaml_file in "${created_files[@]}"; do
    let counter+=1
    anyscale job submit "$yaml_file" --name "${job_name}_no_${counter}" --description "Job ${job_name} number ${counter}/${count}"
done
