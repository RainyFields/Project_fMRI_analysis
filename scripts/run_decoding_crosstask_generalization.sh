# no need to use! current python script allow for direct execution from the terminal

#!/bin/bash

# Source conda.sh to make conda available
source /home/xiaoxuan/miniconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate multfs_xuan2

# Define the parameters
all_tasks=("dmsloc" "1backloc" "1backctg" "1backobj" "interdmsobjABAB" "interdmslocABBA" "interdmslocABAB" "interdmsctgABAB" "interdmsobjABBA" "interdmsctgABBA")
decoding_features=("category" "location")
reps=(1 2 3 4 5)

# Define your script name
SCRIPT="/home/xiaoxuan/projects/202406_fMRI/scripts/feature_decoding_network_level_cross_task_generalization.py"


# Ensure logs directory exists
mkdir -p logs

# Generate jobs and run in parallel
for decoding_feature in "${decoding_features[@]}"; do
    for rep in "${reps[@]}"; do
        echo "$decoding_feature $rep"
    done
done | xargs -n 2 -P 4 bash -c '
decoding_feature="$0"
rep="$1"
log_file="logs/cross_task_${decoding_feature}_rep${rep}.log"
echo "Running decoding_feature=$decoding_feature rep=$rep"
echo "Logging to $log_file"

python3 feature_decoding_network_level_cross_task_generalization.py \
  --tasks dmsloc 1backloc 1backctg 1backobj interdmsobjABAB interdmslocABBA interdmslocABAB interdmsctgABAB interdmsobjABBA interdmsctgABBA \
  --decoding_feature "$decoding_feature" \
  --rep "$rep" > "$log_file" 2>&1
'