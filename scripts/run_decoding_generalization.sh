#!/bin/bash

# Source conda.sh to make conda available
source /home/xiaoxuan/miniconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate multfs_xuan2

# Define the parameters to permute
# tasks=("dmsloc" "1backloc" "1backctg" "1backobj" "interdmsobjABAB" "interdmslocABBA" "interdmslocABAB" "interdmsctgABAB" "interdmsobjABBA" "interdmsctgABBA")
tasks=("interdmsctgABAB" "interdmsobjABBA" "interdmsctgABBA")
# tasks=("interdmsctgABAB" "interdmsctgABBA")
decoding_features=("category" "location")
reps=(1 2 3 4 5)

# Function to run the Python script with given parameters
run_decoding() {
    local task=$1
    local decoding_feature=$2
    local rep=$3
    
    # Create a unique log file name based on the parameters
    local log_file="logs/${task}_${decoding_feature}_rep${rep}.log"
    
    echo "Running task=$task, decoding_feature=$decoding_feature, rep=$rep"
    echo "Logging output to $log_file"
    
    # Run the Python script and redirect output to the log file
    python3 feature_decoding_network_level_generalization.py --tasks "$task" --decoding_feature "$decoding_feature" --rep "$rep" > "$log_file" 2>&1
}

# Export the run_decoding function so it's available in subshells
export -f run_decoding

# Create a directory to store the log files
mkdir -p logs

# Generate the list of parameter combinations
for task in "${tasks[@]}"; do
    for decoding_feature in "${decoding_features[@]}"; do
        for rep in "${reps[@]}"; do
            echo "$task $decoding_feature $rep"
        done
    done
done | xargs -n 3 -P 4 bash -c 'run_decoding "$0" "$1" "$2"'
