#!/bin/bash

# Source conda.sh to make conda available
source /home/xiaoxuan/miniconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate multfs_xuan2

# Define the parameters to permute
# tasks=("dmsloc" "1backloc" "1backctg" "1backobj" "interdmsobjABAB" "interdmslocABBA" "interdmslocABAB" "interdmsctgABAB" "interdmsobjABBA" "interdmsctgABBA")
tasks=("interdmsctgABAB" "interdmsctgABBA")
decoding_features=("category" "location")
phases=("delay" "encoding")
# phases=("delay")
reps=(1)

# Function to run the Python script with given parameters
run_decoding() {
    local task=$1
    local decoding_feature=$2
    local phase=$3
    local rep=$4
    
    # Create a unique log file name based on the parameters
    local log_file="logs/${task}_${decoding_feature}_${phase}_${rep}.log"
    
    echo "Running task=$task, decoding_feature=$decoding_feature, phase=$phase, rep=$rep"
    echo "Logging output to $log_file"
    
    # Run the Python script and redirect output to the log file
    python3 feature_decoding.py --tasks "$task" --decoding_feature "$decoding_feature" --phase "$phase" --rep "$rep" > "$log_file" 2>&1
}

# Export the run_decoding function so it's available in subshells
export -f run_decoding

# Create a directory to store the log files
mkdir -p logs

# Generate the list of parameter combinations
for task in "${tasks[@]}"; do
    for decoding_feature in "${decoding_features[@]}"; do
        for phase in "${phases[@]}"; do
            for rep in "${reps[@]}"; do
                echo "$task $decoding_feature $phase $rep"
            done
        done
    done
done | xargs -n 4 -P 8 bash -c 'run_decoding "$0" "$1" "$2" "$3"'

