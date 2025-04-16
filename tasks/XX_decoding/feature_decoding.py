import os
import h5py
import json
import time
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from helper import decoding
from utils import filter_task_betas, beta_concatenation, make_brain_surface_plot, df_concatenation
import argparse  # Import argparse for command-line argument parsing

# Constants
PARENT_DIR = '/Users/lucasgomez/Desktop/Neuro/Bashivan/Hackthon_WM_fMRI/data'
GLASSER_ATLAS_PATH = os.path.join(PARENT_DIR, 'Glasser_LR_Dense64k.dlabel.nii')
NUM_REGIONS = 360  # Number of regions in the Glasser atlas
NUM_VOXELS = 64984  # Number of voxels
EXCLUDED_SESSION = 1000  # For sanity check purposes

# Load glasser atlas
def load_glasser_atlas():
    glasser_atlas = nib.load(GLASSER_ATLAS_PATH).get_fdata()[0].astype(int)
    print("Glasser Atlas shape:", glasser_atlas.shape)
    return glasser_atlas

# Initialize paths and tasks
def get_paths_and_tasks(subj, excluded_session):
    # Set subject-specific parameters
    sessions = [f"ses{i}" for i in range(1, 17) if i != excluded_session]
    runs = [f"run-{i:02d}" for i in range(1, 6)]
    datadir = f"{PARENT_DIR}/data/glm_betas_encoding_delay_full_TR_betas/{subj}/"
    
    return sessions, runs, datadir

# Load betas and conditions for each task-session-run combination
def load_task_data(tasks, sessions, runs, datadir):
    task_betas = {}
    df_conditions = {}

    for task in tasks:
        print(f"Processing task: {task}")
        task_betas[task] = {}
        df_conditions[task] = {}

        for sess in sessions:
            task_betas[task][sess] = {}
            df_conditions[task][sess] = {}

            for run in runs:
                base_filename = f"glmmethod1_{sess}_task-{task}_{run}"
                h5_file_path = os.path.join(datadir, f"{base_filename}_betas.h5")
                csv_file_path = os.path.join(datadir, f"{base_filename}.csv")

                if not os.path.exists(h5_file_path):
                    print(f"Warning: HDF5 file not found: {h5_file_path}")
                    continue

                print(f"------------Loading: {h5_file_path}-------------")
                try:
                    with h5py.File(h5_file_path, 'r') as h5f:
                        task_betas[task][sess][run] = h5f['betas'][:].copy()

                    if os.path.exists(csv_file_path):
                        df_conditions[task][sess][run] = pd.read_csv(csv_file_path)
                    else:
                        print(f"Warning: CSV file not found: {csv_file_path}")

                except (OSError, KeyError, ValueError) as e:
                    print(f"Error processing {h5_file_path}: {e}")
    return task_betas, df_conditions

# Filter and concatenate task betas for encoding and delay phases
def get_filtered_task_betas(task_betas, df_conditions, first_delay_only=False, second_delay_only=False, third_delay_only=False):
    delay_task_betas, delay_task_df = filter_task_betas(task_betas, df_conditions, phase="delay", first_delay_only=first_delay_only, second_delay_only=second_delay_only, third_delay_only=third_delay_only)
    encoding_task_betas, encoding_task_df = filter_task_betas(task_betas, df_conditions, phase="encoding", first_delay_only=first_delay_only, second_delay_only=second_delay_only, third_delay_only = third_delay_only)
    return delay_task_betas, delay_task_df, encoding_task_betas, encoding_task_df

# Perform decoding on regions and save results
def decode_regions(sessions, glasser_atlas, delay_task_betas, delay_task_df, encoding_task_betas, encoding_task_df, tasks, decoding_feature, phase, classifier, feature_normalization, random_state = 42):
    regionwise_acc = np.zeros((NUM_VOXELS, 1))
    t_stats_map = np.zeros((NUM_VOXELS, 1))
    p_values_map = np.zeros((NUM_VOXELS, 1))

    unique_regions = np.unique(glasser_atlas)
    # print("Unique regions in Glasser atlas:", unique_regions)

    for region in range(1, NUM_REGIONS + 1):
        if region not in unique_regions:
            print(f"Skipping region {region} as it does not exist in the atlas.")
            continue

        print(f"Processing region: {region}")
        region_idx = np.where(glasser_atlas == region)[0]
        if len(region_idx) == 0:
            print(f"Skipping region {region} due to no voxel assignment.")
            continue

        fold_accuracies = []
        for exclude_ses in sessions:

            all_delay_betas = beta_concatenation(delay_task_betas, [exclude_ses])
            all_encoding_betas = beta_concatenation(encoding_task_betas, [exclude_ses])
            all_df = df_concatenation(delay_task_df, [exclude_ses])

            data = all_delay_betas[:, region_idx] if phase == "delay" else all_encoding_betas[:, region_idx]
            labels = all_df[decoding_feature].to_numpy()
            
            if len(data) == 0 or len(labels) == 0:
                print(f"Skipping region {region} due to empty data or labels.")
                continue

            unique_labels, label_counts = np.unique(labels, return_counts=True)
            if len(unique_labels) < 2:
                print(f"Skipping region {region} due to lack of label diversity.")
                continue

            n_splits = min(5, min(label_counts))
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            
            for train_index, test_index in skf.split(data, labels):
                trainset, testset = data[train_index], data[test_index]
                trainlabels, testlabels = np.array(labels)[train_index], np.array(labels)[test_index]
                
                accuracy = decoding(trainset, testset, trainlabels, testlabels, classifier=classifier, confusion=False, feature_normalization=feature_normalization)
                fold_accuracies.append(np.mean(accuracy))
        
        if len(fold_accuracies) == 0:
            print(f"Skipping region {region} due to lack of valid folds.")
            continue
        
        fold_accuracies = [acc for acc in fold_accuracies if not np.isnan(acc)]
        if len(fold_accuracies) == 0:
            print(f"Skipping region {region} due to NaN values in accuracy.")
            continue

        regionwise_acc[region_idx] = np.mean(fold_accuracies)
        t_stat, p_val = ttest_1samp(fold_accuracies, popmean=0.5, alternative='greater')
        t_stats_map[region_idx] = t_stat
        p_values_map[region_idx] = p_val

    return regionwise_acc, t_stats_map, p_values_map

# Correct p-values using FDR
def correct_p_values(p_values_map):
    p_values_list = p_values_map.flatten()
    _, p_values_corrected, _, _ = multipletests(p_values_list, alpha=0.05, method='fdr_bh')
    p_values_map_corrected = p_values_corrected.reshape(p_values_map.shape)
    return p_values_map_corrected

# Save results
def save_results(regionwise_acc, t_stats_map, p_values_map, tasks, decoding_feature, phase, first_delay_only=False, second_delay_only = False, third_delay_only = False, rep=1):
    if first_delay_only:
        results_dir = Path(f"{PARENT_DIR}/results/{'_'.join(tasks)}_{decoding_feature}_{phase}_first_delay_only")
    elif second_delay_only:
        results_dir = Path(f"{PARENT_DIR}/results/{'_'.join(tasks)}_{decoding_feature}_{phase}_second_delay_only")
    elif third_delay_only:
        results_dir = Path(f"{PARENT_DIR}/results/{'_'.join(tasks)}_{decoding_feature}_{phase}_third_delay_only")
    else:
        results_dir = Path(f"{PARENT_DIR}/results/{'_'.join(tasks)}_{decoding_feature}_{phase}")
        
    print(f"results_dir: {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(results_dir, f'regionwise_acc_rep{rep}.npy'), regionwise_acc)
    np.save(os.path.join(results_dir, f't_stats_map_rep{rep}.npy'), t_stats_map)
    np.save(os.path.join(results_dir, f'p_values_map_rep{rep}.npy'), p_values_map)

    # Optionally, save the corrected p-values
    p_values_map_corrected = correct_p_values(p_values_map)
    np.save(os.path.join(results_dir, f'p_values_map_corrected_rep{rep}.npy'), p_values_map_corrected)
    print("all files properly saved!!!!!!!")
# Main function
def main(subj="sub-03", tasks=None, decoding_feature="category", 
         phase="delay", classifier="distance", feature_normalization=True,
         first_delay_only=False, rep=1, second_delay_only=False, random_state = 42, third_delay_only=False):
    
    if tasks is None:
        tasks = ['ctxcol', 'ctxlco']  # Default tasks
    
    # Load data
    sessions, runs, datadir = get_paths_and_tasks(subj, EXCLUDED_SESSION)
    task_betas, df_conditions = load_task_data(tasks, sessions, runs, datadir)
    
    # Filter and concatenate task betas
    delay_task_betas, delay_task_df, encoding_task_betas, encoding_task_df = get_filtered_task_betas(task_betas, df_conditions, first_delay_only=first_delay_only, second_delay_only=second_delay_only, third_delay_only=third_delay_only)
    
    # Load Glasser Atlas
    glasser_atlas = load_glasser_atlas()
    
    # Perform regionwise decoding
    regionwise_acc, t_stats_map, p_values_map = decode_regions(
        sessions, glasser_atlas, delay_task_betas, delay_task_df, encoding_task_betas, encoding_task_df,
        tasks, decoding_feature, phase, classifier, feature_normalization, random_state = random_state
    )
    
    # Save results
    save_results(regionwise_acc, t_stats_map, p_values_map, tasks, decoding_feature, phase, first_delay_only=first_delay_only, second_delay_only=second_delay_only, third_delay_only = third_delay_only, rep=rep)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run feature decoding with specified parameters.")
    parser.add_argument("--tasks", nargs="+", default=["dmsloc"], help="List of tasks to process.")
    parser.add_argument("--decoding_feature", type=str, default="location", help="Decoding feature (e.g., 'category' or 'location').")
    parser.add_argument("--phase", type=str, default="delay", help="Phase to process (e.g., 'delay' or 'encoding').")
    parser.add_argument("--rep", type=int, default=1, help="Repetition number.")
    parser.add_argument("--first_delay_only", action="store_true", help="Whether to process only the first delay.")
    parser.add_argument("--second_delay_only", action="store_true", help="Whether to process exclude the first delay.")
    parser.add_argument("--third_delay_only", action="store_true", help="Whether to process exclude the first two delay.")
    
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(tasks=args.tasks, 
         decoding_feature=args.decoding_feature,
         phase=args.phase,
         rep=args.rep,
         first_delay_only=args.first_delay_only,
         second_delay_only=args.second_delay_only,
         third_delay_only=args.third_delay_only,
         random_state=args.rep,
         )


