import os
import h5py
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from helper import decoding
from utils import filter_task_betas, beta_concatenation, df_concatenation
import argparse

# Constants
PARENT_DIR = Path.cwd().parent
GLASSER_ATLAS_PATH = os.path.join(PARENT_DIR, 'data', 'Glasser_LR_Dense64k.dlabel.nii')
NUM_REGIONS = 12
NUM_VOXELS = 64984
EXCLUDED_SESSION = 1000

# Load Glasser atlas and map Cole networks
def load_network_partitions():
    glasser_atlas = nib.load(GLASSER_ATLAS_PATH).get_fdata()[0].astype(int)
    file_path_base = "/home/xiaoxuan/projects/ColeAnticevicNetPartition/"
    network_region_assignment = np.loadtxt(os.path.join(file_path_base, "cortex_parcel_network_assignments.txt"), dtype=int)
    network_voxelwise_assignment = np.zeros((glasser_atlas.shape[0], 1))
    for region in range(1, 361):
        region_idx = np.where(glasser_atlas == region)[0]
        network_voxelwise_assignment[region_idx] = network_region_assignment[region - 1]
    return network_voxelwise_assignment.squeeze()

# Data loading helpers
def get_paths_and_tasks(subj, excluded_session):
    sessions = [f"ses{i}" for i in range(1, 17) if i != excluded_session]
    runs = [f"run-{i:02d}" for i in range(1, 6)]
    datadir = f"{PARENT_DIR}/data/{subj}/glm_betas/{subj}/glm_betas_encoding_delay_full_TR_betas/{subj}/"
    return sessions, runs, datadir

def load_task_data(tasks, sessions, runs, datadir):
    task_betas = {}
    df_conditions = {}
    for task in tasks:
        task_betas[task] = {}
        df_conditions[task] = {}
        for sess in sessions:
            task_betas[task][sess] = {}
            df_conditions[task][sess] = {}
            for run in runs:
                base = f"glmmethod1_{sess}_task-{task}_{run}"
                h5_path = os.path.join(datadir, f"{base}_betas.h5")
                csv_path = os.path.join(datadir, f"{base}.csv")
                if not os.path.exists(h5_path):
                    continue
                with h5py.File(h5_path, 'r') as h5f:
                    task_betas[task][sess][run] = h5f['betas'][:].copy()
                if os.path.exists(csv_path):
                    df_conditions[task][sess][run] = pd.read_csv(csv_path)
    return task_betas, df_conditions

def get_filtered_task_betas(task_betas, df_conditions, first_delay_only=False, second_delay_only=False):
    delay_betas, delay_df = filter_task_betas(task_betas, df_conditions, phase="delay",
                                               first_delay_only=first_delay_only,
                                               second_delay_only=second_delay_only)
    return delay_betas, delay_df

def decode_cross_task(sessions, glasser_atlas, delay_betas, delay_df,
                      tasks, decoding_feature, classifier, feature_normalization, random_state=42):

    def run_decoder(train_task, test_task):
        acc_map = np.zeros((NUM_VOXELS, 1))
        t_map = np.zeros((NUM_VOXELS, 1))
        p_map = np.zeros((NUM_VOXELS, 1))

        for region in range(NUM_REGIONS + 1):
            print(f"Region {region} | Train: {train_task} → Test: {test_task}")
            region_idx = np.where(glasser_atlas == region)[0]
            if len(region_idx) == 0:
                continue
            fold_accuracies = []

            for exclude_ses in sessions:
                train_data = beta_concatenation({train_task: delay_betas[train_task]}, [exclude_ses])[:, region_idx]
                test_data = beta_concatenation({test_task: delay_betas[test_task]}, [exclude_ses])[:, region_idx]
                train_labels = df_concatenation({train_task: delay_df[train_task]}, [exclude_ses])[decoding_feature].to_numpy()
                test_labels = df_concatenation({test_task: delay_df[test_task]}, [exclude_ses])[decoding_feature].to_numpy()

                if len(train_data) == 0 or len(test_data) == 0:
                    continue

                acc = decoding(train_data, test_data, train_labels, test_labels,
                               classifier=classifier, confusion=False,
                               feature_normalization=feature_normalization)
                fold_accuracies.append(np.mean(acc))

            if fold_accuracies:
                fold_accuracies = [a for a in fold_accuracies if not np.isnan(a)]
                if fold_accuracies:
                    acc_map[region_idx] = np.mean(fold_accuracies)
                    t_stat, p_val = ttest_1samp(fold_accuracies, popmean=0.5, alternative='greater')
                    t_map[region_idx] = t_stat
                    p_map[region_idx] = p_val
        return acc_map, t_map, p_map

    results = {}
    for i, train_task in enumerate(tasks):
        for test_task in tasks[i+1:]:
            key = f"{train_task}_to_{test_task}"
            results[key] = run_decoder(train_task, test_task)
    return results

def correct_p_values(p_map):
    flat_p = p_map.flatten()
    _, corrected, _, _ = multipletests(flat_p, alpha=0.05, method='fdr_bh')
    return corrected.reshape(p_map.shape)

def save_results(acc, t_map, p_map, phase_name, tasks, decoding_feature,
                 first_delay_only=False, second_delay_only=False, rep=1):
    suffix = ""
    if first_delay_only:
        suffix = "_first_delay_only"
    elif second_delay_only:
        suffix = "_second_delay_only"

    result_dir = Path(f"{PARENT_DIR}/network_level_results_cross_task/{'_'.join(tasks)}_{decoding_feature}_{phase_name}{suffix}")
    result_dir.mkdir(parents=True, exist_ok=True)

    np.save(result_dir / f'regionwise_acc_rep{rep}.npy', acc)
    np.save(result_dir / f't_stats_map_rep{rep}.npy', t_map)
    np.save(result_dir / f'p_values_map_rep{rep}.npy', p_map)
    np.save(result_dir / f'p_values_map_corrected_rep{rep}.npy', correct_p_values(p_map))

def main(subj="sub-03", tasks=None, decoding_feature="category",
         classifier="distance", feature_normalization=True,
         first_delay_only=False, second_delay_only=False, rep=1, random_state=42):

    if tasks is None:
        tasks = ["dmsloc"]

    sessions, runs, datadir = get_paths_and_tasks(subj, EXCLUDED_SESSION)
    task_betas, df_conditions = load_task_data(tasks, sessions, runs, datadir)
    delay_betas, delay_df = get_filtered_task_betas(task_betas, df_conditions, first_delay_only, second_delay_only)

    glasser_atlas = load_network_partitions()

    results = decode_cross_task(sessions, glasser_atlas, delay_betas, delay_df,
                                tasks, decoding_feature, classifier, feature_normalization, random_state)

    for phase_name, (acc, t_map, p_map) in results.items():
        save_results(acc, t_map, p_map, phase_name, tasks, decoding_feature,
                     first_delay_only=first_delay_only,
                     second_delay_only=second_delay_only, rep=rep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cross-task region-level decoding analysis during delay.")
    parser.add_argument("--tasks", nargs="+", default=["dmsloc"])
    parser.add_argument("--decoding_feature", type=str, default="category")
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--first_delay_only", action="store_true")
    parser.add_argument("--second_delay_only", action="store_true")
    args = parser.parse_args()

    main(tasks=args.tasks,
         decoding_feature=args.decoding_feature,
         rep=args.rep,
         first_delay_only=args.first_delay_only,
         second_delay_only=args.second_delay_only,
         random_state=args.rep)
