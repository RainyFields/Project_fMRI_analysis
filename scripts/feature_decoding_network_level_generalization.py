import os
import h5py
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import warnings

from helper import decoding
from utils import filter_task_betas, beta_concatenation, df_concatenation

warnings.filterwarnings("ignore")

# Constants
PARENT_DIR = "/mnt/tempdata/Project_fMRI_analysis_data"
GLASSER_ATLAS_PATH = os.path.join(PARENT_DIR, 'data', 'Glasser_LR_Dense64k.dlabel.nii')
NUM_REGIONS = 12
NUM_VOXELS = 64984
EXCLUDED_SESSION = 1000


all_tasks = ["1backloc", "1backctg", "1backobj", "dmsloc",
             "interdmsobjABAB", "interdmslocABBA", "interdmslocABAB",
             "interdmsctgABAB", "interdmsobjABBA", "interdmsctgABBA"]

# decoding_features = ["category", "location"]
decoding_features = ['location']
reps = [1, 2, 5]

def load_network_partitions():
    glasser_atlas = nib.load(GLASSER_ATLAS_PATH).get_fdata()[0].astype(int)
    file_path_base = "/home/xiaoxuan/projects/ColeAnticevicNetPartition/"
    network_region_assignment = np.loadtxt(os.path.join(file_path_base, "cortex_parcel_network_assignments.txt"), dtype=int)
    network_voxelwise_assignment = np.zeros((glasser_atlas.shape[0], 1))
    for region in range(1, 361):
        region_idx = np.where(glasser_atlas == region)[0]
        network_voxelwise_assignment[region_idx] = network_region_assignment[region - 1]
    return network_voxelwise_assignment.squeeze()

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

def decode_phase_crossval(task_sessions, glasser_atlas,
                          betas_dict, df_dict,
                          task, decoding_feature, classifier,
                          feature_normalization,
                          train_phase="delay", test_phase="delay",
                          first_delay_only=False, second_delay_only=False,
                          random_state=42):

    acc_map = np.zeros((NUM_VOXELS, 1))
    t_map = np.zeros((NUM_VOXELS, 1))
    p_map = np.zeros((NUM_VOXELS, 1))

    for region in range(NUM_REGIONS + 1):
        print(f"Region {region} | {train_phase}→{test_phase} | Task: {task}")
        region_idx = np.where(glasser_atlas == region)[0]
        if len(region_idx) == 0:
            continue

        fold_accuracies = []
        for exclude_ses in task_sessions:
            train_sessions = [s for s in task_sessions if s != exclude_ses]
            if len(train_sessions) == 0:
                continue
            try:
                train_betas, train_df = filter_task_betas(betas_dict, df_dict, phase=train_phase,
                                                          first_delay_only=first_delay_only,
                                                          second_delay_only=second_delay_only)
                test_betas, test_df = filter_task_betas(betas_dict, df_dict, phase=test_phase,
                                                        first_delay_only=first_delay_only,
                                                        second_delay_only=second_delay_only)

                train_data = beta_concatenation({task: train_betas[task]}, train_sessions)[:, region_idx]
                test_data = beta_concatenation({task: test_betas[task]}, [exclude_ses])[:, region_idx]
                train_labels = df_concatenation({task: train_df[task]}, train_sessions)[decoding_feature].to_numpy()
                test_labels = df_concatenation({task: test_df[task]}, [exclude_ses])[decoding_feature].to_numpy()

                if len(train_data) == 0 or len(test_data) == 0:
                    continue

                acc = decoding(train_data, test_data, train_labels, test_labels,
                               classifier=classifier, confusion=False,
                               feature_normalization=feature_normalization)
                fold_accuracies.append(np.mean(acc))
            except Exception as e:
                print(f"⚠️ Region {region}, fold {exclude_ses}, error: {e}")
                continue

        if fold_accuracies:
            fold_accuracies = [a for a in fold_accuracies if not np.isnan(a)]
            if fold_accuracies:
                acc_map[region_idx] = np.mean(fold_accuracies)
                t_stat, p_val = ttest_1samp(fold_accuracies, popmean=0.5, alternative='greater')
                t_map[region_idx] = t_stat
                p_map[region_idx] = p_val

    return acc_map, t_map, p_map

def correct_p_values(p_map):
    flat_p = p_map.flatten()
    _, corrected, _, _ = multipletests(flat_p, alpha=0.05, method='fdr_bh')
    return corrected.reshape(p_map.shape)

def main():
    subj = "sub-03"
    classifier = "distance"
    feature_normalization = True
    first_delay_only = False
    second_delay_only = False

    sessions, runs, datadir = get_paths_and_tasks(subj, EXCLUDED_SESSION)
    glasser_atlas = load_network_partitions()

    for decoding_feature in decoding_features:
        print(f"\n==> Decoding Feature: {decoding_feature}")
        for rep in reps:
            print(f"\n---- Repetition {rep} ----")

            task_betas, df_conditions = load_task_data(all_tasks, sessions, runs, datadir)
            phase_pairs = [("encoding", "encoding"), ("encoding", "delay"), ("delay", "encoding"), ("delay", "delay"), ]

            for train_phase, test_phase in phase_pairs:
                print(f"\n### Phase: Train={train_phase}, Test={test_phase}")

                for task in tqdm(all_tasks, desc=f"[{decoding_feature} | rep {rep} | {train_phase}->{test_phase}]", ncols=100):
                    task_sessions = list(df_conditions[task].keys())
                    if len(task_sessions) == 0:
                        print(f"⚠️ Skipping {task}: no sessions available.")
                        continue

                    acc, t_map, p_map = decode_phase_crossval(
                        task_sessions, glasser_atlas, task_betas, df_conditions,
                        task, decoding_feature, classifier,
                        feature_normalization,
                        train_phase=train_phase, test_phase=test_phase,
                        first_delay_only=first_delay_only,
                        second_delay_only=second_delay_only,
                        random_state=rep
                    )

                    suffix = f"{train_phase}_to_{test_phase}"
                    result_dir = Path(f"{PARENT_DIR}/network_level_results_same_task_LOSO/{task}_{decoding_feature}_{suffix}")
                    result_dir.mkdir(parents=True, exist_ok=True)

                    np.save(result_dir / f'regionwise_acc_rep{rep}.npy', acc)
                    np.save(result_dir / f't_stats_map_rep{rep}.npy', t_map)
                    np.save(result_dir / f'p_values_map_rep{rep}.npy', p_map)
                    np.save(result_dir / f'p_values_map_corrected_rep{rep}.npy', correct_p_values(p_map))

if __name__ == "__main__":
    main()
