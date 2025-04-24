import os
import h5py
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from helper import decoding

warnings.filterwarnings("ignore")

# Constants
PARENT_DIR = "/mnt/tempdata/Project_fMRI_analysis_data"
GLASSER_ATLAS_PATH = os.path.join(PARENT_DIR, 'data', 'Glasser_LR_Dense64k.dlabel.nii')
EXCLUDED_SESSION = 1000

all_tasks = ["1backloc", "1backctg", "1backobj", "dmsloc",
             "interdmsobjABAB", "interdmslocABBA", "interdmslocABAB",
             "interdmsctgABAB", "interdmsobjABBA", "interdmsctgABBA"]
chance_level = 1 / len(all_tasks)

def load_network_partitions():
    glasser_atlas = nib.load(GLASSER_ATLAS_PATH).get_fdata()[0].astype(int)
    file_path_base = "/home/xiaoxuan/projects/ColeAnticevicNetPartition/"
    network_region_assignment = np.loadtxt(os.path.join(file_path_base, "cortex_parcel_network_assignments.txt"), dtype=int)
    network_voxelwise_assignment = np.zeros(glasser_atlas.shape)
    for region in range(1, 361):
        region_idx = np.where(glasser_atlas == region)[0]
        network_voxelwise_assignment[region_idx] = network_region_assignment[region - 1]
    return network_voxelwise_assignment.astype(int)

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

def decode_task_identity_by_network(task_betas, df_conditions,
                                     glasser_atlas, network_voxelwise_assignment,
                                     all_tasks, sessions, classifier="distance",
                                     feature_normalization=True, chance_level=None):

    network_ids = sorted(set(network_voxelwise_assignment[network_voxelwise_assignment > 0]))
    task_indices = {t: i for i, t in enumerate(all_tasks)}
    results = np.full((len(network_ids), len(all_tasks)), np.nan)

    for i, net_id in enumerate(network_ids):
        print(f"\n🔍 Decoding for Network {net_id}")
        region_idx = np.where(network_voxelwise_assignment == net_id)[0]

        X, y, ses_labels = [], [], []
        for j, task in enumerate(all_tasks):
            for sess in sessions:
                for run in df_conditions[task].get(sess, {}):
                    betas = task_betas[task].get(sess, {}).get(run)
                    df = df_conditions[task].get(sess, {}).get(run)
                    if betas is None or df is None:
                        continue
                    X.append(betas[region_idx, :].T)

                    y += [task_indices[task]] * X[-1].shape[0]
                    ses_labels += [sess] * X[-1].shape[0]

            if len(X) == 0:
                print(f"⚠️ No data for Network {net_id}")
                continue
            
            X = np.concatenate(X, axis=0)
            y = np.array(y)
            print(f"X shape: {X.shape}")
            print(f"y shape: {len(y)}")
            ses_labels = np.array(ses_labels)

            unique_sess = sorted(set(ses_labels))
            fold_preds = []

            for leave_out_sess in unique_sess:
                train_idx = ses_labels != leave_out_sess
                test_idx = ses_labels == leave_out_sess
                if sum(test_idx) == 0 or sum(train_idx) == 0:
                    continue

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                acc = decoding(X_train, X_test, y_train, y_test,
                            classifier=classifier,
                            feature_normalization=feature_normalization,
                            confusion=False)
                fold_preds.append(np.mean(acc))
            print(f"len of fold_preds: {len(fold_preds)}")
            if fold_preds:
                results[i, j] = np.mean(fold_preds, axis=0)

    return results, network_ids

def plot_heatmap(results, network_ids, all_tasks, out_path, chance_level=None):
    plt.figure(figsize=(12, 6))
    sns.heatmap(results, xticklabels=all_tasks,
                yticklabels=[f"Net {i}" for i in network_ids],
                cmap="vlag", center=chance_level,
                annot=True, fmt=".2f", cbar_kws={"label": "Accuracy"})
    plt.title("Task Identity Decoding Accuracy per Network")
    plt.xlabel("Task")
    plt.ylabel("Network")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    subj = "sub-03"
    classifier = "distance"
    feature_normalization = True

    sessions, runs, datadir = get_paths_and_tasks(subj, EXCLUDED_SESSION)
    glasser_atlas = nib.load(GLASSER_ATLAS_PATH).get_fdata()[0].astype(int)
    network_voxelwise_assignment = load_network_partitions()

    task_betas, df_conditions = load_task_data(all_tasks, sessions, runs, datadir)

    results, network_ids = decode_task_identity_by_network(
        task_betas, df_conditions,
        glasser_atlas, network_voxelwise_assignment,
        all_tasks, sessions, classifier=classifier,
        feature_normalization=feature_normalization,
        chance_level=chance_level
    )

    output_path = f"{PARENT_DIR}/task_identity_decoding/task_identity_heatmap.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plot_heatmap(results, network_ids, all_tasks, output_path, chance_level=chance_level)

if __name__ == "__main__":
    main()
