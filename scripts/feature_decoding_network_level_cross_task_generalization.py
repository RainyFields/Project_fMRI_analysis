

import os
import numpy as np
import pandas as pd
import nibabel as nib
from collections import defaultdict
from pathlib import Path
from itertools import product
from utils import filter_task_betas
from helper import decoding

# ========== Config ==========
PARENT_DIR = "/mnt/tempdata/Project_fMRI_analysis_data"
GLASSER_ATLAS_PATH = os.path.join(PARENT_DIR, 'data', 'Glasser_LR_Dense64k.dlabel.nii')
SESSION_PAIR_DIR = os.path.join(PARENT_DIR, "network_level_results_cross_task_LOSO/task_pair_session_csvs_strategy2/")
OUTPUT_DIR = os.path.join(PARENT_DIR, "network_level_results_cross_task_LOSO/decoding_results_strategy2/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_REGIONS = 12
EXCLUDED_SESSION = 1000
DECODING_FEATURE = "category"
CLASSIFIER = "distance"
FEATURE_NORMALIZATION = True
MIN_TRIALS = 5
SUBJ = "sub-03"
# TASKS = ["1backloc", "1backctg", "dmsloc"]
TASKS = ["1backloc", "1backctg", "1backobj", "dmsloc",
         "interdmsobjABAB", "interdmslocABBA", "interdmslocABAB",
         "interdmsctgABAB", "interdmsobjABBA", "interdmsctgABBA"]

# ========== Helper Functions ==========

def load_network_partitions():
    glasser_atlas = nib.load(GLASSER_ATLAS_PATH).get_fdata()[0].astype(int)
    file_path_base = "/home/xiaoxuan/projects/ColeAnticevicNetPartition/"
    network_region_assignment = np.loadtxt(
        os.path.join(file_path_base, "cortex_parcel_network_assignments.txt"), dtype=int)
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
    import h5py
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

def get_filtered_task_betas(task_betas, df_conditions):
    delay_betas, delay_df = filter_task_betas(task_betas, df_conditions, phase="delay")
    return delay_betas, delay_df

def load_selected_session_pairs(task_pairs, max_pairs_per_task=10):
    all_pairs = []
    for task_a, task_b in task_pairs:
        fname = f"{task_a}_to_{task_b}_session_pairs.csv"
        path = os.path.join(SESSION_PAIR_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["task_a"] = task_a
            df["task_b"] = task_b
            if len(df) > max_pairs_per_task:
                df = df.sample(n=max_pairs_per_task, random_state=42)
            all_pairs.append(df)
        else:
            print(f"⚠️ Missing: {fname}")
    return pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()

def df_concatenation(df_dict, session_list):
    all_dfs = []
    for sess in session_list:
        for task in df_dict:
            if sess in df_dict[task]:
                for run in df_dict[task][sess]:
                    all_dfs.append(df_dict[task][sess][run])
            else:
                print(f"⚠️ Session {sess} not in df_dict for task {task}")
    if not all_dfs:
        raise ValueError(f"No condition data found for sessions: {session_list}")
    return pd.concat(all_dfs, axis=0, ignore_index=True)

def beta_concatenation(beta_dict, session_list):
    all_betas = []
    for sess in session_list:
        for task in beta_dict:
            if sess in beta_dict[task]:
                for run in beta_dict[task][sess]:
                    all_betas.append(beta_dict[task][sess][run])
            else:
                print(f"⚠️ Session {sess} not in beta_dict for task {task}")
    if not all_betas:
        raise ValueError(f"No beta data found for sessions: {session_list}")
    return np.concatenate(all_betas, axis=1)

def decode_all_pairs_grouped(pair_df, delay_betas, delay_df, glasser_atlas, output_dir):
    grouped = pair_df.groupby(["task_a", "task_b"])
    for (task_a, task_b), df_group in grouped:
        print(f"\n🧠 Decoding {task_a} → {task_b} across {len(df_group)} session pairs")

        pair_results = []

        for idx, row in df_group.iterrows():
            sessions_a = row['sessions_a'].split(";")
            sessions_b = row['sessions_b'].split(";")

            for region in range(NUM_REGIONS + 1):
                region_idx = np.where(glasser_atlas == region)[0]
                if len(region_idx) == 0:
                    continue

                try:
                    train_X = beta_concatenation({task_a: delay_betas[task_a]}, sessions_a)[region_idx, :].T
                    train_y = df_concatenation({task_a: delay_df[task_a]}, sessions_a)[DECODING_FEATURE].to_numpy()

                    test_X = beta_concatenation({task_b: delay_betas[task_b]}, sessions_b)[region_idx, :].T
                    test_y = df_concatenation({task_b: delay_df[task_b]}, sessions_b)[DECODING_FEATURE].to_numpy()

                    if len(train_X) < MIN_TRIALS or len(test_X) < MIN_TRIALS:
                        continue

                    acc = decoding(train_X, test_X, train_y, test_y,
                                   classifier=CLASSIFIER, confusion=False,
                                   feature_normalization=FEATURE_NORMALIZATION)
                    mean_acc = np.mean(acc)

                    pair_results.append({
                        "region": region,
                        "accuracy": mean_acc,
                        "train_trials": len(train_X),
                        "test_trials": len(test_X),
                        "sessions_a": ";".join(sessions_a),
                        "sessions_b": ";".join(sessions_b)
                    })

                except Exception as e:
                    print(f"⚠️ Skipping {task_a} → {task_b} | Region {region} due to error: {e}")
                    continue

        # Save results immediately for this task pair
        if pair_results:
            df_results = pd.DataFrame(pair_results)
            fname = f"{task_a}_to_{task_b}_accuracies.csv"
            df_results.to_csv(os.path.join(output_dir, fname), index=False)
            print(f"✅ Saved: {fname}")
        else:
            print(f"⚠️ No results to save for {task_a} → {task_b}")


def save_accuracies(accuracies, output_dir):
    for task_a in accuracies:
        for task_b in accuracies[task_a]:
            df = pd.DataFrame(accuracies[task_a][task_b])
            if not df.empty:
                filename = f"{task_a}_to_{task_b}_accuracies.csv"
                filepath = os.path.join(output_dir, filename)
                df.to_csv(filepath, index=False)
                print(f"✅ Saved: {filepath}")
            else:
                print(f"⚠️ No results to save for {task_a} → {task_b}")

# ========== Main ==========

def main():
    sessions, runs, datadir = get_paths_and_tasks(SUBJ, EXCLUDED_SESSION)
    task_betas, df_conditions = load_task_data(TASKS, sessions, runs, datadir)
    delay_betas, delay_df = get_filtered_task_betas(task_betas, df_conditions)
    task_pairs = list(product(TASKS, TASKS))
    pair_df = load_selected_session_pairs(task_pairs, max_pairs_per_task=10)
    glasser_atlas = load_network_partitions()
    decode_all_pairs_grouped(pair_df, delay_betas, delay_df, glasser_atlas, OUTPUT_DIR)
    print("✅ Finished decoding all task pairs.")

if __name__ == "__main__":
    main()
