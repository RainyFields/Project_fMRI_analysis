import os
import h5py
import pandas as pd

# Config
PARENT_DIR = "/mnt/tempdata/Project_fMRI_analysis_data"
subj = "sub-03"
all_tasks = ["dmsloc", "1backloc", "1backctg", "1backobj",
             "interdmsobjABAB", "interdmslocABBA", "interdmslocABAB",
             "interdmsctgABAB", "interdmsobjABBA", "interdmsctgABBA"]

EXCLUDED_SESSION = 1000
sessions = [f"ses{i}" for i in range(1, 17) if i != EXCLUDED_SESSION]
runs = [f"run-{i:02d}" for i in range(1, 6)]

# Data path
datadir = f"{PARENT_DIR}/data/{subj}/glm_betas/{subj}/glm_betas_encoding_delay_full_TR_betas/{subj}/"

# Store results
task_session_counts = {}
task_trial_counts = {}  # dict of {task: {session: total_trials}}

for task in all_tasks:
    available_sessions = set()
    session_trial_counts = {}

    for sess in sessions:
        session_total_trials = 0
        has_data = False

        for run in runs:
            base = f"glmmethod1_{sess}_task-{task}_{run}"
            h5_path = os.path.join(datadir, f"{base}_betas.h5")
            csv_path = os.path.join(datadir, f"{base}.csv")

            if os.path.exists(h5_path):
                has_data = True

            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    session_total_trials += len(df)
                except Exception as e:
                    print(f"⚠️ Error reading {csv_path}: {e}")

        if has_data:
            available_sessions.add(sess)
            session_trial_counts[sess] = session_total_trials

    task_session_counts[task] = len(available_sessions)
    task_trial_counts[task] = session_trial_counts

# Print refined results
print("\n✅ Session counts per task:")
for task, count in task_session_counts.items():
    print(f"{task:25s}: {count} sessions")

print("\n✅ Trial counts per session (non-zero only):")
for task, sess_trials in task_trial_counts.items():
    print(f"\n🧠 Task: {task}")
    if not sess_trials:
        print("  ⚠️ No sessions with data.")
        continue
    for sess in sorted(sess_trials.keys()):
        print(f"  {sess:6s} → {sess_trials[sess]:3d} trials")
