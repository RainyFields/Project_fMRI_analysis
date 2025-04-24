# import itertools
# import pandas as pd
# import os

# # Step 1: Task → session → trial count dictionary
# task_trial_counts = {
#     "dmsloc": {f"ses{i}": 32 for i in range(1, 17)},
#     "1backloc": {"ses1": 90, "ses10": 90, "ses13": 90, "ses14": 90, "ses15": 90, "ses16": 90,
#                  "ses2": 90, "ses3": 90, "ses4": 90, "ses7": 90, "ses8": 90, "ses9": 90},
#     "1backctg": {"ses11": 90, "ses5": 90},
#     "1backobj": {"ses12": 90, "ses6": 90},
#     "interdmsobjABAB": {"ses1": 96, "ses10": 96, "ses3": 96, "ses4": 96, "ses9": 96},
#     "interdmslocABBA": {"ses1": 96, "ses10": 96, "ses11": 192, "ses13": 192, "ses14": 192, "ses16": 96,
#                         "ses3": 96, "ses6": 192, "ses7": 96, "ses9": 96},
#     "interdmslocABAB": {"ses10": 96, "ses11": 96, "ses12": 96, "ses14": 96, "ses15": 192, "ses16": 96,
#                         "ses2": 192, "ses4": 96, "ses5": 96, "ses6": 96, "ses7": 96, "ses8": 96},
#     "interdmsctgABAB": {"ses12": 96, "ses13": 96, "ses16": 96, "ses2": 96, "ses4": 96},
#     "interdmsobjABBA": {"ses12": 96, "ses3": 96, "ses5": 96, "ses8": 192},
#     "interdmsctgABBA": {"ses1": 96, "ses15": 96, "ses5": 96, "ses7": 96, "ses9": 96}
# }

# # Step 2: Define how many sessions to subsample per task
# def get_session_requirements(task):
#     return 3 if task == "dmsloc" else 1

# # Step 3: Generate combinations for one (task_a, task_b) pair
# def generate_subsample_combinations(task_trial_counts, task_a, task_b):
#     sessions_a = list(task_trial_counts.get(task_a, {}).keys())
#     sessions_b = list(task_trial_counts.get(task_b, {}).keys())

#     n_a = get_session_requirements(task_a)
#     n_b = get_session_requirements(task_b)

#     if len(sessions_a) < n_a or len(sessions_b) < n_b:
#         return []

#     combinations = []
#     for subset_a in itertools.combinations(sessions_a, n_a):
#         remaining_b = [s for s in sessions_b if s not in subset_a]
#         if len(remaining_b) < n_b:
#             continue
#         for subset_b in itertools.combinations(remaining_b, n_b):
#             combinations.append({
#                 "task_a": task_a,
#                 "task_b": task_b,
#                 "sessions_a": ";".join(subset_a),
#                 "sessions_b": ";".join(subset_b)
#             })
#     return combinations

# # Step 4: Create output folder and generate all CSVs
# output_dir = "/mnt/tempdata/Project_fMRI_analysis_data/network_level_results_cross_task_LOSO/task_pair_session_csvs"
# os.makedirs(output_dir, exist_ok=True)

# all_tasks = list(task_trial_counts.keys())
# generated_files = []

# for task_a in all_tasks:
#     for task_b in all_tasks:  # includes self-pairs
#         combinations = generate_subsample_combinations(task_trial_counts, task_a, task_b)
#         if combinations:
#             df = pd.DataFrame(combinations)
#             filename = f"{task_a}_to_{task_b}_session_pairs.csv"
#             filepath = os.path.join(output_dir, filename)
#             df.to_csv(filepath, index=False)
#             generated_files.append(filename)

# # Step 5: Print a short summary
# print(f"\n✅ Saved {len(generated_files)} CSV files to: {output_dir}")
# print("📄 Sample files:")
# for f in generated_files[:5]:
#     print(" -", f)


# strategy 2

import itertools
import pandas as pd
import os

# Step 1: Task → session → trial count dictionary
task_trial_counts = {
    "dmsloc": {f"ses{i}": 32 for i in range(1, 17)},
    "1backloc": {"ses1": 90, "ses10": 90, "ses13": 90, "ses14": 90, "ses15": 90, "ses16": 90,
                 "ses2": 90, "ses3": 90, "ses4": 90, "ses7": 90, "ses8": 90, "ses9": 90},
    "1backctg": {"ses11": 90, "ses5": 90},
    "1backobj": {"ses12": 90, "ses6": 90},
    "interdmsobjABAB": {"ses1": 96, "ses10": 96, "ses3": 96, "ses4": 96, "ses9": 96},
    "interdmslocABBA": {"ses1": 96, "ses10": 96, "ses11": 192, "ses13": 192, "ses14": 192, "ses16": 96,
                        "ses3": 96, "ses6": 192, "ses7": 96, "ses9": 96},
    "interdmslocABAB": {"ses10": 96, "ses11": 96, "ses12": 96, "ses14": 96, "ses15": 192, "ses16": 96,
                        "ses2": 192, "ses4": 96, "ses5": 96, "ses6": 96, "ses7": 96, "ses8": 96},
    "interdmsctgABAB": {"ses12": 96, "ses13": 96, "ses16": 96, "ses2": 96, "ses4": 96},
    "interdmsobjABBA": {"ses12": 96, "ses3": 96, "ses5": 96, "ses8": 192},
    "interdmsctgABBA": {"ses1": 96, "ses15": 96, "ses5": 96, "ses7": 96, "ses9": 96}
}

# Step 2: Output directory for strategy 2
output_dir = "/mnt/tempdata/Project_fMRI_analysis_data/network_level_results_cross_task_LOSO/task_pair_session_csvs_strategy2"
os.makedirs(output_dir, exist_ok=True)

# Step 3: Generate combinations using new strategy
def generate_strategy2_combinations(task_trial_counts, task_a, task_b):
    sessions_a_all = set(task_trial_counts.get(task_a, {}).keys())
    sessions_b_all = set(task_trial_counts.get(task_b, {}).keys())

    valid_combinations = []

    for test_sess in sessions_b_all:
        train_sessions = list(sessions_a_all - {test_sess})
        if train_sessions:
            valid_combinations.append({
                "task_a": task_a,
                "task_b": task_b,
                "sessions_a": ";".join(sorted(train_sessions)),
                "sessions_b": test_sess
            })

    return valid_combinations

# Step 4: Generate all combinations and save to CSV
all_tasks = list(task_trial_counts.keys())
generated_files = []

for task_a in all_tasks:
    for task_b in all_tasks:  # includes self-pairs
        combinations = generate_strategy2_combinations(task_trial_counts, task_a, task_b)
        if combinations:
            df = pd.DataFrame(combinations)
            filename = f"{task_a}_to_{task_b}_session_pairs.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            generated_files.append(filename)

# Step 5: Print summary
print(f"\n✅ Saved {len(generated_files)} strategy2 CSV files to: {output_dir}")
print("📄 Sample files:")
for f in generated_files[:5]:
    print(" -", f)
