# System imports
import os
import h5py
import glob

# Data imports
import numpy as np
import pandas as pd
import torch

# Other imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sentence_transformers import util


class Brain:
    """
    desc: Custom class to load and process whole-brain fMRI data.
    args:
        basedir: str
            Base directory path.
        datadir: str
            Data directory path.
        betasdir: str
            Betas directory path.
        subj: str
            Subject ID.
        runs: list of str
            List of runs.
        sessions: list of str
            List of sessions.
        tasks: list of str
            List of tasks.
        glasser_atlas: numpy array
            Glasser atlas data.
    """
    def __init__(self, basedir, datadir, betasdir, subj, runs, sessions, tasks, glasser_atlas):
        # Base directory paths
        self.basedir = basedir
        self.datadir = datadir
        self.betasdir = betasdir
        
        # Meta info
        self.subj = subj
        self.runs = runs
        self.sessions = sessions
        self.tasks = tasks

        # Region info
        self.glasser_atlas = glasser_atlas

        # Data dictionaries
        self.task_betas = {}
        self.df_conditions = {}
        self.mapped_atlas = {}

    def load_betas(self, normalized=True, correct=True):
        """
        desc: Load betas from h5 files.
        args:
            normalized: bool
                If True, load normalized betas.
            correct: bool
                If True, load correct betas.
        """
        normalized_str = 'normalized_' if normalized else ''
        correct_str = 'correcttrial_' if correct else ''

        for task in self.tasks:
            for sess in self.sessions:
                for run in self.runs:
                    # Find a matching h5 beta file
                    pattern = os.path.join(
                        self.betasdir,
                        f"{normalized_str}{correct_str}glmmethod1_{sess}_task-{task}_{run}_betas.h5"
                    )
                    matching_h5_files = glob.glob(pattern)

                    if not matching_h5_files:
                        continue  # No matching file

                    h5_path = matching_h5_files[0]  # Assuming one match per pattern

                    with h5py.File(h5_path, 'r') as h5f:
                        betas = h5f['betas'][:].copy()

                    # CSV path (assumes similar structure, minus '_betas.h5')
                    csv_path = os.path.join(
                        self.betasdir,
                        f"{correct_str}glmmethod1_{sess}_task-{task}_{run}.csv"
                    )

                    if not os.path.exists(csv_path):
                        continue # No matching CSV file

                    df = pd.read_csv(csv_path)

                    # Set nested dictionary structure safely
                    self.task_betas.setdefault(task, {}).setdefault(sess, {})[run] = betas
                    self.df_conditions.setdefault(task, {}).setdefault(sess, {})[run] = df
        print('Loaded betas for ' + self.subj)
 

    def filter_betas(self, method='mean'):
        """
        desc: Filter betas based on the method.
        args:
            method: str
                Method to filter betas. Options: 'mean', 'encoding_only', 'decoding_only'.
        """
        filtered_task_betas = self.task_betas.copy()

        for task in self.task_betas.keys():
            for sess in self.task_betas[task].keys():
                for run in self.task_betas[task][sess].keys():
                    # Get conditions and betas
                    task_df = self.df_conditions[task][sess][run]
                    betas  = self.task_betas[task][sess][run]

                    # Filter conditions for delay and encoding
                    filtered_delay_df = task_df[(task_df['prev_stimulus'] == 1000) & (task_df['regressor_type'] == 'delay')]
                    filtered_encoding_df = task_df[(task_df['prev_stimulus'] == 1000) & (task_df['regressor_type'] == 'encoding')]  

                    # Divide all values of filtered_delay_df.loc[:, 0] by 2
                    filtered_delay_df.loc[:, 'Unnamed: 0'] = filtered_delay_df.loc[:, 'Unnamed: 0'] // 2
                    betas_delay = betas[:, filtered_delay_df['Unnamed: 0'].to_numpy()]

                    # Divide all values of filtered_encoding_df.loc[:, 0] by 2
                    filtered_encoding_df.loc[:, 'Unnamed: 0'] = filtered_encoding_df.loc[:, 'Unnamed: 0'] // 2
                    betas_encoding = betas[:, filtered_encoding_df['Unnamed: 0'].to_numpy()]

                    if method == 'mean':
                        # Average betas_delay and betas_encoding
                        filtered_betas = np.mean([betas_delay, betas_encoding], axis=0)
                    elif method == 'encoding_only':
                        # Only keep betas_encoding
                        filtered_betas = betas_encoding
                    elif method == 'decoding_only':
                        # Only keep betas_delay
                        filtered_betas = betas_delay

                    filtered_task_betas[task][sess][run] = filtered_betas

        self.task_betas = filtered_task_betas

    def load_and_network_map_atlas(self, network_file, network_mapping):
        """
        desc: Load and map atlas based on network regions mapping.
        args:
            network_file: str
                Path to the network txt file.
            network_mapping: dict
                Mapping of network regions to names.
        """
        # Load network region assignment
        network_region_assignment = np.loadtxt(network_file, dtype=int)

        # Initialize mapped_atlas
        for region in range(1 ,361):
            region_idxs = np.where(self.glasser_atlas == region)[0]
            network_name = network_mapping[network_region_assignment[region-1]]

            # Add network name and region indexes to mapped_atlas
            if network_name not in self.mapped_atlas.keys():
                self.mapped_atlas[network_name] = region_idxs
            else:
                self.mapped_atlas[network_name] = np.concatenate((self.mapped_atlas[network_name], region_idxs))
                
    def load_and_map_atlas(self, table_path): # NOTE: This may be wrong! I Remember XX saying not to use the Glasser excel table to map
            """
            desc: Load and map atlas based on the table path.
            args:
                table_path: str
                    Path to the atlas mapping table.
            """
            map_df_lh = pd.read_excel(table_path)
            map_df_lh = map_df_lh.iloc[:,0:2]

            # Rename columns
            map_df_lh.columns = ['region_id', 'region_name']
            map_df_rh = map_df_lh.copy()

            # Increase map_df_rh region_id by 180 and index by 180
            map_df_rh['region_id'] = map_df_rh['region_id'] + 180
            map_df_rh.index = map_df_rh.index + 180

            # Concatenate both hemispheres
            map_df = pd.concat([map_df_lh, map_df_rh], axis=0)

            for id, name in zip(map_df['region_id'], map_df['region_name']):

                # indexes of id in glasser_atlas
                if str(name) not in self.mapped_atlas.keys():
                    self.mapped_atlas[str(name)] = np.where(self.glasser_atlas == id)[0]
                else:
                    self.mapped_atlas[str(name)] = np.concatenate((self.mapped_atlas[str(name)], np.where(self.glasser_atlas == id)[0]))

            print("Mapped Atlas Size: ", len(self.mapped_atlas.keys()))

    def average_task_betas_per_sess(self):
        """
        desc: Average betas over sessions (but not runs).
        return: dict
            Dictionary with average betas for each region, task, and session.
        """

        avg_betas_dict = {}

        # Across each region
        for name, indexes in self.mapped_atlas.items():
            avg_betas_dict[name] = {session:{} for session in self.sessions}
            # Across each sess
            for sess in avg_betas_dict[name].keys():
                # Across each task
                for task in self.task_betas.keys():
                    if sess in list(self.task_betas[task].keys()):
                        run_betas = torch.tensor([])
                        # Across each run
                        for run in self.task_betas[task][sess].keys():
                            # Betas for all trials
                            betas = torch.tensor(self.task_betas[task][sess][run][indexes])

                            # Check if betas is all zeros
                            if torch.sum(betas) == 0:
                                print('Found all zeros for ' + name + ' ' + task + ' ' + sess + ' ' + run)
                                continue

                            # Mean beta across trials
                            trial_avg_betas = torch.mean(betas, axis=1).unsqueeze(1)

                            # Add to run_betas
                            run_betas = torch.cat((run_betas, trial_avg_betas), axis=1)

                        # Check if run_betas is not empty
                        if run_betas.shape[0] != 0:
                            run_betas = torch.mean(run_betas, axis=1)
                            avg_betas_dict[name][sess][task] = run_betas
                        else:
                            print('no data at: ', name, task, sess, run)
                            
        return avg_betas_dict  

    def average_task_betas_over_sess_run(self):
        """
        desc: Average betas over sessions and runs.
        return: dict
            Dictionary with average betas for each region and task.
        """

        avg_betas_dict = {}

        # Across each region
        for name, indexes in self.mapped_atlas.items():
            avg_betas_dict[name] = {}

            # Across each task
            for task in self.task_betas.keys():
                sess_betas = torch.tensor([])

                sessions = list(self.task_betas[task].keys())

                # Across each session
                for sess in sessions:
                    run_betas = torch.tensor([])

                    # Across each run
                    for run in self.task_betas[task][sess].keys():
                        # Betas for all trials
                        betas = torch.tensor(self.task_betas[task][sess][run][indexes])

                        # Check if betas is all zeros
                        if torch.sum(betas) == 0:
                            continue

                        # Mean beta across trials
                        trial_avg_betas = torch.mean(betas, axis=1).unsqueeze(1)

                        # Add to run_betas
                        run_betas = torch.cat((run_betas, trial_avg_betas), axis=1)

                    # Check if run_betas is not empty
                    if run_betas.shape[0] != 0:
                        run_betas = torch.mean(run_betas, axis=1).unsqueeze(1)
                        sess_betas = torch.cat((sess_betas, run_betas), axis=1)
                    else:
                        print('no data at: ', name, task, sess, run)
                
                # Final betas averaging accross sessions
                sess_betas = torch.mean(sess_betas, axis=1)  
                avg_betas_dict[name][task] = sess_betas
    
        return avg_betas_dict

    def average_task_betas_over_halfsess_run(self, first_half): # NOTE: Made this to analyse a weird finding where first half of sessions showed better results than second half
        """
        desc: Average betas over half of the sessions and runs.
        args:
            first_half: bool
                If True, average over the first half of sessions.
                If False, average over the second half of sessions.
        return: dict
            Dictionary with average betas for each region and task.
        """
        avg_betas_dict = {}

        # Across each region
        for name, indexes in self.mapped_atlas.items():
            avg_betas_dict[name] = {}
            # Across each task
            for task in self.task_betas.keys():
                sess_betas = torch.tensor([])

                sessions = list(self.task_betas[task].keys())

                # Get half of sessions
                if first_half:
                    sessions = sessions[:len(sessions)//2]
                else:
                    sessions = sessions[len(sessions)//2:]

                # Across each session
                for sess in sessions:
                    run_betas = torch.tensor([])

                    # Across each run
                    for run in self.task_betas[task][sess].keys():
                        # Betas for all trials
                        betas = torch.tensor(self.task_betas[task][sess][run][indexes])

                        # Check if betas is all zeros
                        if torch.sum(betas) == 0:
                            print('Found all zeros for ' + name + ' ' + task + ' ' + sess + ' ' + run)
                            continue

                        # Mean beta across trials
                        trial_avg_betas = torch.mean(betas, axis=1).unsqueeze(1)

                        # Add to run_betas
                        run_betas = torch.cat((run_betas, trial_avg_betas), axis=1)

                    # Check if run_betas is not empty
                    if run_betas.shape[0] != 0:
                        run_betas = torch.mean(run_betas, axis=1).unsqueeze(1)
                        sess_betas = torch.cat((sess_betas, run_betas), axis=1)
                    else:
                        print('no data at: ', name, task, sess, run)
                sess_betas = torch.mean(sess_betas, axis=1)  
                avg_betas_dict[name][task] = sess_betas
    
        return avg_betas_dict

    def leave_1session_out_average_task_betas(self, out_sess): 
        """
        desc: Average betas over sessions and runs, leaving one session out.
        args:
            out_sess: str
                Session to leave out.
        return: dict
            Dictionary with average betas for each region and task, leaving one session out.
        """
        avg_betas_dict = {}

        # Across each region
        for name, indexes in self.mapped_atlas.items():
            avg_betas_dict[name] = {}

            # Across each task
            for task in self.task_betas.keys():
                sess_betas = torch.tensor([])

                sessions = list(self.task_betas[task].keys())

                # Across each session
                for sess in sessions:
                    if sess == out_sess:
                        continue
                    run_betas = torch.tensor([])

                    # Across each run
                    for run in self.task_betas[task][sess].keys():
                        # Betas for all trials
                        betas = torch.tensor(self.task_betas[task][sess][run][indexes])

                        # Check if betas is all zeros
                        if torch.sum(betas) == 0:
                            print('Found all zeros for ' + name + ' ' + task + ' ' + sess + ' ' + run)
                            continue

                        # Mean beta across trials
                        trial_avg_betas = torch.mean(betas, axis=1).unsqueeze(1)

                        # Add to run_betas
                        run_betas = torch.cat((run_betas, trial_avg_betas), axis=1)

                    # Check if run_betas is not empty
                    if run_betas.shape[0] != 0:
                        run_betas = torch.mean(run_betas, axis=1).unsqueeze(1)
                        sess_betas = torch.cat((sess_betas, run_betas), axis=1)
                    else:
                        print('no data at: ', name, task, sess, run)
                sess_betas = torch.mean(sess_betas, axis=1)  
                avg_betas_dict[name][task] = sess_betas

        return avg_betas_dict
    
    def compare_region_rsm(self, betas_dict, baseline_rsm):
        """
        desc: Compare the RSM of a region with a baseline RSM.
        args:
            betas_dict: dict
                Dictionary with betas for a region.
            baseline_rsm: numpy array
                Baseline RSM to compare with.
        return: float
            Pearson correlation coefficient between the region RSM and the baseline RSM.
        """

        # Brain RSM
        betas = torch.stack(list(betas_dict.values()))
        brain_rsm = util.cos_sim(betas, betas).numpy()

        # Only need upper triangle of the RSMs
        baseline_rsm = baseline_rsm[np.triu_indices(baseline_rsm.shape[0], k=1)]
        brain_rsm = brain_rsm[np.triu_indices(brain_rsm.shape[0], k=1)]

        # Flatten the RSMs
        brain_rsm_flat = brain_rsm.flatten()
        baseline_rsm_flat = baseline_rsm.flatten()
        
        # Compute Pearson correlation
        correlation, _ = pearsonr(brain_rsm_flat, baseline_rsm_flat)
        
        return correlation

    def compare_all_regions_rsm(self, betas_dict, baseline_rsm, save_path, print_top_k=10):
        """
        desc: Compare the RSM of all regions with a baseline RSM.
        args:
            betas_dict: dict
                Dictionary with betas for all regions.
            baseline_rsm: numpy array
                Baseline RSM to compare with.
            save_path: str
                Path to save the results.
            print_top_k: int
                Number of top regions to print.
        return: None
        """
        rsm_similarities = pd.DataFrame(columns=['region', 'correlation'])

        for region in betas_dict.keys():
            corr = self.compare_region_rsm(betas_dict[region], baseline_rsm)
            rsm_similarities.loc[len(rsm_similarities)] = [region, corr]

        rsm_similarities.to_csv(save_path + 'rsm_similarities.csv')

        # Print top k regions
        print(rsm_similarities.nlargest(print_top_k, 'correlation'))
    
    def plot_region_rsm(self, region, betas_dict, save_path):
        """
        desc: Plot the RSM of a region.
        args:
            region: str
                Region name.
            betas_dict: dict
                Dictionary with betas for the region.
            save_path: str
                Path to save the plot.
        return: None
        """

        betas = torch.stack(list(betas_dict.values()))
        brain_rsm = util.cos_sim(betas, betas).numpy()

        # Plot region rsm with ticklabels as sentences
        plt.clf()
        plt.imshow(brain_rsm, cmap='hot', interpolation='nearest')
        plt.colorbar()
        _ = plt.xticks(range(len(betas_dict.keys())), betas_dict.keys(), rotation=90)
        _ = plt.yticks(range(len(betas_dict.keys())), betas_dict.keys())
        plt.title(region)
        plt.savefig(save_path + region + '_rsm.png')

    def plot_all_regions_rsm(self, betas_dict, save_path, ):
        """
        desc: Plot the RSM of all regions.
        args:
            betas_dict: dict
                Dictionary with betas for all regions.
            save_path: str
                Path to save the plot.
        return: None
        """
        # If save_path does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for region in betas_dict.keys(): 

            self.plot_region_rsm(region, betas_dict[region], save_path)

    def plot_all_sessions_rsm(self, avg_per_sess_task_betas, save_path):
        """
        desc: Plot the RSM of all sessions.
        args:
            avg_per_sess_task_betas: dict
                Dictionary with average betas for each region, task, and session.
            save_path: str
                Path to save the plot.
        return: None
        """

        # If save_path does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for region in avg_per_sess_task_betas.keys(): 
            for session in avg_per_sess_task_betas[region].keys():
                self.plot_region_rsm(region + '_' + session, avg_per_sess_task_betas[region][session], save_path)
            

