import os
import numpy as np
import copy
import matplotlib.ticker as ticker
import surfplot
import nibabel as nib
import pandas as pd
from neuromaps.datasets import fetch_fslr

from functools import lru_cache
from pathlib import Path

# Get the current working directory
current_directory = Path.cwd()

# Get the parent directory (one level up)
parent_dir = str(current_directory.parent)
glasser_atlas_str= parent_dir + '/data/Glasser_LR_Dense64k.dlabel.nii'
glasser_atlas = nib.load(glasser_atlas_str).get_fdata()[0].astype(int)
num_regions = 360

def make_brain_surface_plot(data, fig_title, cmap="bwr", center_value=0):
    """
    Plots brain surface data using the fsLR veryinflated surface.

    Parameters:
    - data (numpy.ndarray): The statistical map data to be plotted.
    - fig_title (str): Title of the plot.
    - cmap (str): Colormap to use (default: "bwr" for blue-white-red).
    - center_value (float): Center value for color scale adjustment (default: 0).

    Returns:
    - fig (matplotlib.figure.Figure): The generated figure object.
    """
    # Load fsLR veryinflated surfaces (left and right hemisphere)
    surfaces = fetch_fslr()
    lh, rh = surfaces['veryinflated']

    # Initialize surface data array
    surface_dat = np.zeros(glasser_atlas.shape)

    # Map input data to brain regions
    for roi in range(1, num_regions + 1):
        roi_ind = np.where(glasser_atlas == roi)[0]
        surface_dat[roi_ind] = data[roi_ind]

    # Determine color range based on colormap choice
    if cmap == "bwr":
        max_abs_dev = np.max(np.abs(surface_dat - center_value))
        vmin, vmax = center_value - max_abs_dev, center_value + max_abs_dev
    else:
        vmin, vmax = np.min(surface_dat), np.max(surface_dat)

    # Create the surface plot
    p = surfplot.Plot(lh, rh, size=(500, 350), zoom=1.8)
    p.add_layer(surface_dat.T, cmap=cmap, color_range=[vmin, vmax])

    # Build the figure with colorbar
    fig = p.build(figsize=(4, 4), colorbar=True, cbar_kws={'fontsize': 8})

    # Format colorbar tick labels
    if fig.axes:
        cbar = fig.axes[-1]  # Get the colorbar axis
        cbar.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # Add plot title
    fig.suptitle(fig_title, y=0.98, fontsize=14)

    # Adjust layout for better appearance
    fig.tight_layout()

    return fig




def df_concatenation(df, excluded_ses=None):
    """
    Concatenates dataframes across all available tasks, sessions, and runs.

    Parameters:
    - df (dict): Nested dictionary containing task/session/run dataframes.
    - excluded_ses (list, optional): List of session names to exclude from concatenation.

    Returns:
    - pd.DataFrame: Concatenated dataframe across all tasks and runs.
    """
    if excluded_ses is None:
        excluded_ses = []

    all_df = []

    try:
        for taskname, sessions in df.items():
            for ses, runs in sessions.items():
                if ses not in excluded_ses:
                    for run, run_df in runs.items():
                        all_df.append(run_df)
    except AttributeError as e:
        print(f"Warning: Issue processing df. Error: {e}")
        for ses, runs in df.items():
            if ses not in excluded_ses:
                for run, run_df in runs.items():
                    all_df.append(run_df)

    # Ensure there's data to concatenate
    if not all_df:
        raise ValueError("No valid dataframe data found. Check the df structure and excluded sessions.")

    concatenated_df = pd.concat(all_df, axis=0, ignore_index=True)

    return concatenated_df






def preprocess_contrast_map(contrast_map, excluded_ses=None):
    """
    Preprocesses the contrast_map to flatten it into a list of beta arrays,
    excluding the specified sessions.

    Parameters:
    - contrast_map (dict): Nested dictionary containing task/session/run beta arrays.
    - excluded_ses (list, optional): List of sessions to exclude from concatenation.

    Returns:
    - list: List of beta arrays.
    """
    if excluded_ses is None:
        excluded_ses = []

    all_contrast_map_betas = []

    try:
        for taskname, sessions in contrast_map.items():
            for ses, runs in sessions.items():
                if ses not in excluded_ses:
                    for run, betas in runs.items():
                        all_contrast_map_betas.append(betas)
    except AttributeError as e:
        print(f"Warning: Issue processing contrast_map. Error: {e}")
        for ses, runs in contrast_map.items():
            if ses not in excluded_ses:
                for run, betas in runs.items():
                    all_contrast_map_betas.append(betas)

    return all_contrast_map_betas

# @lru_cache(maxsize=None)
def beta_concatenation(contrast_map, excluded_ses=None):
    """
    Concatenates beta values across all available tasks, sessions, and runs.

    Parameters:
    - contrast_map (dict): Nested dictionary containing task/session/run beta arrays.
    - excluded_ses (list, optional): List of sessions to exclude from concatenation.

    Returns:
    - np.ndarray: Concatenated beta values across all dimensions (transposed).
    """
    
    all_contrast_map_betas = preprocess_contrast_map(contrast_map, excluded_ses)
 
    # Ensure that there is data to concatenate
    if not all_contrast_map_betas:
        raise ValueError("No valid beta data found. Check contrast_map structure and excluded sessions.")

    # Preallocate array if possible (requires knowing the total size)
    total_size = sum(betas.shape[1] for betas in all_contrast_map_betas)
    if total_size > 0:
        concatenated_betas = np.empty((all_contrast_map_betas[0].shape[0], total_size))
        start_idx = 0
        for betas in all_contrast_map_betas:
            end_idx = start_idx + betas.shape[1]
            concatenated_betas[:, start_idx:end_idx] = betas
            start_idx = end_idx
    else:
        concatenated_betas = np.concatenate(all_contrast_map_betas, axis=1)

    return concatenated_betas.T

def filter_task_betas(task_betas, df_conditions, phase="delay", first_delay_only=True, second_delay_only=False, third_delay_only = False):
    """
    Filters the task betas based on specified conditions.

    Parameters:
    - task_betas (dict): Dictionary containing beta values for each task/session/run.
    - df_conditions (dict): Dictionary containing conditions for each task/session/run.
    - phase (str): The phase to filter on (default is "delay").
    - first_delay_only (bool): If True, filters only trials where 'prev_stimulus' is 1000.

    Returns:
    - filtered_task_betas (dict): Filtered beta values.
    - filtered_task_df (dict): Filtered conditions dataframe.
    """

    # Deep copy to avoid modifying original data
    filtered_task_betas = copy.deepcopy(task_betas)
    filtered_task_df = copy.deepcopy(df_conditions)

    for task, sessions in task_betas.items():
        for sess, runs in sessions.items():
            for run, betas in runs.items():
                task_df = df_conditions[task][sess].get(run)

                # Safety check to ensure data exists
                if task_df is None or betas is None:
                    print(f"Warning: Missing data for task={task}, session={sess}, run={run}")
                    continue

                # Apply filtering conditions
                if first_delay_only:
                    condition_mask = (task_df['prev_stimulus'] == 1000) & (task_df["regressor_type"] == phase)
                elif second_delay_only:
                    condition_mask = (task_df['prev_stimulus'] != 1000) & (task_df['prev_prev_stimulus'] == 1000) & (task_df["regressor_type"] == phase)
                elif third_delay_only:
                    condition_mask = (task_df['prev_stimulus'] != 1000) & (task_df['prev_prev_stimulus'] != 1000) & (task_df["regressor_type"] == phase)
                else:
                    condition_mask = task_df["regressor_type"] == phase

                # Ensure indexing consistency and type safety
                filtered_df = task_df[condition_mask].reset_index(drop=True)
                betas_phase = betas[:, condition_mask.to_numpy()]

                # Store filtered data
                filtered_task_betas[task][sess][run] = betas_phase
                filtered_task_df[task][sess][run] = filtered_df

    return filtered_task_betas, filtered_task_df


