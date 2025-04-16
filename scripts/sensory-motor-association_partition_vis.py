###### run this locally!!!!!!!!

import os
import h5py
import json
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

from scipy import stats
from sklearn import svm
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_1samp
from sklearn.model_selection import cross_val_score, StratifiedKFold

from helper import decoding
from utils import filter_task_betas, beta_concatenation, make_brain_surface_plot, df_concatenation

# load glasser atlas
parent_dir = Path.cwd().parent
glasser_atlas_str= os.path.join(parent_dir,'data', 'Glasser_LR_Dense64k.dlabel.nii')
glasser_atlas = nib.load(glasser_atlas_str).get_fdata()[0].astype(int)
print("Glassier Atlas shape:", glasser_atlas.shape)
num_regions = 360

local_filename = "HCP-MMP1_UniqueRegionList.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(local_filename)

cortex_classification = {
    "Sensory": [
        "Primary_Visual",
        "MT+_Complex_and_Neighboring_Visual_Areas",
        "Dorsal_Stream_Visual",
        "Early_Visual",
        "Ventral_Stream_Visual",
        "Early_Auditory",
        "Auditory_Association",
        "Somatosensory_and_Motor"
    ],
    "Motor": [
        "Premotor",
        "Paracentral_Lobular_and_Mid_Cingulate"
    ],
    "Association": [
        "Posterior_Cingulate",
        "Temporo-Parieto_Occipital_Junction",
        "Temporo-Parieto-Occipital_Junction",
        "Superior_Parietal",
        "Dorsolateral_Prefrontal",
        "Anterior_Cingulate_and_Medial_Prefrontal",
        "Orbital_and_Polar_Frontal",
        "Inferior_Frontal",
        "Posterior_Opercular",
        "Insular_and_Frontal_Opercular",
        "Inferior_Parietal",
        "Medial_Temporal",
        "Lateral_Temporal"
    ]
}

# visualization of sensory-motor-association partition
from nilearn import datasets, plotting
import matplotlib.pyplot as plt
fsaverage = datasets.fetch_surf_fsaverage()

# Initialize a mapping array with 0 (background)
region_map = np.zeros_like(glasser_atlas)

# Assign Sensory-Motor regions as 1, Association regions as 2
for index, row in df.iterrows():
    region_name = row["cortex"]  # Assuming column contains region names
    region_index = row["regionID"]  # Assuming column contains index mapping to atlas

    if region_name in cortex_classification["Sensory"]:
        region_map[glasser_atlas == region_index] = 1
    elif region_name in cortex_classification["Motor"]:
        region_map[glasser_atlas == region_index] = 2
    elif region_name in cortex_classification["Association"]:
        region_map[glasser_atlas == region_index] = 3



import surfplot
import matplotlib.ticker as ticker

# Determine color range based on colormap choice
cmap = "coolwarm"  # Change if needed
center_value = 0  # Default center value for normalization

# Flatten the region map for surfplot
surface_dat = region_map.flatten()

if cmap == "bwr":
    max_abs_dev = np.max(np.abs(surface_dat - center_value))
    vmin, vmax = center_value - max_abs_dev, center_value + max_abs_dev
else:
    vmin, vmax = np.min(surface_dat), np.max(surface_dat)

# Create the surface plot
p = surfplot.Plot(fsaverage.infl_left, fsaverage.infl_right, size=(500, 350), zoom=1.8)
p.add_layer(surface_dat.T, cmap=cmap, color_range=[vmin, vmax])

# Build the figure with colorbar
fig = p.build(figsize=(4, 4), colorbar=True, cbar_kws={'fontsize': 8})

# Format colorbar tick labels
if fig.axes:
    cbar = fig.axes[-1]  # Get the colorbar axis
    cbar.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

# Add plot title
fig.suptitle("Sensory-Motor-Association Partition", y=0.98, fontsize=14)

# Adjust layout for better appearance
fig.tight_layout()
