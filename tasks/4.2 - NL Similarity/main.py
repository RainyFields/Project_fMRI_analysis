from brain import Brain
from sentence_transformer import ST_TaskSim
import nibabel as nib 

if __name__ == '__main__':

    # Tasks
    tasks = ['1backloc', '1backctg', '1backobj', 'ctxlco' , 'ctxcol', 
        'interdmsobjABAB', 'interdmslocABBA', 'interdmslocABAB',
        'interdmsctgABAB', 'interdmsobjABBA','interdmsctgABBA'] 

    # Task instructions
    ins_mapping = {
               '1backloc': 'Match stimuli 1 and 2, 2 and 3, 3 and 4, 4 and 5, 5 and 6 based on LOCATION. Respond as fast as you can',
               '1backctg': 'Match stimuli 1 and 2, 2 and 3, 3 and 4, 4 and 5, 5 and 6 based on CATEGORY. Respond as fast as you can',
               '1backobj': 'Match stimuli 1 and 2, 2 and 3, 3 and 4, 4 and 5, 5 and 6 based on IDENTITY. Respond as fast as you can',
               'ctxlco': 'If stimuli 1 and 2 match in LOCATION, match stimuli 2 and 3 based on CATEGORY, otherwise on IDENTITY.',
               'ctxcol': 'If stimuli 1 and 2 match in CATEGORY, match stimuli 2 and 3 based on IDENTITY, otherwise on LOCATION.',
               'interdmslocABBA': 'Match stimuli 2 and 3, then stimuli 1 and 4 based on LOCATION. Respond as fast as you can',
               'interdmsctgABBA': 'Match stimuli 2 and 3, then stimuli 1 and 4 based on CATEGORY. Respond as fast as you can',
               'interdmsobjABBA': 'Match stimuli 2 and 3, then stimuli 1 and 4 based on IDENTITY. Respond as fast as you can',
               'interdmslocABAB': 'Match stimuli 1 and 3, and stimuli 2 and 4 based on LOCATION. Respond as fast as you can',
               'interdmsctgABAB': 'Match stimuli 1 and 3, and stimuli 2 and 4 based on CATEGORY. Respond as fast as you can',
               'interdmsobjABAB': 'Match stimuli 1 and 3, and stimuli 2 and 4 based on IDENTITY. Respond as fast as you can',
               }

    # Task features
    task_features= {'loc':['1backloc', 'interdmslocABAB', 'interdmslocABBA'], 
                    'ctg':['1backctg', 'interdmsctgABAB','interdmsctgABBA' ], 
                    'obj':['1backobj', 'interdmsobjABAB', 'interdmsobjABBA']}

    # Subject, runs, sessions
    subj = 'sub-03'
    runs = ['run-01', 'run-02', 'run-03', 'run-04', 'run-05']
    sessions = ['ses1', 'ses2', 'ses3', 'ses4', 'ses5', 'ses6', 'ses7', 'ses8', 'ses9', 'ses10', 'ses11', 'ses12', 'ses13', 'ses14', 'ses15', 'ses16']

    # Directory paths
    basedir = '/Users/lucasgomez/Desktop/Neuro/Bashivan/Hackthon_WM_fMRI/'
    datadir = basedir + 'data/'
    betasdir = datadir + 'data/glm_betas_encoding_delay_full_TR_betas/' + subj + '/'
    network_file =  datadir + 'ColeAnticevicNetPartition/cortex_parcel_network_assignments.txt'
    glasser_atlas_str= datadir + 'Glasser_LR_Dense64k.dlabel.nii'
    table_path = datadir + 'Glasser_2016_Table.xlsx'
    figures_path = basedir + 'tasks/4.2 - NL Similarity/figures/'

    network_mapping = {
                        1: "primary visual",
                        2: "secondary visual",
                        3: "somatomotor",
                        4: "cingulo-opercular",
                        5: "dorsal attention",
                        6: "language",
                        7: "frontparietal",
                        8: "auditory",
                        9: "default mode",
                        10: "posterior multimodal",
                        11: "ventral multimodal",
                        12: "orbito-affective",
                        }

    # Atlas 
    glasser_atlas = nib.load(glasser_atlas_str).get_fdata()[0].astype(int)
    print('Number of regions:', glasser_atlas.shape)

    # Plot sentencetransformer embedding rsm
    st_tasksim = ST_TaskSim(tasks=tasks, ins_mapping=ins_mapping, model_name='all-MiniLM-L6-v2')
    sim_scores = st_tasksim.sentence_similarity()
    st_tasksim.plot_heatmap(sim_scores, save_path=figures_path)

    # Load brain data
    brain = Brain(basedir=basedir, datadir=datadir, betasdir=betasdir, subj=subj, runs=runs, sessions=sessions, tasks=tasks, glasser_atlas=glasser_atlas)
    brain.load_betas(normalized=False, correct=False)

    # Filter brain data
    brain.filter_betas()

    # Load and map network
    brain.load_and_network_map_atlas(network_file=network_file, network_mapping=network_mapping) # UNCOMMENT IF DOING NETWORKS

    # Load and map atlas
    # brain.load_and_map_atlas(table_path=table_path)  # UNCOMMENT IF DOING ALL REGIONS

    # Get betas per task and region; averaged over all trials for sessions and runs
    avg_task_betas = brain.average_task_betas_over_sess_run()
    
    # Plot rsm of avg_task_betas
    brain.plot_all_regions_rsm(avg_task_betas, save_path=figures_path + 'avg_betas_all_sess/')

    # Compare rsms of avg_task_betas and st_tasksim
    brain.compare_all_regions_rsm(avg_task_betas, sim_scores, save_path=figures_path + 'avg_betas_all_sess/', print_top_k=10)

    # Get betas per task and region; averaged over all trails & runs for half of sessions
    avg_task_betas_first_half = brain.average_task_betas_over_halfsess_run(True)
    avg_task_betas_second_half = brain.average_task_betas_over_halfsess_run(False)

    # Plot rsm of avg_task_betas_first_half
    brain.plot_all_regions_rsm(avg_task_betas_first_half, save_path=figures_path + 'avg_betas_half_sess/first_half/')

    # Plot rsm of avg_task_betas_second_half
    brain.plot_all_regions_rsm(avg_task_betas_second_half, save_path=figures_path + 'avg_betas_half_sess/second_half/')

    # Get betas per task, region, and session; averaged over all trials for runs
    avg_per_sess_task_betas = brain.average_task_betas_per_sess()

    # Plot rsm of avg_per_sess_task_betas
    brain.plot_all_sessions_rsm(avg_per_sess_task_betas, save_path=figures_path + 'avg_betas_per_sess/')

    # Get betas per task and region; averaged over all trials for sessions and runs, with one session left out
    for out_sess in sessions:
        avg_1o_sess_task_betas = brain.leave_1session_out_average_task_betas(out_sess=out_sess)
        # Plot rsm of avg_1o_sess_task_betas
        brain.plot_all_regions_rsm(avg_1o_sess_task_betas, save_path=figures_path + 'avg_betas_per_sess_leave_out/' + out_sess + '/')