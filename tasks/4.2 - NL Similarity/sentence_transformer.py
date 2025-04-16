from sentence_transformers import SentenceTransformer, util
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import pearsonr
import numpy as np

class ST_TaskSim:
    """ 
    Class to compute the similarity between task instructions using a SentenceTransformer model.

    Parameters:
    tasks: list of strings
        List of task names to compare.
    task_features: dict
        Dictionary of task features.
    ins_mapping: dict
        Dictionary of task names and their instructions.
    figures_path: string
        Path to save the figures.
    model_name: string 
        Name of the SentenceTransformer model to use.
    """
    def __init__(self, tasks, ins_mapping, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

        self.tasks = tasks
        self.ins_mapping = ins_mapping

        self.tasks.sort(key=lambda x: list(ins_mapping.keys()).index(x))

        self.instructions = [ins for name, ins in list(ins_mapping.items()) if name in tasks]
        self.names = [name for name, ins in list(ins_mapping.items()) if name in tasks]

    def encode(self, sentences, convert_to_tensor=False):
        # Encode the instructions
        return self.model.encode(sentences, convert_to_tensor=convert_to_tensor)

    def sentence_similarity(self):
        # Compute embeddings
        embeddings = self.encode(self.instructions, convert_to_tensor=True)
        
        # Compute cosine-similarities for each sentence with each other sentence
        scores = util.cos_sim(embeddings, embeddings).cpu().numpy()

        return scores

    def plot_heatmap(self, scores, save_path):
        # Create a heatmap from the RSM with ticklabels as sentences
        plt.imshow(scores, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.xticks(range(len(self.names)), self.names, rotation=90)
        plt.yticks(range(len(self.names)), self.names, )
        plt.savefig(save_path + 'sentence_transformer_heatmap.png', bbox_inches='tight')
        return



