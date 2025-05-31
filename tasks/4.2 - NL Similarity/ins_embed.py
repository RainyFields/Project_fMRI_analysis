from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class InsEmbedder(object):
    """
    desc: Instruction Encoder class
    args:
        - model: instruction encoder model
        - tokenizer: instruction tokenizer
        - tasks: list of task names to compare.
        - ins_mapping: Dictionary of task names and their instructions.
    """
    def __init__(self, model, tokenizer, tasks, ins_mapping):
        self.model = model
        self.tokenizer = tokenizer

        self.tasks = tasks
        self.ins_mapping = ins_mapping

        self.tasks.sort(key=lambda x: list(ins_mapping.keys()).index(x))

        self.instructions = [ins for name, ins in list(ins_mapping.items()) if name in tasks]
        self.names = [name for name, ins in list(ins_mapping.items()) if name in tasks]

    def update_tasks(self, tasks):
        """
        desc: Update the tasks to compare.
        args:
            - tasks: list of task names to compare.
        """
        self.tasks = tasks
        self.instructions = [ins for name, ins in list(self.ins_mapping.items()) if name in tasks]
        self.names = [name for name, ins in list(self.ins_mapping.items()) if name in tasks]

    def __call__(self, instruction, normalize: bool, device='cpu'):
        """
        args:
            - instruction: str, instruction to encode
            - pool: bool, whether to use mean pooling
            - device: str, device to use
        """

        #Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Compute token embeddings
        # task = 'text-matching'
        # task_id = self.model._adaptation_map[task]
        # adapter_mask = torch.full((len(instruction),), task_id, dtype=torch.int32)
        with torch.no_grad():
            ins_tensor  = self.tokenizer(instruction, padding=True, truncation=False, return_tensors='pt').to(device)
            # lm_output = self.model(**ins_tensor, adapter_mask=adapter_mask)
            lm_output = self.model(**ins_tensor)

        # Perform pooling
        sentence_embeddings = mean_pooling(lm_output, ins_tensor['attention_mask'])
        
        # Normalize embeddings
        if normalize:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            
        return sentence_embeddings

    def sentence_similarity(self, normalize=False):
        # Compute embeddings
        embeddings = self(self.instructions, normalize=normalize)
        
        # Compute cosine-similarities for each sentence with each other sentence
        scores = cosine_similarity(embeddings.cpu().numpy(), embeddings.cpu().numpy())

        return scores
    
    def plot_heatmap(self, scores, save_path):
        # Create a heatmap from the RSM with ticklabels as sentences
        plt.clf()
        plt.imshow(scores, cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.xticks(range(len(self.names)), self.names, rotation=90)
        plt.yticks(range(len(self.names)), self.names, )
        plt.savefig(save_path + 'sentence_transformer_heatmap.png', bbox_inches='tight')
        return