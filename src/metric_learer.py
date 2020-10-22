#from faiss_knn import  FaissKNeighbors
from process_data_for_knn import process_dataset_for_knn,sample_normalizer
from pathlib import Path
import numpy as np
import torch
from torch import  nn, optim
from online_triplet_loss.losses import *
from tqdm import tqdm 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def query_loss(input_classification,classifications,indices,distances):
    return_classifications = [classifications[x] for x in indices]

    dist_matched=0
    dist_missed = 0
    for return_class , distance in zip(return_classifications,distances):
        
        if return_class == input_classification:
            dist_matched+=distance
        else:
            dist_missed+=distance
    
    return (dist_matched/dist_missed)/len(input_classification)



data_path = Path("processed_data/data_processed_10000_1000000.0.csv")
df, *_ = process_dataset_for_knn(data_path,divide_distributions=False)
classifications = df['classification'].to_list()
classifications_unique = sorted(list(set(classifications)))
classifications_indexer = {x:classifications_unique.index(x) for x in classifications_unique}
classifciations_numeric = [classifications_indexer[x] for x in classifications]
df_numeric = df.select_dtypes(include=np.number)

# faiss_knn = FaissKNeighbors(df_numeric,metric='L2')
# faiss_knn.train()

labels = torch.tensor(classifciations_numeric).to(device)
print('done')
class Metric(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Parameter(torch.ones(df_numeric.shape[1],requires_grad=True))
        

        
    def forward(self,x):
        return torch.mul(self.fc1,x)

model = Metric().to(device)
print(list(model.parameters()))
optimizer = optim.Adam(model.parameters(),lr=0.1)
dataset_torch = torch.tensor(df_numeric.values).to(device)

loss_overall=0
epoch_max = int(1e+6)
losses = []
for epoch in tqdm(range(epoch_max)):
    optimizer.zero_grad()
    
    
    embeddings = model(dataset_torch)
    loss= batch_hard_triplet_loss(labels,embeddings,margin=0.1,device='cuda')
    loss.backward()
    print(loss)

    
    losses.append(loss.item())
    optimizer.step()

    
    

        




