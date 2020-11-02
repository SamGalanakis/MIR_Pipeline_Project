from evaluate_dataset import Evaluator
from process_data_for_knn import process_dataset_for_knn,sample_normalizer
from pathlib import Path
import numpy as np
import torch
from torch import  nn, optim
from online_triplet_loss.losses import *
from tqdm import tqdm 
import matplotlib.pyplot as plt
import torch.nn.functional as F
import wandb
wandb.init(project="metriclearner")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = "processed_data/data_coarse1_processed_10000_10000000.0.csv"




data_path = Path("processed_data/data_coarse1_processed_10000_10000000.0.csv")
df, *_ = process_dataset_for_knn(data_path,divide_distributions=True)
classifications = df['classification'].to_list()
classifications_unique = sorted(list(set(classifications)))
classifications_indexer = {x:classifications_unique.index(x) for x in classifications_unique}
classifications_numeric = [classifications_indexer[x] for x in classifications]
df_numeric = df.select_dtypes(include=np.number)



labels = torch.tensor(classifications_numeric).to(device)



class Metric(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(df_numeric.shape[1],requires_grad=True))
        # self.fc1 = nn.Linear(df_numeric.shape[1],df_numeric.shape[1])
        
        # self.fc2 = nn.Linear(df_numeric.shape[1],df_numeric.shape[1])
        

    def forward(self,x):
        # x = self.fc1(x)

        # x=self.fc2(F.relu(x))
        # x = F.relu(x)
        normed_weigths = self.weights/torch.norm(self.weights)
        x = torch.mul(normed_weigths,x)
        x=x/torch.norm(x)
        
        return x


model = Metric().to(device)
model = model.float()
wandb.watch(model)
print(list(model.parameters()))
optimizer = optim.Adam(model.parameters(),lr=0.0001)
dataset_torch = torch.tensor(df_numeric.values).to(device).float()

loss_overall=0
epoch_max = int(1e+6)
losses = []


for epoch in range(epoch_max):
    optimizer.zero_grad()
    
    
    dotted_features = model(dataset_torch)
    loss=   batch_hard_triplet_loss(labels,dotted_features,margin =  0.1,device='cuda')
    loss.backward()
    

    
    losses.append(loss.item())
    optimizer.step()
    wandb.log({'Loss':loss.item()})
    
    if epoch % 2000 == 0 :
        
    
        df_input = df.copy()
        df_numeric_input = df_input.select_dtypes(np.number) 
        df_input[df_numeric_input.columns] = dotted_features.cpu().detach().numpy()
        evaluator = Evaluator(df_input)
        evaluator.evaluate()
        metrics_dict = evaluator.analysis()
        precision = metrics_dict['precision']
        f1 = metrics_dict['precision']
        recall = metrics_dict['f1']
        
        print(f"Loss: {loss.item()}, Precision: {precision}, Recall: {recall}")

    
    

        




