import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch, posixpath
from pathlib import Path
from hw2vec.config import *
from hw2vec.graph2vec.models import *  
from hw2vec.hw2graph import *

cfg = Config(sys.argv[1:])

nx_graphs = []
hw2graph = HW2GRAPH(cfg)
hw_project_path = Path(posixpath.normpath(sys.path[0]))
hw_graph = hw2graph.code2graph(hw_project_path)
nx_graphs.append(hw_graph)

data_proc = DataProcessor(cfg)
for hw_graph in nx_graphs:
    data_proc.process(hw_graph)

print("\nInput: ")
input_data = data_proc.get_graphs()[0]
print(input_data)

print("\nPickled Objects: ")
with open('dfg_tj_rtl.pkl', 'rb') as file:
    dataset = pickle.load(file)
for data in dataset:
    print(data)

print("\nModel: ")
model = GRAPH2VEC(cfg)
model_path = Path(cfg.model_path)
model.load_model(str(model_path/"model.cfg"), str(model_path/"model.pth"))
model.eval()
print(model)

x = input_data['x']
edge_index = input_data['edge_index']

output = model(x, edge_index, batch=input_data.batch)
predictions = torch.sigmoid(output[0]) 

threshold = 0.5
res = (predictions > threshold).float().mean().item()

if(res == float(1)):
    print("\nTrojan Detected.")
else: 
    print("\nTrojan Not Detected.")
