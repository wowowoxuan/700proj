import torch.utils.data as data
from dataset_task_0 import Dataset_copy

dataset = Dataset_copy('./test.npy')
test_loader = data.DataLoader(dataset)
for (x,y) in test_loader:
    print("=================================分割线=================================================")
    print(x)
    print(y)