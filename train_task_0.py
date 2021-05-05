import torch.utils.data as data
import sys
sys.path.append('./data')
from dataset_task_0 import Dataset_copy
from model.Deepsets import Deepset
import torch.nn as nn
import torch.optim as optim

EPOCHs = 10

dataset = Dataset_copy('./data/test.npy')
train_loader = data.DataLoader(dataset)

loss_fun = nn.L1Loss()
model = Deepset().cuda().train()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHs):
    for (x,y) in train_loader:
        x = x.cuda()
        y = y.cuda()
        print(x)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fun(output,y)
        print(loss)
        loss.backward()
        optimizer.step()
