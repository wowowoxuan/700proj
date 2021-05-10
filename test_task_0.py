import torch.utils.data as data
import sys
sys.path.append('./data')
from dataset_task_0 import Dataset_copy
from model.Deepsets import Deepset
import torch.nn as nn
import torch.optim as optim
import torch

EPOCHs = 50

val_set = Dataset_copy('./data/val.npy')
val_loader = data.DataLoader(val_set)

loss_fun = nn.L1Loss()
model = Deepset().cuda()
model.load_state_dict(torch.load('./train_task_0_statedict/best.pth'))
optimizer = optim.Adam(model.parameters(), lr=0.001)
correct_total = 0
total = 0
with torch.no_grad():
    for (x,y) in val_loader:
        x = x.cuda()
        y = y.cuda()
        output = model(x)
        pred = output.argmax(2)
        label = y.argmax(2)
        # print(label.shape)
        # print(pred==label) 
        temp1 = (pred!=label).sum(1)
        temp2 = (label!=label).sum(1)

        correct = torch.sum(temp1==temp2)
        correct_total += correct.item()
        total += x.shape[0]
print(correct_total/total)

            
