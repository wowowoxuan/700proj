import torch.utils.data as data
import sys
sys.path.append('./data')
from dataset_task_0 import Dataset_copy
from model.Deepsets import Deepset
import torch.nn as nn
import torch.optim as optim
import torch

EPOCHs = 50
# def my_collate(batch):
#     data = torch.stack([item[0] for item in batch], 0)
#     data = data.float()
#     target = torch.FloatTensor([item[1] for item in batch])
#     return [data, target]

# def collate_fn(batch):
#     batch.sort(key=lambda x: len(x[1]), reverse=True)
#     img, label = zip(*batch)
#     pad_label = []
#     lens = []
#     max_len = len(label[0])
#     for i in range(len(label)):
#         temp_label = [0] * max_len
#         temp_label[:len(label[i])] = label[i]
#         pad_label.append(temp_label)
#         lens.append(len(label[i]))
#     return img, pad_label, lens
dataset = Dataset_copy('./data/train.npy')
val_set = Dataset_copy('./data/val.npy')
print(len(val_set))
train_loader = data.DataLoader(dataset)
val_loader = data.DataLoader(val_set)

loss_fun = nn.L1Loss()
model = Deepset().cuda().train()
optimizer = optim.Adam(model.parameters(), lr=0.001)
flag = 0
best_acc = 0
early_stop = 5
for epoch in range(EPOCHs):
    model.train()
    for (x,y) in train_loader:
        x = x.cuda()
        y = y.cuda()
        # print(x)
        optimizer.zero_grad()
        output = model(x)

        # print(output.shape)
        loss = loss_fun(output,y)
        # print(loss)
        loss.backward()
        optimizer.step()
    total = 0
    correct_total = 0
    model.eval()
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
    if correct_total/total > best_acc:
        best_acc = correct_total/total
        flag = 0
        print('best model saved')
        torch.save(model.state_dict(), './train_task_0_statedict/best.pth')
    else:
        flag += 1
        if flag == early_stop:
            break
            
