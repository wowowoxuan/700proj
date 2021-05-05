import torch
import torch.nn as nn
m = nn.MaxPool1d(2,stride = 2)
input = torch.randn(1,10,20)
input = input.transpose(1,2)
print(input.shape)
output = m(input)
print(output.shape)