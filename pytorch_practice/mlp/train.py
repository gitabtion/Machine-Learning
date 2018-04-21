import torch
from torch.autograd import Variable
import torch.nn as nn
from mlp import model

x0 = torch.rand(10000)
y0 = torch.zeros(10000)

x1 = torch.rand(10000)
y1 = torch.ones(10000)
x1 = x1 - y1
x = torch.cat((x0, x1), ).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

x = torch.unsqueeze(x, 1)

x = Variable(x)
y = Variable(y)

model = model.mlp(2)
if torch.cuda.is_available():
    model = model.cuda()
    x = x.cuda()
    y = y.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
loss_func = nn.CrossEntropyLoss()
print(model)
print(x.shape)
for t in range(10000):
    out = model(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if torch.cuda.is_available():
    model = model.cpu()

torch.save(model, './module.pkl')

print('done!')
