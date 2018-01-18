import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1024)

fig = plt.figure()
ax = fig.add_subplot(111)

x0 = torch.rand(500)
y0 = torch.zeros(500)

x1 = torch.rand(500)
y1 = torch.ones(500)
x1 = x1 - y1
x = torch.cat((x0, x1), ).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

x = torch.unsqueeze(x, 1)

x = Variable(x)
y = Variable(y)

model = torch.load('./module.pkl')
if torch.cuda.is_available():
    model = model.cuda()
    x = x.cuda()
    y = y.cuda()
else:
    model = model.cpu()
loss_func = nn.CrossEntropyLoss()
print(model)

out = model(x)
loss = loss_func(out, y)
print(loss)
print(out)

ax.plot(out.data.numpy())

for tick in ax.yaxis.get_major_ticks():
    tick.label1On = False
    tick.label2On = True
    tick.label2.set_color('green')

plt.show()

outs = out.data.numpy()
y_data = y.data.numpy()
num_acc = 0
for i in range(1000):
    if outs[i][0] > outs[i][1] and y_data[i] == 0:
        num_acc += 1
    elif outs[i][0] < outs[i][1] and y_data[i] == 1:
        num_acc += 1
print(num_acc / 1000)
