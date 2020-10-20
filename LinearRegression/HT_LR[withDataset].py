#Load pytorch library
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
"""
dummy dataset with normal distribution

"""
X = torch.randn(100,1)*10
#uncomment if it is exact linear
#Y = X 
#add noise
Y = X+ 5*torch.randn(100,1)
plt.plot(X.numpy(),Y.numpy(),'p')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

class HT_LR(nn.Module):
    #initial input and output as intances
    def __init__(self, input_size, output_size):
        super().__init__()
        self.HT_linear = nn.Linear(input_size,output_size) 
    def forward(self, x):
        y = self.HT_linear(x)
        return y
    
#seed
torch.manual_seed(123)
model = HT_LR(1,1)
print(model)

#get the w and b parameter
[w,b] = model.parameters()
w1 = w[0][0]
b1 = b[0]
print(w1,b1)

# function to get parameters w and b
def get_params():
    return(w[0][0].item(), b[0].item())

def plot_fit(title):
    plt.title = title
    w1, b1 = get_params()
    x1 =  np.array([-30,30])
    y1 = w1*x1 + b1
    plt.plot(x1,y1,'red')
    plt.scatter(X,Y)
    plt.show()

plot_fit('Initial model')