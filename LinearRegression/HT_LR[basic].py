#Load pytorch library
import torch
import torch.nn as nn

class HT_LR(nn.Module):
    #initial input and output as intances
    def __init__(self, input_size, output_size):
        super().__init__()
        self.HT_linear = nn.Linear(input_size,output_size) 
    def forward(self, x):
        y = self.HT_linear(x)
        return y
    
"""
let's see better what the model do

"""
#seed
torch.manual_seed(123)
model =  HT_LR(1,1)
#print the weight and bias
print(list(model.parameters()))

#let's make input more general
x_in = torch.tensor([[1.0],[5.0],[2.0]])
y_hat = model.forward(x_in)
print("Input (x_in): ", x_in)
print("output (y_hat): ", y_hat) 
