import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from pydantic import BaseModel,StrictFloat
from typing import Optional
from ignite.contrib.metrics.regression import MeanAbsoluteRelativeError
import math

#https://discuss.pytorch.org/t/why-is-the-hidden-state-initialized-to-zero-for-every-batch-when-doing-forwad-pass/101263/2
#https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
#https://discuss.pytorch.org/t/lstm-hidden-state-logic/48101
#

#A custom dataset class must implement three functions: __init__,__len__,__getitem__
#__init__ function is run once when instantiating the Dataset object We initialize the directory containing the images,annotations file and both transforms
#__len__function returns the number of samples in our dataset
#__getitem__ function loads ad returns a sample from the dataset at the given index idx.
class TypeChecks(BaseModel):
    x : np.ndarray
    y : np.ndarray
    class Config:
        arbitrary_types_allowed = True

#Custom dataset class allows you to carry out transformations on your data and then create the data set
#see example here 
#https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md
class prepare_dataset(Dataset):
    def __init__(self,x,y):
        TypeChecks(x = x,y = y)
        self.x = torch.tensor(x,dtype = torch.float32)
        self.y = torch.tensor(y,dtype = torch.float32)
        self.len = x.shape[0]

    def __getitem__(self, index):
        return self.x[index,:,:],self.y[index]
    def __len__(self):
        return self.len
    
#Input Dimension - represent the size of the input at each time step (how many features per time step)
#Hidden Dimension - reprsents the size of the hidden state and cell state at each time step
#Number of layers - the number of LSTM layers stacked on top of each other  

#Recurrent modules from torch.nn will get an input sequence and output a sequence of the same length. Just take the last element from that output sequence

#LSTM output of lstm_layer(input) will contain 2 outputs. The first output is the tensor output with a shape (Time steps, batch size, hidden neuron number(hidden size))
#Second output is the final hidden/ final cell state. The size of each cell state will be (num layers, num time sequences, num neurons)
#(This is the final cell state after back prop is done) for that element

#contiguous (https://stackoverflow.com/questions/48915810/pytorch-what-does-contiguous-do)
#A one dimensional array[0,1,2,3,4] is contiguous if its items are laid out in memory next to each other. 
#.contiguous rearranges the memory such that the tensor is C contiguous again

#hidden state in LSTMS serve as memory. We arent supposed to learn the hidden state or learn from it,
#so we can detach the values from the model, to use them in prediction but not use them in back prop
class LSTMnetwork(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(LSTMnetwork,self).__init__()
        self.hidden_size  = hidden_dim
        self.input_size = input_dim
        self.lstm_layer = nn.LSTM(self.input_size,self.hidden_size,num_layers = 1,dropout = 0.75,batch_first = True)
        self.dense_layer1 = nn.Linear(in_features=hidden_dim,out_features=256,bias = True)
        self.dense_layer2 = nn.Linear(in_features=256,out_features=1,bias=True)
    def forward(self,x,h):
        #https://discuss.pytorch.org/t/initialization-of-first-hidden-state-in-lstm-and-truncated-bptt/58384
        #Try with zero initialization too
        #Initial hidden state (short term memory)
        #(num_layers,num_batch,num_units(hiddensize))
        #h_0 = torch.empty(size = (1,1,100))
        #torch.nn.init.xavier_normal_(h_0)
        #Initial cell state (long term memory)
        #c_0 = torch.empty(size = (1,1,100))
        #we initialize the hidden state and cell state at the start of each forward iteration for each sample. 
        #If we dont do tha the back propagation runs through old batches
        #try with detaching the hidden state and cell state
        #while in theory it is true that the back prop through time would result in running through old batches, in practice
        #it would run out of memory
        #torch.nn.init.xavier_normal_(c_0)
        lstm_output,hidden_cell_state = self.lstm_layer(x,h)
        # print(lstm_output)
        #obtaining the last output for the time sequence
        #Reshape from (num sequences, batch size, neuron_num) to (num sequences * batch size, neuron_num)
        last_output = lstm_output.contiguous().view(-1,self.hidden_size)[-1]
        # print(last_output)
        output1 = self.dense_layer1(last_output)
        # print(output1)
        yield_value = self.dense_layer2(output1)
        # print(yield_value)
        return yield_value,hidden_cell_state
    def init_hidden(self,batch):
        weights = next(self.parameters()).data
        #initializing new hidden state and cell state to be used at the start of every epoch
        hidden = (torch.nn.init.xavier_normal_(weights.new(1,1,100)),torch.nn.init.xavier_normal_(weights.new(1,1,100)))
        return hidden
        
#why we detach the hidden state from the parameters?
#https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426/12
class TrainingValidatingLSTM:
    def __init__(self):
        self.error = 0
        self.validate_counts = 0
    def train(self,dataloader,model,loss_fn,optimizer,batch_size):
        model.train()
        for batch,(X,y) in enumerate(dataloader):
            hidden_state = model.init_hidden(batch_size)
            yield_pred,hidden_state = model(X,hidden_state)
            loss = loss_fn(yield_pred,y)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            # hidden_state = self.repackage_hidden(hidden_state)
    def _RMSE(self,ypred,y):
        error = float((sum((ypred - y)**2)/y.size()[0])**0.5)
        self.error += error
    def repackage_hidden(self,hidden_state):
        return tuple(i.detach() for i in hidden_state)
    def validate(self,dataloader,model,hidden_state):
        model.eval()
        with torch.no_grad():
            for batch,(X,y) in enumerate(dataloader):
                ypred,hidden_state = model(X,hidden_state)
                self.validate_counts += 1
                self._RMSE(ypred,y)

        avg_rmse = self.error/self.validate_counts
        return avg_rmse





            




