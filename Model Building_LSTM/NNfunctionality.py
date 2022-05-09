from optparse import Option
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from pydantic import BaseModel,StrictFloat,StrictInt
from typing import Optional
from ignite.contrib.metrics.regression import MeanAbsoluteRelativeError
from ignite.engine import create_supervised_evaluator
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
    x1 : Optional[np.ndarray] = None
    x2 : Optional[np.ndarray] = None
    y : Optional[np.ndarray] = None
    hidden_size : Optional[StrictInt] = None
    input_size : Optional[StrictInt] = None
    batch_size : Optional[StrictInt] = None
    extra_features : Optional[StrictInt] = None
    time_steps : Optional[StrictInt] = None
    dense_size : Optional[StrictInt] = None
    dropout : Optional[StrictFloat] = None
    class Config:
        arbitrary_types_allowed = True

#Custom dataset class allows you to carry out transformations on your data and then create the data set
#see example here 
#https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md
class prepare_dataset(Dataset):
    def __init__(self,x1,x2,y):
        #x is timeseries histogram data
        #z is water data
        #y is yield data
        TypeChecks(x1 = x1,x2 = x2,y = y)
        self.x1 = torch.tensor(x1,dtype = torch.float32)
        self.x2 = torch.tensor(x2,dtype=torch.float32)
        self.y = torch.tensor(y,dtype = torch.float32)
        self.len = x1.shape[0]

    def __getitem__(self, index):
        return self.x1[index,:,:],self.x2[index],self.y[index]
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

#LSTM cell state will collect and keep information until it comes across a situation, directed by the forget layer, that requires it to forget the previously held data
#eg: The cell state might include the gender of the present subject so that correct pronouns can be used, but when it sees a new subject in the new input,
# it will forget hte gender of the old subject
#In the input layer, we try to decide which information should be newly added. This is done by using a sigmoid layer and a tanh layer which multiplies the outputs
#Then the output of the input gate is added to the output of the (multiplication of the output of the forget gate and the previous cell state)
#In the output gate, we take the new input from the hidden state and input sequence, passed through a sigmoid layer, then multiplied with output of the tanh layer which took in the
#addition of the forget gate * cell state  + input gate

class LSTMnetwork(nn.Module):
    def __init__(self,input_dim,hidden_dim,dense_size,batch_size,extra_features,dropout,time_steps):
        super(LSTMnetwork,self).__init__()
        TypeChecks(hidden_size = hidden_dim,input_size = input_dim,dense_size = dense_size,extra_features = extra_features,dropout = dropout,time_steps = time_steps)
        self.hidden_size  = hidden_dim
        self.input_size = input_dim
        self.time_steps = time_steps
        self.dense_size = dense_size
        self.batch_size = batch_size
        self.extra_features = extra_features
        self.dropout = dropout
        self.lstm_layer = nn.LSTM(self.input_size,self.hidden_size,num_layers = 1,dropout = self.dropout,batch_first = True)
        self.dense_layer1 = nn.Linear(in_features=self.hidden_size + self.extra_features,out_features=self.dense_size,bias = True)
        self.dense_layer2 = nn.Linear(in_features=self.dense_size,out_features=1,bias=True)
    def forward(self,x1,x2,h):
        #https://discuss.pytorch.org/t/initialization-of-first-hidden-state-in-lstm-and-truncated-bptt/58384

        #we initialize the hidden state and cell state at the start of each forward iteration for each sample. 
        #If we dont do tha the back propagation runs through old batches
        #try with detaching the hidden state and cell state
        #while in theory it is true that the back prop through time would result in running through old batches, in practice
        #it would run out of memory

        #as pytorch only applies drop out if there are 2 stacked layers, we use a hacky way of unrolling the LSTM to apply dropout at each transition
        for xt in x1.view(self.time_steps,self.input_size):
            lstm_output1,h = self.lstm_layer(xt.view(1,1,self.input_size),h)
            
            
        #Reshape from (num sequences, batch size, neuron_num) to (num sequences * batch size, neuron_num)
        # last_output = lstm_output1.contiguous().view(-1,self.hidden_size)[self.time_steps-1::self.time_steps,]
        last_output = lstm_output1.contiguous().view(-1,self.hidden_size)
        # print(lstm_output)
        x2 = torch.reshape(x2,(self.batch_size,1))
        last_output = torch.concat((last_output,x2),1)
        # print(last_output)
        output1 = self.dense_layer1(last_output)
        # print(output1)
        yield_value = self.dense_layer2(output1)
        # print(yield_value)
        return yield_value,h
    def init_hidden(self,batch):
        weights = next(self.parameters()).data
        #initializing new hidden state and cell state to be used at the start of every epoch
        hidden = (torch.nn.init.xavier_normal_(weights.new(1,batch,self.hidden_size)),torch.nn.init.xavier_normal_(weights.new(1,batch,self.hidden_size)))
        return hidden
        
#why we detach the hidden state from the parameters?
#https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426/12
class TrainingValidatingLSTM:
    def __init__(self):
        self.error = 0
        self.validate_counts = 0
        self.validating_batch_size = None
    def train(self,dataloader,model,loss_fn,optimizer,batch_size):
        model.train()
        #here a batch will be 14 years of a certain country
        for batch,(X1,X2,y) in enumerate(dataloader):
            #hidden state is instantiated at the beginning as the hidden state is not a learn parameter
            hidden_state = model.init_hidden(batch_size)
            yield_pred,hidden_state = model(X1,X2,hidden_state)
            loss = loss_fn(yield_pred,y)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # hidden_state = self.repackage_hidden(hidden_state)
    def _RMSE(self,ypred,y):
        error = float((sum((ypred - y)**2)/y.size()[0])**0.5)
        self.error += error
    def _MARE(self,ypred,y):
        self.error += torch.sum(torch.abs(ypred.reshape(y.shape) - y)/torch.abs(y))/y.shape[0]
        
    def repackage_hidden(self,hidden_state):
        return tuple(i.detach() for i in hidden_state)
    def validate(self,dataloader,model,batch_size):
        model.eval()
        with torch.no_grad():
            for batch,(X1,X2,y) in enumerate(dataloader):
                hidden_state =  model.init_hidden(batch_size)
                ypred,hidden_state = model(X1,X2,hidden_state)
                self.validate_counts += 1
                self._MARE(ypred,y)
                # self._RMSE(ypred,y)
        
        avg_error = self.error / self.validate_counts
        return avg_error,model





            




