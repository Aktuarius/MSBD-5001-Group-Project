import numpy as np
import pandas as pd
import torch
from torch import nn
from pathlib import Path
import re
from readprocess import readprocess
from NNfunctionality import *
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.preprocessing import MinMaxScaler
from itertools import product

#Creates an instance to process all the data and give train and test dataset
#train on 13 years and predict on final year
def processdata(hist_path,yield_path,water_path):
    data_class = readprocess(featurepath=hist_path,yieldpath=yield_path,waterpath = water_path,train_yearcount=13,test_yearcount=1,num_timesteps=46,num_features=576)
    return data_class

#Extracts the data that was processed in the processdata function
#Also normalizes water data
def create_train_test(data_class):
    #scale the water data
    scaler = MinMaxScaler()
    water_array_train = data_class.water_array_train.reshape((-1,1))
    water_array_test = data_class.water_array_test.reshape((-1,1))
    scaler.fit(water_array_train)
    water_array_train = scaler.transform(water_array_train).reshape((len(data_class.country_list)*13,))
    water_array_test = scaler.transform(water_array_test).reshape((len(data_class.country_list)*1,))
    #set the X data and yield data for Train and Test sets
    X_train = data_class.dataset_train_resized
    y_train = data_class.yield_array_train
    X_test = data_class.dataset_test_resized
    y_test = data_class.yield_array_test
    return X_train,water_array_train,y_train,X_test,water_array_test,y_test


def create_dataloaders(X_train,water_array_train,y_train,X_test,water_array_test,y_test):
    dataset_train = prepare_dataset(X_train,water_array_train,y_train)
    dataloader_train = DataLoader(dataset_train,batch_size=1,shuffle=False,batch_sampler=None)
    dataset_test = prepare_dataset(X_test,water_array_test,y_test)
    dataloader_test = DataLoader(dataset_test,batch_size=1,shuffle=False,batch_sampler=None)
    return dataloader_train,dataloader_test

def train_and_validate(dataloader_train,dataloader_test):
    hidden_units = [50,100]
    dense_units = [200,250]
    dropout = [0.15,0.25]

    combinations = list(product([i for i,j in enumerate(hidden_units)], [i for i,j in enumerate(dense_units)],[i for i,j in enumerate(dropout)]))
    parameter_list = list()

    for combination in combinations:
        parameter_list.append([hidden_units[combination[0]],dense_units[combination[1]],dropout[combination[2]]])
    results_dict = {}
    for parameter_set in parameter_list:
        epoch_no = 40
        hidden_units = parameter_set[0]
        dense_units = parameter_set[1]
        dropout = parameter_set[2]
        trainvalidate = TrainingValidatingLSTM()
        model = LSTMnetwork(32,hidden_dim = hidden_units,dense_size = dense_units,batch_size = 1,extra_features = 1,dropout = dropout,time_steps=46)
        mse_loss = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(),lr = 0.001)
        for i in range(epoch_no):
            trainvalidate.train(dataloader=dataloader_train,model=model,loss_fn=mse_loss,optimizer = optimizer,batch_size=1)

        avg_error_valid,model = trainvalidate.validate(dataloader=dataloader_test,model = model,batch_size =1)
        avg_error_train,model = trainvalidate.validate(dataloader=dataloader_train,model = model,batch_size =1)
        print(avg_error_valid,avg_error_train)
        results_dict[f"Hidden Units : {hidden_units},Dense Units : {dense_units}, Dropout : {dropout}"] = (avg_error_valid,model)
    return results_dict


if __name__ == "__main__":
    hist_path = Path(Path.cwd(),"..","ProcessedHistograms").resolve()
    yield_path = Path(Path.cwd(),"..","Yield Data","all_country_crop_yield_tons_per_hectare.csv").resolve()
    water_path = Path(Path.cwd(),"..","WaterProcessed").resolve()

    data_class = prepare_dataset(hist_path,yield_path,water_path)

    X_train,water_array_train,y_train,X_test,water_array_test,y_test = create_train_test(data_class)

    dataloader_train,dataloader_test = create_dataloaders(X_train,water_array_train,y_train,X_test,water_array_test,y_test)

    results_dict = train_and_validate(dataloader_train,dataloader_test)

