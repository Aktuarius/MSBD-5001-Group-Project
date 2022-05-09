import pathlib
import numpy as np
import pandas as pd
from pathlib import Path, WindowsPath
from pydantic import BaseModel,StrictStr
from typing import Optional,Dict
import re
import json

from pydantic.types import StrictInt

class TypeChecks(BaseModel):
    featurepath : Optional[pathlib.WindowsPath] = None
    yieldpath : Optional[pathlib.WindowsPath] = None
    waterpath : Optional[pathlib.WindowsPath] = None
    yearcount : Optional[StrictInt] = None
    num_timesteps : Optional[StrictInt] = None
    num_features : Optional[StrictInt] = None

    class Config:
        arbitrary_types_allowed = True


class readprocess:
    @property
    def featurepath(self):
        return self._featurepath
    @featurepath.setter
    def featurepath(self,pathvalue):
        TypeChecks(featurepath = pathvalue)
        self._featurepath = pathvalue
    @property
    def yieldpath(self):
        return self._yieldpath
    @yieldpath.setter
    def yieldpath(self,pathvalue):
        TypeChecks(yieldpath = pathvalue)
        self._yieldpath = pathvalue
    @property
    def waterpath(self):
        return self._waterpath
    @waterpath.setter
    def waterpath(self,pathvalue):
        TypeChecks(waterpath = pathvalue)
        self._waterpath = pathvalue
    @property
    def yearcount(self):
        return self._yearcount
    @yearcount.setter
    def yearcount(self,value):
        TypeChecks(yearcount = value)
        self._yearcount = value
    @property
    def num_timesteps(self):
        return self._num_timesteps
    @num_timesteps.setter
    def num_timesteps(self,value):
        TypeChecks(num_timesteps = value)
        self._num_timesteps = value
    @property
    def num_features(self):
        return self._num_features
    @num_features.setter
    def num_features(self,value):
        TypeChecks(num_features = value)
        self._num_features = value


    def __init__(self,featurepath,yieldpath,waterpath,train_yearcount,test_yearcount,num_timesteps,num_features):
        self.featurepath = featurepath
        self.waterpath = waterpath
        self.yieldpath = yieldpath
        self.train_yearcount = train_yearcount
        self.test_yearcount = test_yearcount
        self.num_timesteps = num_timesteps
        self.num_features = num_features
        self.process_final_dataset(self.featurepath)
    # def read_feature_data(self,filepath):
    #     year_path = map(lambda x:(re.findall(r"_([0-9]+).",str(x))[0],str(x)),filepath.iterdir())
    #     self.data_dict = dict()
    #     for i in year_path:
    #         self.data_dict[i[0]] = np.load(i[1])
    #     self.data_dict = {year:np.transpose(i) for year,i in self.data_dict.items()}
    #     self.process_final_dataset(self.data_dict)


    #default way that pytorch accepts is (seq,batch,feature)
    #if batch_first = True, (batch, seq, feature) 
    def process_final_dataset(self,filepath):
        #(seq,batch,feature)
        #Selects only countries in the yield list
        self.country_list_features = [str(i).split("\\")[-1] for i in filepath.iterdir()]
        self.yield_country_list = list(pd.read_csv(self.yieldpath)["Country Name"])
        self.country_list_final = [i for i in filepath.iterdir() if str(i).split("\\")[-1] in self.yield_country_list]
        self.country_num = len(self.country_list_final)
        self.left_out_countries = [i for i in self.country_list_features if i not in self.yield_country_list]
        print(" , ".join(self.left_out_countries))
        self.dataset_train = np.zeros(shape = [self.country_num * self.train_yearcount,self.num_timesteps,self.num_features])
        self.dataset_test = np.zeros(shape = [self.country_num * self.test_yearcount,self.num_timesteps,self.num_features])
        k = 0
        l = 0
        self.country_list = list()
        for i in self.country_list_final:
            self.country_list.append(str(i).split("\\")[-1])
            for j in list(i.iterdir())[:-1]:
                self.dataset_train[k,:,:] = np.load(j).transpose()
                k+=1
            self.dataset_test[l,:,:] = np.load(list(i.iterdir())[-1]).transpose()
            l+=1
        #As the data is very sparse with many zeros, we shall sum over periods to get less sparse data(essentially we are rebinning)
        self.resize_training_data(18)
        self.read_yield_data(self.yieldpath)
        self.read_water_data(self.waterpath)

    def resize_training_data(self,sum_period):
        self.dataset_train_resized = np.zeros(shape = (self.country_num * self.train_yearcount,self.num_timesteps,int(self.num_features/sum_period)))
        self.dataset_test_resized = np.zeros(shape = (self.country_num * self.test_yearcount,self.num_timesteps,int(self.num_features/sum_period)))
        for i in range(self.dataset_train.shape[0]):
            self.dataset_train_resized[i,:,:] = np.add.reduceat(self.dataset_train[i,:,:],indices=list(range(0,self.num_features,sum_period)),axis = 1)
        for j in range(self.dataset_test.shape[0]):
            self.dataset_test_resized[j,:,:] = np.add.reduceat(self.dataset_test[j,:,:],indices=list(range(0,self.num_features,sum_period)),axis = 1)

    def read_yield_data(self,filepath):
        self.yield_df = pd.read_csv(filepath)
        self.yield_dict_train = {country:self.yield_df.loc[self.yield_df["Country Name"] == country].iloc[:,1:self.train_yearcount + 1].to_numpy().reshape((self.train_yearcount,)) for country in self.country_list}
        self.yield_dict_test = {country:self.yield_df.loc[self.yield_df["Country Name"] == country].iloc[:,self.train_yearcount + 1].to_numpy().reshape((self.test_yearcount,)) for country in self.country_list}
        self.yield_df_train = pd.DataFrame(self.yield_dict_train)
        self.yield_df_test = pd.DataFrame(self.yield_dict_test)
        #make it to a single array

        self.yield_array_train = self.yield_df_train.to_numpy().reshape((self.train_yearcount*self.country_num,),order = 'F')
        self.yield_array_test = self.yield_df_test.to_numpy().reshape((self.test_yearcount*self.country_num,),order = 'F')
    def read_water_data(self,filepath):
        #All available waterfiles paths
        self.waterfile_names = [(re.findall(r"[A-Za-z\s]+Water",str(i).split("\\")[-1])[0][:-5],i) for i in filepath.iterdir()]
        #Selecting the filepaths that are in self.country_list
        self.required_paths = [i[1] for i in self.waterfile_names if i[0] in self.country_list]
        self.water_array_train = np.zeros(shape = (len(self.required_paths)*self.train_yearcount,))
        self.water_array_test = np.zeros(shape = (len(self.required_paths)*self.test_yearcount,))
        for i,path in zip(range(len(self.required_paths)),self.required_paths):
            with open(str(path)) as f:
                data = json.load(f)
            self.water_array_train[i*self.train_yearcount:(i+1)*self.train_yearcount] = np.array([i[1] for i in list(data.items())[:self.train_yearcount]])
            self.water_array_test[i*self.test_yearcount:(i+1)*self.test_yearcount] = np.array([i[1]for i in list(data.items())[self.train_yearcount:]])

        


        

        

