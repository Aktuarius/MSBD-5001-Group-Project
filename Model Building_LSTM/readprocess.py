import pathlib
import numpy as np
import pandas as pd
from pathlib import Path, WindowsPath
from pydantic import BaseModel,StrictStr
from typing import Optional,Dict
import re

class TypeChecks(BaseModel):
    featurepath : Optional[pathlib.WindowsPath] = None
    yieldpath : Optional[pathlib.WindowsPath] = None
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

    def __init__(self,featurepath,yieldpath):
        self.featurepath = featurepath
        self.yieldpath = yieldpath
        self.read_feature_data(self.featurepath)
        self.read_yield_data(self.yieldpath)

    def read_feature_data(self,filepath):
        year_path = map(lambda x:(re.findall(r"_([0-9]+).",str(x))[0],str(x)),filepath.iterdir())
        self.data_dict = dict()
        for i in year_path:
            self.data_dict[i[0]] = np.load(i[1])
        self.data_dict = {year:np.transpose(i) for year,i in self.data_dict.items()}
        self.process_final_dataset(self.data_dict)


    #default way that pytorch accepts is (seq,batch,feature)
    #if batch_first = True, (batch, seq, feature) 
    def process_final_dataset(self,data_dict):
        #(seq,batch,feature)
        shp = data_dict[list(data_dict.keys())[0]].shape
        self.final_dataset = np.zeros(shape = [len(data_dict),shp[0],shp[1]])
        for year,i in zip(data_dict.keys(),range(len(data_dict))):
            self.final_dataset[i,:,:] = data_dict[year]
    
    def read_yield_data(self,filepath):
        self.yield_df = pd.read_csv(filepath,delimiter="\t")
        self.yield_df["Total"] = self.yield_df["Cereal Yield"] + self.yield_df["Wheat Yield"] + self.yield_df["Maize Yield"] + self.yield_df["Rice Yield"]
        self.yield_dict = {key:np.array(grp["Total"]) for key,grp in self.yield_df.groupby("Country",sort = False)}

