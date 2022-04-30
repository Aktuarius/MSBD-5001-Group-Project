import pandas as pd
import numpy as np


if __name__ == '__main__':
     # data = pd.read_csv('all_country_crop_yield.csv')
     # print(data.head())
     # cols = data.columns.to_list()
     # crop_type_list = [i.split('_')[0] for i in cols if 'gap' in i]
     data = pd.read_csv('yield.csv')
     print(data.head())
     columns = data.columns.to_list()[:1] + data.columns.to_list()[-20:-3]
     data_sel = data[columns]
     data_sel = data_sel.dropna().reset_index(drop = True)
     for i in columns[1:]:
          data_sel[i] = data_sel[i] / 1000


