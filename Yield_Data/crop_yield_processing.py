import pandas as pd
import numpy as np


if __name__ == '__main__':
     data = pd.read_csv('yield.csv')
     columns = data.columns.to_list()[:1] + data.columns.to_list()[-20:-3]
     data_sel = data[columns]
     data_sel = data_sel.dropna().reset_index(drop = True)
     for i in columns[1:]:
          data_sel[i] = data_sel[i] / 1000
     print('Processed crop yield data. Sample as follow:')
     print(data_sel.head())
     data_sel.to_csv('all_country_crop_yield_tons_per_hectare.csv', index=False)
     print('Saved Crop Yield data.')


