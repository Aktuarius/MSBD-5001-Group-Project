import pandas as pd
import numpy as np
import os

from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer, Flatten
from tensorflow.keras.optimizers import Adamax, Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from sklearn.metrics import mean_squared_error
from collections import Counter
from PIL import Image as im
from mdn import MDN, get_mixture_loss_func
from sklearn.preprocessing import MinMaxScaler
import json

def get_label_data() -> pd.DataFrame:
    label_path = str(Path(os.getcwd()).parent.parent.absolute()) + '/Yield_Data/all_country_crop_yield_tons_per_hectare.csv'
    df = pd.read_csv(label_path)
    df = df.set_index(['Country Name'])
    df = df.iloc[:, :-3]
    # scaler = MinMaxScaler(feature_range=(0, 50))
    # temp = df.copy().to_numpy().reshape(-1, 1)
    # scaler = scaler.fit(temp)
    # for i in df.columns.tolist():
    #     df[i] = scaler.transform(df[i].values.reshape(-1, 1))
    # df.iloc[:, :] = scaler.fit_transform(df.iloc[:, :])


    # processed_path = Path(str(Path(os.getcwd()).parent.parent.absolute()) + '/ProcessedHistograms')
    # country_list = sorted(processed_path.glob('*'))
    # countries_list_nme = [str(country).split('\\')[-1].replace('_', ' ') for country in country_list]
    # temp2 = [i in countries_list_nme for i in df.index.tolist()]
    # df2 = df[temp2]
    # print('Max value in africa countries:', df2.max().max())
    # print('Min value in africa countries:', df2.min().min())
    return df


def LSTM_Model(input_instance, features_size):
    n_features = 1
    m = 32
    LSTM_model = Sequential()
    LSTM_model.add(LSTM(256, return_sequences=True, input_shape=(input_instance, features_size)))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(LSTM(units=256, return_sequences=True, input_shape=(input_instance, features_size)))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(LSTM(units=256, return_sequences=False))
    LSTM_model.add(Dropout(0.2))
    # LSTM_model.add(Flatten())
    # LSTM_model.add(Dense(1))
    # LSTM_model.compile(optimizer=Adam(), loss=MeanAbsoluteError())
    # LSTM_model.compile(optimizer=Adam(learning_rate=0.01), loss=MeanSquaredError())
    LSTM_model.add(MDN(n_features, m))
    LSTM_model.compile(loss=get_mixture_loss_func(n_features, m), optimizer=Adam(lr=0.001))
    return LSTM_model


def LSTM_model_prediction() -> list:
    # data_test = normalize_test(data_test, 0, 1, scaler)

    df_label = get_label_data()
    train_x, train_y, test_x, valid_x, valid_y = LSTM_data_extraction_and_batching(df_label)

    train_x = train_x.reshape(len(train_x) , len(train_x[0]), len(train_x[0][0]))
    # test_x = test_x.reshape(len(test_x), len(test_x[0]), len(test_x[0][0]))
    valid_x = valid_x.reshape(len(valid_x), len(valid_x[0]), len(valid_x[0][0]))

    LSTM_model = LSTM_Model(len(train_x[0]), len(train_x[0][0]))
    LSTM_model.fit(train_x, train_y, epochs=260, batch_size=len(train_x), validation_data=(valid_x, valid_y))

    # LSTM_pred = LSTM_model.predict(test_x)
    valid_pred = LSTM_model.predict(valid_x)
    train_pred = LSTM_model.predict(train_x)

    train_pred = [i[0] for i in train_pred]
    valid_pred = [i[0] for i in valid_pred]
    rmse = mean_squared_error(train_y, train_pred)
    valid_rmse = mean_squared_error(valid_y, valid_pred)

    print('-'*50)
    print('Training RMSE:', rmse)
    print('Valid RMSE', valid_rmse)
    print('-' * 50)
    print(train_y[:10])
    print(train_pred[:10])
    print('-' * 50)
    print(valid_y[:10])
    print(valid_pred[:10])
    LSTM_pred = []
    return LSTM_pred



def LSTM_data_extraction_and_batching(df_label:pd.DataFrame) -> [list, list, list, list, list]:
    processed_path = Path(str(Path(os.getcwd()).parent.parent.absolute()) + '/ProcessedHistograms')
    country_list = sorted(processed_path.glob('*'))
    train_x, train_y, test_x, valid_x, valid_y = [], [], [], [], []
    counter = 0
    sum = 0

    counter_new = 0
    sum_new = 0

    scaler_data = None
    for country in country_list:
        if '.' in str(country):
            continue
        data_list = sorted(country.glob('*.npy'))
        country_nme = str(country).split('\\')[-1]
        country_nme = country_nme.replace('_', ' ')
        if country_nme == 'Kenya':
            continue
        try:
            country_label = df_label.loc[country_nme]
            for i in data_list:
                year = str(i).split('_')[-1][:4]
                data_array = np.load(str(i.absolute()))
                for j in data_array:
                    counter += Counter(j)[0]
                    sum += len(j)
                if scaler_data is None:
                    scaler_data = data_array
                data = np.transpose(data_array)
                data = im.fromarray(data).resize((100, 10))
                data = np.array(data)
                data = (data - np.min(scaler_data)) / (np.max(scaler_data) - np.min(scaler_data))
                # data = np.transpose(data)
                for j in data:
                    counter_new += Counter(j)[0]
                    sum_new += len(j)
                if year < '2015':
                    train_x.append(data)
                    train_y.append(country_label[year])
                elif year < '2019':
                    valid_x.append(data)
                    valid_y.append(country_label[year])
                else:
                    test_x.append(data)
        except KeyError:
            print(country_nme, 'label dataset is missing.', 'Passed ', country_nme, 'data.')
    print('Training dataset sizes:', len(train_x))
    print('Validation dataset sizes:', len(valid_x))
    print('Testing dataset sizes:', len(test_x))
    print('Number of 0:', counter)
    print('Number of sum element:', sum)
    print('Percentage of 0:', counter/sum)
    print('Number of 0 new:', counter_new)
    print('Number of sum element new:', sum_new)
    print('Percentage of 0 new:', counter_new/sum_new)

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(valid_x), np.array(valid_y)


if __name__ == '__main__':
    df_label = get_label_data()
    # train_x, train_y, test_x, valid_x, valid_y = LSTM_data_extraction_and_batching(df_label)
    LSTM_pred = LSTM_model_prediction()
    # print('Prediction:', LSTM_pred)
    a = 3