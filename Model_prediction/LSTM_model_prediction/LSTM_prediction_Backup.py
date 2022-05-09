import pandas as pd
import numpy as np
import os

from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer, Flatten
from tensorflow.keras.optimizers import Adamax, Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from collections import Counter
from PIL import Image as im
from mdn import MDN, get_mixture_loss_func,sample_from_output
from sklearn.preprocessing import MinMaxScaler
import json
import matplotlib.pyplot as plt
from keras.callbacks import History, Callback
from tensorflow.keras import metrics


def get_selected_africa_coutries_list() -> list:
    processed_path = Path(str(Path(os.getcwd()).parent.parent.absolute()) + '/ProcessedHistograms')
    country_list = sorted(processed_path.glob('*'))
    return [str(country).split('\\')[-1].replace('_', ' ') for country in country_list]


def get_water_data() -> pd.DataFrame:
    countries_list_nme = get_selected_africa_coutries_list()
    processed_path = Path(str(Path(os.getcwd()).parent.parent.absolute()) + '/WaterProcessed')
    country_list = sorted(processed_path.glob('*'))
    data = []
    for i in country_list:
        f = open(str(i))
        water_data = json.load(f)
        country_name =  str(i).split('\\')[-1].split('.')[-2][:-5].replace('_', ' ')
        water_data['country_nme'] = country_name
        if country_name in countries_list_nme:
            data.append(water_data)
    water_df = pd.DataFrame(data=data)
    water_df = water_df.set_index('country_nme')
    return water_df

def get_evdi_data():
    countries_list_nme = get_selected_africa_coutries_list()
    processed_path = Path(str(Path(os.getcwd()).parent.parent.absolute()) + '/NDVI')
    country_list = sorted(processed_path.glob('*'))

    data = []
    for i in country_list:
        if ('.' in str(i)) or (str(i).split('\\')[-1] == 'Data'):
            continue
            
        f = open(str(i))
        water_data = json.load(f)
        country_name = str(i).split('\\')[-1].split('.')[-2][:-5].replace('_', ' ')
        water_data['country_nme'] = country_name
        if country_name in countries_list_nme:
            data.append(water_data)
    water_df = pd.DataFrame(data=data)
    water_df = water_df.set_index('country_nme')
    return water_df

def get_label_data() -> [pd.DataFrame, MinMaxScaler] :
    label_path = str(Path(os.getcwd()).parent.parent.absolute()) + '/Yield_Data/all_country_crop_yield_tons_per_hectare.csv'
    df = pd.read_csv(label_path)
    df = df.set_index(['Country Name'])
    df = df.iloc[:, :-3]
    countries_list_nme = get_selected_africa_coutries_list()
    temp2 = [i in countries_list_nme for i in df.index.tolist()]
    df2 = df[temp2]
    print('Data Range Before Scale:',df2.max().max(), ' to ',  df2.min().min())
    scaler = MinMaxScaler(feature_range=(0, 2))
    temp = df2.copy().to_numpy().reshape(-1, 1)
    scaler = scaler.fit(temp)
    for i in df2.columns.tolist():
        df2.loc[:, i] = scaler.transform(df2[i].values.reshape(-1, 1))
    print('Data Range After Scale:', df2.max().max(), ' to ', df2.min().min())
    return df2, scaler


def LSTM_Model(input_instance, features_size, opt):
    n_features = 1
    m = 32
    LSTM_model = Sequential()
    LSTM_model.add(LSTM(128, return_sequences=True, input_shape=(input_instance, features_size)))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(LSTM(units=128, return_sequences=True, input_shape=(input_instance, features_size)))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(LSTM(units=128, return_sequences=False))
    LSTM_model.add(Dropout(0.2))
    # LSTM_model.add(Flatten())
    # LSTM_model.add(Dense(1))
    # LSTM_model.compile(optimizer=Adam(), loss=MeanAbsoluteError())
    # LSTM_model.compile(optimizer=Adam(learning_rate=0.01), loss=MeanSquaredError())
    LSTM_model.add(MDN(n_features, m))
    # LSTM_model.compile(loss=get_mixture_loss_func(n_features, m), optimizer=Adam(lr=0.001)) # 200 Epoch 256 LSTM
    # LSTM_model.compile(loss=get_mixture_loss_func(n_features, m), optimizer=Adamax(lr=0.0001)) # 1000 Epoch 512 LSTM
    # LSTM_model.compile(loss=get_mixture_loss_func(n_features, m), optimizer=Adamax(lr=0.00001))  # 5000 Epoch 128 LSTM 0.22
    # LSTM_model.compile(loss=get_mixture_loss_func(n_features, m), optimizer=Adamax(lr=0.0001))  # 600 Epoch 128 LSTM 0.26
    # LSTM_model.compile(loss=get_mixture_loss_func(n_features, m), optimizer=Adam(lr=0.0001))  # 10000 Epoch 128 LSTM 0.24 MAPE
    # LSTM_model.compile(loss=get_mixture_loss_func(n_features, m), optimizer=Adam(lr=0.001))  # 10000 Epoch 128 LSTM
    # LSTM_model.compile(loss=get_mixture_loss_func(n_features, m), optimizer=opt, metrics=[metrics.MeanSquaredLogarithmicError(),
    #                                                                                       metrics.MeanAbsolutePercentageError(),
    #                                                                                       metrics.RootMeanSquaredError()])
    LSTM_model.compile(loss=get_mixture_loss_func(n_features, m), optimizer=opt)
    return LSTM_model


class TestCallback(Callback):
    def __init__(self, valid_data, scaler):
        self.valid_data = valid_data
        self.loss_list = []
        self.msle_list = []
        self.rmse_list = []
        self.mape_list = []
        self.scaler = scaler

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.valid_data
        epoch_pred = self.model.predict(x)
        valid_pred = np.apply_along_axis(sample_from_output, 1, epoch_pred, 1, 32, temp=1.0).reshape(-1, 1)
        valid_y = self.scaler.inverse_transform(y.reshape(-1, 1))
        # valid_pred = self.scaler.inverse_transform(epoch_pred.reshape(-1, 1))
        valid_rmse = mean_squared_error(valid_y, valid_pred)
        valid_mape = abs((valid_y - valid_pred) / valid_y).mean()

        self.rmse_list.append(valid_rmse)
        self.mape_list.append(valid_mape)

        print(f'  Valid rmse: {valid_rmse}, mape {valid_mape}')


def LSTM_model_prediction() -> list:
    # data_test = normalize_test(data_test, 0, 1, scaler)

    df_label, scaler = get_label_data()

    opt_list = [Adam(learning_rate=0.001)
                ,Adam(learning_rate=0.0001)
                # ,Adam(learning_rate=0.00001)
                ,Adamax(learning_rate=0.001)
                ,Adamax(learning_rate=0.0001)
                # ,Adamax(learning_rate=0.00001)
                ]

    opt_list_str = ['Adam(lr=0.001)',
                'Adam(lr=0.0001)',
                'Adam(lr=0.00001)',
                'Adamax(lr=0.001)',
                'Adamax(lr=0.0001)',
                'Adamax(lr=0.00001)']

    dim_list = [(100,10), (100, 100), (10, 100)]

    epoch_list = [100]
    rmse_list = []
    mape_list = []

    rmse_list_train = []
    mape_list_train = []

    all_opt_eval_object = []
    for opt in opt_list:
        rmse_list_temp = []
        mape_list_temp = []

        rmse_list_train_temp = []
        mape_list_train_temp = []

        for epoch in epoch_list:

            train_x, train_y, test_x, valid_x, valid_y = LSTM_data_extraction_and_batching(df_label, dim_list[])

            train_x = train_x.reshape(len(train_x) , len(train_x[0]), len(train_x[0][0]))
            # test_x = test_x.reshape(len(test_x), len(test_x[0]), len(test_x[0][0]))
            valid_x = valid_x.reshape(len(valid_x), len(valid_x[0]), len(valid_x[0][0]))

            epoch_eval = TestCallback((valid_x, valid_y), scaler)

            LSTM_model = LSTM_Model(len(train_x[0]), len(train_x[0][0]), opt)
            history = LSTM_model.fit(train_x, train_y, epochs=epoch, batch_size=len(train_x),
                                     validation_data=(valid_x, valid_y),
                                     callbacks=[epoch_eval])

            all_opt_eval_object.append(epoch_eval)
    print('Finished Training')

    plt.figure(figsize=(10, 6), dpi=72)
    for i in range(len(all_opt_eval_object)):
        eval_obj = all_opt_eval_object[i]
        plt.plot(range(len(eval_obj.rmse_list)), eval_obj.rmse_list , label=opt_list_str[i])
    plt.legend()
    plt.title("RMSE comparison")
    plt.show()

    plt.figure(figsize=(10, 6), dpi=72)
    for i in range(len(all_opt_eval_object)):
        eval_obj = all_opt_eval_object[i]
        plt.plot(range(len(eval_obj.mape_list)), eval_obj.mape_list, label=opt_list_str[i])
    plt.legend()
    plt.title("MAPE comparison")
    plt.show()

            # # LSTM_pred = LSTM_model.predict(test_x)
            # valid_pred = LSTM_model.predict(valid_x)
            # train_pred = LSTM_model.predict(train_x)
            #
            # train_pred = np.array([i[0] for i in train_pred])
            # valid_pred = np.array([i[0] for i in valid_pred])
            #
            # train_y = scaler.inverse_transform(train_y.reshape(-1, 1))
            # train_pred = scaler.inverse_transform(train_pred.reshape(-1, 1))
            # valid_y = scaler.inverse_transform(valid_y.reshape(-1, 1))
            # valid_pred = scaler.inverse_transform(valid_pred.reshape(-1, 1))
            #
            # rmse = mean_squared_error(train_y, train_pred)
            # valid_rmse = mean_squared_error(valid_y, valid_pred)
            #
            # mape = abs((train_y - train_pred) / train_y).mean()
            # valid_mape = abs((valid_y - valid_pred) / valid_y).mean()
            #
            # rmse_list_temp.append(valid_rmse)
            # mape_list_temp.append(valid_mape)
            # rmse_list_train_temp.append(rmse)
            # mape_list_train_temp.append(mape)

            # print('-' * 50)
            # print('Training RMSE:', rmse, '\nValid RMSE', valid_rmse)
            # print('-' * 50)
            # print('Training MAPE:', mape, '\nValid MAPE:', valid_mape)
            # print('-' * 50)

        # rmse_list.append(rmse_list_temp)
        # mape_list.append(mape_list_temp)
        # rmse_list_train.append(rmse_list_train_temp)
        # mape_list_train.append(mape_list_train_temp)

    # print(rmse_list)
    # print(mape_list)
    # print(rmse_list_train)
    # print(mape_list_train)


    # plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'], color='red', linewidth=1.0, linestyle='--' )
    # plt.plot(range(len(history.history['loss'])), history.history['loss'], color='Blue', linewidth=1.0, linestyle='-' )
    # plt.show()


    # print(train_y[:10])
    # print(train_pred[:10])
    # print('-' * 50)
    # print(valid_y[:10])
    # print(valid_pred[:10])
    LSTM_pred = []

    return LSTM_pred



def LSTM_data_extraction_and_batching(df_label:pd.DataFrame, resize_dim) -> [list, list, list, list, list]:
    processed_path = Path(str(Path(os.getcwd()).parent.parent.absolute()) + '/ProcessedHistograms')
    country_list = sorted(processed_path.glob('*'))
    train_x, train_y, test_x, valid_x, valid_y = [], [], [], [], []
    counter, sum,counter_new, sum_new= 0, 0, 0, 0

    scaler_data = None
    for country in country_list:
        if '.' in str(country):
            continue
        data_list = sorted(country.glob('*.npy'))
        country_nme = str(country).split('\\')[-1]
        country_nme = country_nme.replace('_', ' ')

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
                data = im.fromarray(data).resize(resize_dim)
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
    print('-'*50)
    print('Training dataset sizes:', len(train_x))
    print('Validation dataset sizes:', len(valid_x))
    print('Testing dataset sizes:', len(test_x))
    # print('Number of 0:', counter)
    # print('Number of sum element:', sum)
    print('Percentage of 0:', counter/sum)
    # print('Number of 0 new:', counter_new)
    # print('Number of sum element new:', sum_new)
    print('Percentage of 0 new:', counter_new/sum_new)

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(valid_x), np.array(valid_y)


if __name__ == '__main__':
    # df_label = get_label_data()
    # temp = get_water_data()
    # train_x, train_y, test_x, valid_x, valid_y = LSTM_data_extraction_and_batching(df_label)
    LSTM_pred = LSTM_model_prediction()
    # print('Prediction:', LSTM_pred)
    a = 3