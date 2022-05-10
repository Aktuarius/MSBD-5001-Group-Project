import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import pickle

from autogluon.tabular import TabularPredictor
from pathlib import Path
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer, Flatten
from tensorflow.keras.optimizers import Adamax, Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from collections import Counter
from PIL import Image as im
from mdn import MDN, get_mixture_loss_func,sample_from_output
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import History, Callback
from tensorflow.keras import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from sklearn.linear_model import Ridge


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

def get_ndvi_evi_data() -> [pd.DataFrame, pd.DataFrame]:
    countries_list_nme = get_selected_africa_coutries_list()
    processed_path = Path(str(Path(os.getcwd()).parent.parent.absolute()) + '/NDVI/Processed_edvi_data')
    country_list = sorted(processed_path.glob('*'))
    ndvi = []
    evi = []
    ndvi_index_list = None
    evi_index_list = None
    for country in country_list:
        data_list = sorted(country.glob('*.csv'))
        country_nme = str(country).split('\\')[-1]
        country_nme = country_nme.replace('_', ' ')
        if country_nme not in countries_list_nme:
            continue
        for i in data_list:
            file_name = str(i).split('\\')[-1].split('_')[0]
            temp_data = pd.read_csv(str(i), index_col = 0)
            temp_data.columns = [country_nme]

            if file_name == 'NDVI':
                if ndvi_index_list is None:
                    ndvi_index_list = temp_data.index.tolist()
                temp_data = temp_data.reset_index(drop=True)
                ndvi.append(temp_data)
            else:
                if evi_index_list is None:
                    evi_index_list = temp_data.index.tolist()
                temp_data = temp_data.reset_index(drop=True)
                evi.append(temp_data)
    ndvi_df = pd.concat(ndvi, axis=1)
    evi_df = pd.concat(evi, axis=1)
    ndvi_df.index = ndvi_index_list
    evi_df.index = evi_index_list
    return ndvi_df, evi_df


def ndvi_evi_monthly_to_yearly(ndvi, evi):
    ndvi = ndvi.reset_index()
    evi = evi.reset_index()

    ndvi['year'] = ndvi['index'].apply(lambda x: x[:4])
    evi['year'] = evi['index'].apply(lambda x: x[:4])

    ndvi_mean_df = ndvi.groupby(['year']).mean()
    ndvi_max_df = ndvi.groupby(['year']).max()
    ndvi_min_df = ndvi.groupby(['year']).min()

    evi_mean_df = evi.groupby(['year']).mean()
    evi_max_df = evi.groupby(['year']).max()
    evi_min_df = evi.groupby(['year']).min()

    # ndvi_mean_df = ndvi_mean_df.drop(columns=['index'])
    ndvi_max_df = ndvi_max_df.drop(columns=['index'])
    ndvi_min_df = ndvi_min_df.drop(columns=['index'])

    evi_max_df = evi_max_df.drop(columns=['index'])
    evi_min_df = evi_min_df.drop(columns=['index'])

    ndvi_mean_df = ndvi_mean_df.T
    ndvi_max_df = ndvi_max_df.T
    ndvi_min_df = ndvi_min_df.T
    evi_mean_df = evi_mean_df.T
    evi_max_df = evi_max_df.T
    evi_min_df = evi_min_df.T

    return ndvi_mean_df, ndvi_max_df, ndvi_min_df, evi_mean_df, evi_max_df, evi_min_df

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
    LSTM_model.add(Flatten())
    LSTM_model.add(Dense(1))
    LSTM_model.compile(optimizer=Adamax(learning_rate=0.001), loss=MeanAbsoluteError())
    # LSTM_model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanAbsoluteError())
    # LSTM_model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    # LSTM_model.add(MDN(n_features, m))
    # LSTM_model.compile(loss=get_mixture_loss_func(n_features, m), optimizer=Adam(lr=0.001)) # 200 Epoch 256 LSTM
    # LSTM_model.compile(loss=get_mixture_loss_func(n_features, m), optimizer=Adamax(lr=0.0001)) # 1000 Epoch 512 LSTM
    # LSTM_model.compile(loss=get_mixture_loss_func(n_features, m), optimizer=Adamax(lr=0.00001))  # 5000 Epoch 128 LSTM 0.22
    # LSTM_model.compile(loss=get_mixture_loss_func(n_features, m), optimizer=Adamax(lr=0.0001))  # 600 Epoch 128 LSTM 0.26
    # LSTM_model.compile(loss=get_mixture_loss_func(n_features, m), optimizer=Adam(lr=0.0001))  # 10000 Epoch 128 LSTM 0.24 MAPE
    # LSTM_model.compile(loss=get_mixture_loss_func(n_features, m), optimizer=Adam(lr=0.001))  # 10000 Epoch 128 LSTM
    # LSTM_model.compile(loss=get_mixture_loss_func(n_features, m), optimizer=opt, metrics=[metrics.MeanSquaredLogarithmicError(),
    #                                                                                       metrics.MeanAbsolutePercentageError(),
    #                                                                                       metrics.RootMeanSquaredError()])
    # LSTM_model.compile(loss=get_mixture_loss_func(n_features, m), optimizer=opt)
    return LSTM_model


class TestCallback(Callback):
    def __init__(self, train_data, valid_data, scaler):
        self.valid_data = valid_data
        self.train_data = train_data
        self.loss_list = []
        self.msle_list = []
        self.rmse_list = []
        self.mape_list = []
        self.scaler = scaler
        self.best_model = None
        self.min_valid_mape = 1
        self.min_train_pred = []
        self.min_valid_pred = []
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.valid_data
        epoch_pred = self.model.predict(x)
        # valid_pred = np.apply_along_axis(sample_from_output, 1, epoch_pred, 1, 32, temp=1.0).reshape(-1, 1)
        valid_y = self.scaler.inverse_transform(y.reshape(-1, 1))
        valid_pred = self.scaler.inverse_transform(epoch_pred.reshape(-1, 1))
        valid_rmse = mean_squared_error(valid_y, valid_pred, squared=False)
        valid_mape = abs((valid_y - valid_pred) / valid_y).mean()

        self.rmse_list.append(valid_rmse)
        self.mape_list.append(valid_mape)

        # results = self.model.evaluate(x, y)
        # print(results)
        if self.min_valid_mape > valid_mape:
            self.min_valid_mape = valid_mape

            # Save the model
            self.best_model = clone_model(self.model)
            self.best_model.build((None, 10))
            self.best_model.compile(optimizer=Adamax(learning_rate=0.001), loss=MeanAbsoluteError())
            self.best_model.set_weights(self.model.get_weights())

            train_x, train_y = self.train_data
            # Get the prediction first in case anythings
            self.min_train_pred = self.model.predict(train_x)
            self.min_valid_pred = self.model.predict(x)

            self.best_epoch = epoch

        print(f'  Valid rmse: {valid_rmse}, mape {valid_mape}')


def LSTM_model_prediction() -> list:
    # data_test = normalize_test(data_test, 0, 1, scaler)
    df_label, scaler = get_label_data()
    opt_list = [Adam(learning_rate=0.001)
                # , Adam(learning_rate=0.0001)
                # , Adam(learning_rate=0.00001)
                # , Adamax(learning_rate=0.001)
                # , Adamax(learning_rate=0.0001)
                # ,Adamax(learning_rate=0.00001)
                ]
    opt_list_str = ['Adam(lr=0.001)',
                'Adam(lr=0.0001)',
                'Adam(lr=0.00001)',
                'Adamax(lr=0.001)',
                'Adamax(lr=0.0001)',
                'Adamax(lr=0.00001)']
    dim_list = ['ori',
                # (100, 10),
                # (50, 50),
                # (10, 100)
                ]
    dim_eval_object = []
    history_list = []
    lowest_valid_mape = []
    epoch_list = [5000]

    for dim in dim_list:
        all_opt_eval_object = []
        for opt in opt_list:
            for epoch in epoch_list:
                train_x, train_y, test_x, valid_x, valid_y, hm_train, hm_test = LSTM_data_extraction_and_batching(df_label, dim)
                train_x = train_x.reshape(len(train_x) , len(train_x[0]), len(train_x[0][0]))
                valid_x = valid_x.reshape(len(valid_x), len(valid_x[0]), len(valid_x[0][0]))
                epoch_eval = TestCallback((train_x, train_y), (valid_x, valid_y), scaler)
                LSTM_model = LSTM_Model(len(train_x[0]), len(train_x[0][0]), opt)
                history = LSTM_model.fit(train_x, train_y, epochs=epoch, batch_size=len(train_x),
                                         validation_data=(valid_x, valid_y),
                                         callbacks=[epoch_eval])
                all_opt_eval_object.append(epoch_eval)
                history_list.append(history)
        # epoch_eval.best_model.save('D:\Saved_LSTM_model\Adam_0_001')
        print('Finished Training')

        # plt.figure(figsize=(10, 6), dpi=72)
        # for i in range(len(all_opt_eval_object)):
        #     eval_obj = all_opt_eval_object[i]
        #     plt.plot(range(len(eval_obj.rmse_list)), eval_obj.rmse_list , label=opt_list_str[i])
        # plt.legend()
        # plt.title("RMSE comparison")
        # plt.show()
        #
        # plt.figure(figsize=(10, 6), dpi=72)
        # for i in range(len(all_opt_eval_object)):
        #     eval_obj = all_opt_eval_object[i]
        #     plt.plot(range(len(eval_obj.mape_list)), eval_obj.mape_list, label=opt_list_str[i])
        # plt.legend()
        # plt.title("MAPE comparison")
        # plt.show()
        min_mape = 1
        for i in all_opt_eval_object:
            if min_mape > min(i.mape_list):
                min_mape = min(i.mape_list)
        lowest_valid_mape.append(min_mape)

        dim_eval_object.append(all_opt_eval_object)

    print('Min MAPE for all three shape', lowest_valid_mape)
    LSTM_pred = []

    return LSTM_pred

def hybrid_prediction():
    params_setting = {'bootstrap': [True, False],
                     'max_depth': [2, 3, 4, 5, 6, 7, None],
                     'max_features': ['auto', 'sqrt'],
                     'min_samples_leaf': [1, 2, 4],
                     'min_samples_split': [2, 5, 10],
                     'n_estimators': [50, 150, 200, 250, 300, 400, 500, 600, 1000, 1200, 1400, 1600, 1800, 2000]}

    df_label, scaler = get_label_data()
    train_x, train_y, test_x, valid_x, valid_y, hm_train_x, hm_valid_x = LSTM_data_extraction_and_batching(df_label, 'ori')

    df_scaler = MinMaxScaler(feature_range=(0,4)).fit(hm_train_x.iloc[:, :])
    hm_train_x.iloc[:, :] = df_scaler.transform(hm_train_x.iloc[:, :])
    hm_valid_x.iloc[:, :] = df_scaler.transform(hm_valid_x.iloc[:, :])

    model = keras.models.load_model('D:\Saved_LSTM_model\model1')

    train_pred = model.predict(train_x)
    valid_pred = model.predict(valid_x)

    hm_train_x['lstm_pred'] = train_pred
    hm_valid_x['lstm_pred'] = valid_pred

    hm_train_x['label'] = train_y
    hm_valid_x['label'] = valid_y


    # predictor = TabularPredictor(label='label', path=r'D:\automl\save', problem_type='regression').fit(hm_train_x, presets=['best_quality'], num_stack_levels=2)
    # auto_pred = predictor.predict(hm_valid_x)
    #
    # print('LSTM + AUTOML, MAPE:', abs((valid_y - auto_pred) / valid_y).mean())

    # gs_rf = GridSearchCV(estimator= RandomForestRegressor(), param_grid=params_setting,cv=5, n_jobs=-1, verbose=2)
    # gs_rf.fit(hm_train_x, train_y)


    valid_pred = gs_rf.predict(hm_valid_x)
    valid_y = scaler.inverse_transform(valid_y.reshape(-1, 1))
    valid_pred = scaler.inverse_transform(valid_pred.reshape(-1, 1))
    valid_rmse = mean_squared_error(valid_y, valid_pred, squared=False)
    valid_mape = abs((valid_y - valid_pred) / valid_y).mean()

    print('Final RMSE:', valid_rmse)
    print('Final MAPE:', valid_mape)



    return

def LSTM_data_extraction_and_batching(df_label:pd.DataFrame, resize_dim) -> [list, list, list, list, list, pd.DataFrame, pd.DataFrame]:
    processed_path = Path(str(Path(os.getcwd()).parent.parent.absolute()) + '/ProcessedHistograms')
    country_list = sorted(processed_path.glob('*'))
    train_x, train_y, test_x, valid_x, valid_y = [], [], [], [], []
    water_train_x, water_test_x = [], []
    ndvi_mean_train_x, ndvi_mean_test_x = [], []
    ndvi_max_train_x, ndvi_max_test_x = [], []
    ndvi_min_train_x, ndvi_min_test_x = [], []
    evi_mean_train_x, evi_mean_test_x = [], []
    evi_max_train_x, evi_max_test_x = [], []
    evi_min_train_x, evi_min_test_x = [], []
    year_train_x, year_test_x = [], []

    counter, sum,counter_new, sum_new= 0, 0, 0, 0

    water_data = get_water_data()
    ndvi, evi =  get_ndvi_evi_data()
    ndvi_mean_df, ndvi_max_df, ndvi_min_df, evi_mean_df, evi_max_df, evi_min_df = ndvi_evi_monthly_to_yearly(ndvi, evi)

    scaler_data = None
    for country in country_list:
        if '.' in str(country):
            continue
        data_list = sorted(country.glob('*.npy'))
        country_nme = str(country).split('\\')[-1]
        country_nme = country_nme.replace('_', ' ')

        # if country_nme == 'Madagascar':
        #     print('333')
        try:
            country_label = df_label.loc[country_nme]
            water_label = water_data.loc[country_nme]
            ndvi_mean_label = ndvi_mean_df.loc[country_nme]
            ndvi_max_label = ndvi_max_df.loc[country_nme]
            ndvi_min_label = ndvi_min_df.loc[country_nme]
            evi_mean_label = evi_mean_df.loc[country_nme]
            evi_max_label = evi_max_df.loc[country_nme]
            evi_min_label = evi_min_df.loc[country_nme]

            for i in data_list:
                year = str(i).split('_')[-1][:4]
                data_array = np.load(str(i.absolute()))
                for j in data_array:
                    counter += Counter(j)[0]
                    sum += len(j)
                if scaler_data is None:
                    scaler_data = data_array
                data = np.transpose(data_array)
                if resize_dim != 'ori':
                    data = im.fromarray(data).resize(resize_dim)
                    data = np.array(data)
                for j in data:
                    counter_new += Counter(j)[0]
                    sum_new += len(j)
                if year < '2015':
                    train_x.append(data)
                    train_y.append(country_label[year])

                    water_train_x.append(water_label[year])
                    ndvi_mean_train_x.append(ndvi_mean_label[year])
                    ndvi_max_train_x.append(ndvi_max_label[year])
                    ndvi_min_train_x.append(ndvi_min_label[year])
                    evi_mean_train_x.append(evi_mean_label[year])
                    evi_max_train_x.append(evi_max_label[year])
                    evi_min_train_x.append(evi_min_label[year])
                    year_train_x.append(int(year))

                elif year < '2019':
                    valid_x.append(data)
                    valid_y.append(country_label[year])

                    water_test_x.append(water_label[year])
                    ndvi_mean_test_x.append(ndvi_mean_label[year])
                    ndvi_max_test_x.append(ndvi_max_label[year])
                    ndvi_min_test_x.append(ndvi_min_label[year])
                    evi_mean_test_x.append(evi_mean_label[year])
                    evi_max_test_x.append(evi_max_label[year])
                    evi_min_test_x.append(evi_min_label[year])
                    year_test_x.append(int(year))

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
    # print('Percentage of 0:', counter/sum)
    # print('Number of 0 new:', counter_new)
    # print('Number of sum element new:', sum_new)
    # print('Percentage of 0 new:', counter_new/sum_new)

    hybrid_model_train_data = pd.DataFrame(data={
                                                 # 'water':water_train_x,
                                                 'ndvi_mean':ndvi_mean_train_x,
                                                 # 'ndvi_max':ndvi_max_train_x,
                                                 # 'ndvi_min':ndvi_min_train_x,
                                                 'evi_mean':evi_mean_train_x,
                                                 # 'evi_max':evi_max_train_x,
                                                 # 'evi_min':evi_min_train_x,
                                                 'year': year_train_x
                                                 })

    hybrid_model_test_data = pd.DataFrame(data={
                                                 # 'water':water_test_x,
                                                 'ndvi_mean':ndvi_mean_test_x,
                                                 # 'ndvi_max':ndvi_max_test_x,
                                                 # 'ndvi_min':ndvi_min_test_x,
                                                 'evi_mean':evi_mean_test_x,
                                                 # 'evi_max':evi_max_test_x,
                                                 # 'evi_min':evi_min_test_x,
                                                 'year':year_test_x
                                                 })

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(valid_x), np.array(valid_y), hybrid_model_train_data, hybrid_model_test_data


if __name__ == '__main__':
    df_label, scaler = get_label_data()
    # temp = get_water_data()
    train_x, train_y, test_x, valid_x, valid_y, hm_train_x, hm_valid_x = LSTM_data_extraction_and_batching(df_label, 'ori')

    df_scaler = MinMaxScaler(feature_range=(0,4)).fit(hm_train_x.iloc[:, :])
    hm_train_x.iloc[:, :] = df_scaler.transform(hm_train_x.iloc[:, :])
    hm_valid_x.iloc[:, :] = df_scaler.transform(hm_valid_x.iloc[:, :])

    params_setting = {'alpha': [i/10000 for i in range(1, 100000, 100)]}

    gs_ridge = GridSearchCV(estimator= Ridge(), param_grid=params_setting,cv=5, n_jobs=-1, verbose=2)
    gs_ridge.fit(hm_train_x, train_y)


    valid_pred = gs_ridge.predict(hm_valid_x)

    valid_y = scaler.inverse_transform(valid_y.reshape(-1, 1))
    valid_pred = scaler.inverse_transform(valid_pred.reshape(-1, 1))
    valid_rmse = mean_squared_error(valid_y, valid_pred, squared=False)
    valid_mape = abs((valid_y - valid_pred) / valid_y).mean()

    print('Best estimator', gs_ridge.best_estimator_)
    print('Best score', gs_ridge.best_score_)
    print('RMSE', RMSE)
    print('MAPE', MAPE)

    # LSTM_pred = LSTM_model_prediction()
    # print('Prediction:', LSTM_pred)
    # hybrid_prediction()
    a = 3