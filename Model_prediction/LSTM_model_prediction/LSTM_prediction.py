import pandas as pd
import numpy as np
import os

from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
from sklearn.metrics import mean_squared_error

def get_label_data() -> pd.DataFrame:
    label_path = str(Path(os.getcwd()).parent.parent.absolute()) + '/Yield_Data/all_country_crop_yield_tons_per_hectare.csv'
    df = pd.read_csv(label_path)
    df = df.set_index(['Country Name'])
    return df


def LSTM_Model(input_instance, features_size):
    LSTM_model = Sequential()
    LSTM_model.add(LSTM(512, return_sequences=True, input_shape=(input_instance, features_size)))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(LSTM(units=256, return_sequences=True, input_shape=(input_instance, features_size)))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(LSTM(units=128, return_sequences=False))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(Dense(1))
    LSTM_model.compile(optimizer='adam', loss='mean_squared_error')
    return LSTM_model


def LSTM_model_prediction() -> list:
    # data_test = normalize_test(data_test, 0, 1, scaler)

    df_label = get_label_data()
    train_x, train_y, test_x = LSTM_data_extraction_and_batching(df_label)

    train_x = train_x.reshape(len(train_x) , len(train_x[0]), len(train_x[0][0]))
    test_x = test_x.reshape(len(test_x), len(test_x[0]), len(test_x[0][0]))

    LSTM_model = LSTM_Model(len(train_x[0]), len(train_x[0][0]))
    LSTM_model.fit(train_x, train_y, epochs=40, batch_size=len(train_x))

    LSTM_pred = LSTM_model.predict(test_x)
    train_pred = LSTM_model.predict(train_x)

    train_pred = [i[0] for i in train_pred]
    rmse = mean_squared_error(train_y, train_pred)
    print('Training RMSE:', rmse)


    return LSTM_pred


def LSTM_data_extraction_and_batching(df_label:pd.DataFrame) -> [list, list]:
    processed_path = Path(str(Path(os.getcwd()).parent.parent.absolute()) + '/ProcessedHistograms')
    country_list = sorted(processed_path.glob('*'))
    train_x, train_y, test_x = [], [], []

    for country in country_list:
        if '.' in str(country):
            continue
        data_list = sorted(country.glob('*.npy'))
        country_nme = str(country).split('\\')[-1]
        country_label = df_label.loc[country_nme]
        for i in data_list:
            year = str(i).split('_')[-1][:4]
            data_array = np.load(str(i.absolute()))
            if year < '2019':
                train_x.append(data_array)
                train_y.append(country_label[year])
            else:
                test_x.append(data_array)

    return np.array(train_x), np.array(train_y), np.array(test_x)


if __name__ == '__main__':
    # df_label = get_label_data()
    # train_x, train_y, test_x = LSTM_data_extraction_and_batching(df_label)
    LSTM_pred = LSTM_model_prediction()
    print('Prediction:', LSTM_pred)