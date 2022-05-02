import pandas as pd
import numpy as np
import os
import torch

from sklearn.metrics import mean_squared_error
from pathlib import Path


# the model calss
class Deep_Neural_network(torch.nn.Module):
    def __init__(self, input_n:int, output_n:int,  device):
        super(Deep_Neural_network, self).__init__()
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(input_n, 2048, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2, inplace=False),
            torch.nn.Linear(2048, 2048 , bias=True),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(0.2, inplace=False),
            torch.nn.Linear(2048, 512, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2, inplace=False),
            torch.nn.Linear(512, 512, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(0.2, inplace=False),
            torch.nn.Linear(512, 128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2, inplace=False),
            torch.nn.Linear(128, 128, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(0.2, inplace=False),
            torch.nn.Linear(128, output_n, bias=True)
        ).to(device)

    def forward(self, x):
        x = self.stack(x)
        return x


# define the model for prediction
def deep_neural_network(train_X:np.ndarray, train_y:np.ndarray, test_x:np.ndarray, learning_rate:float, iter:int) -> list :
    device = torch.device('cuda')

    print(type(device))

    loss = torch.nn.MSELoss()

    # Load Train data
    train_X = torch.from_numpy(train_X).type(torch.FloatTensor).cuda()
    train_y = torch.from_numpy(train_y).type(torch.FloatTensor).cuda()
    test_x = torch.from_numpy(test_x).type(torch.FloatTensor).cuda()

    model = Deep_Neural_network(len(train_X[0]), len(train_y[0]), device)

    # Optimal function
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_loss = []
    training_RMSE = []

    # Epoch to train
    for iter_num in range(iter):
        cnt = 0
        cnt += 1
        model.train()
        train_y_pred = model(train_X)
        loss_train_val = loss(train_y_pred, train_y)
        training_loss.append(loss_train_val.item())
        opt.zero_grad()
        loss_train_val.backward()
        opt.step()

        training_RMSE.append(mean_squared_error(train_y.data.cpu().numpy(), train_y_pred.data.cpu().numpy(), squared=False))

    testing_y_pred = model(test_x)
    return testing_y_pred.data.cpu().numpy()


def get_train_test_data(df_label:pd.DataFrame) -> [list, list, list]:
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
            data_array = np.load(str(i.absolute())).reshape(-1, 1)
            if year < '2019':
                train_x.append(data_array)
                train_y.append(country_label[year])
            else:
                test_x.append(data_array)

    return np.array(train_x), np.array(train_y), np.array(test_x)


def DNN_model_prediction() -> list:

    df_label = get_label_data()
    train_x, train_y, test_x = get_train_test_data(df_label)

    # data_test = normalize_test(data_test, 0, 1, scaler_x)

    # train_x, train_y = pre_training_process_v2('DNN', data_scaled, predict_cnt)

    min_rmse = 10000
    setting = [0, 0]
    min_results = []
    best_model = None

    for i in range(20, 50, 5):
        for j in range(100, 700, 10):

            results, model, train_predict = deep_neural_network(train_x, train_y, test_x, i / 10000, j)
            RMSE = mean_squared_error(train_y, train_predict, squared=False)
            print(i, j, RMSE)
            if (RMSE < min_rmse):  # or (RMSE < RMSE_3MA):
                min_rmse = RMSE
                setting = i / 10000, j
                min_test_predict = results
                min_train_predict = train_predict
                best_model = model

    test_pred = []

    return test_pred

def get_label_data() -> pd.DataFrame:
    label_path = str(Path(os.getcwd()).parent.parent.absolute()) + '/Yield_Data/all_country_crop_yield_tons_per_hectare.csv'
    df = pd.read_csv(label_path)
    df = df.set_index(['Country Name'])
    return df



if __name__ == '__main__':
    # df_label = get_label_data()
    # train_x, train_y, test_x = LSTM_data_extraction_and_batching(df_label)
    # LSTM_pred = LSTM_model_prediction()
    # print('Prediction:', LSTM_pred)
    # df_label = get_label_data()
    # train_x, train_y, test_x = get_train_test_data(df_label)
    test_pred = DNN_model_prediction()

