@@ -9,7 +9,7 @@ from pathlib import Path

# the model calss
class Deep_Neural_network(torch.nn.Module):
    def __init__(self, input_n:int, output_n:int,  device):
    def __init__(self, input_n:int, output_n:int,  device:torch.device):
        super(Deep_Neural_network, self).__init__()
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(input_n, 2048, bias=True),
@ -39,10 +39,10 @@ class Deep_Neural_network(torch.nn.Module):


# define the model for prediction
def deep_neural_network(train_X:np.ndarray, train_y:np.ndarray, test_x:np.ndarray, learning_rate:float, iter:int) -> list :
def deep_neural_network(train_X:np.ndarray, train_y:np.ndarray, test_x:np.ndarray, learning_rate:float, iter:int) -> [list, list]:
    device = torch.device('cuda')

    print(type(device))
    # print(type(device))

    loss = torch.nn.MSELoss()

@ -73,7 +73,9 @@ def deep_neural_network(train_X:np.ndarray, train_y:np.ndarray, test_x:np.ndarra
        training_RMSE.append(mean_squared_error(train_y.data.cpu().numpy(), train_y_pred.data.cpu().numpy(), squared=False))

    testing_y_pred = model(test_x)
    return testing_y_pred.data.cpu().numpy()
    train_pred = model(train_X)

    return testing_y_pred.data.cpu().numpy(), train_pred.data.cpu().numpy()


def get_train_test_data(df_label:pd.DataFrame) -> [list, list, list]:
@ -96,7 +98,7 @@ def get_train_test_data(df_label:pd.DataFrame) -> [list, list, list]:
            else:
                test_x.append(data_array)

    return np.array(train_x), np.array(train_y), np.array(test_x)
    return np.array(train_x).reshape(len(train_x), -1), np.array(train_y).reshape(-1, 1), np.array(test_x).reshape(len(test_x), -1)


def DNN_model_prediction() -> list:
@ -116,19 +118,16 @@ def DNN_model_prediction() -> list:
    for i in range(20, 50, 5):
        for j in range(100, 700, 10):

            results, model, train_predict = deep_neural_network(train_x, train_y, test_x, i / 10000, j)
            test_predict , train_predict = deep_neural_network(train_x, train_y, test_x, i / 10000, j)
            RMSE = mean_squared_error(train_y, train_predict, squared=False)
            print(i, j, RMSE)
            if (RMSE < min_rmse):  # or (RMSE < RMSE_3MA):
                min_rmse = RMSE
                setting = i / 10000, j
                min_test_predict = results
                min_test_predict = test_predict
                min_train_predict = train_predict
                best_model = model

    test_pred = []

    return test_pred
    return min_test_predict

def get_label_data() -> pd.DataFrame:
    label_path = str(Path(os.getcwd()).parent.parent.absolute()) + '/Yield_Data/all_country_crop_yield_tons_per_hectare.csv'




if __name__ == '__main__':
    # df_label = get_label_data()
    # train_x, train_y, test_x = LSTM_data_extraction_and_batching(df_label)
    # LSTM_pred = LSTM_model_prediction()
    # print('Prediction:', LSTM_pred)
    # df_label = get_label_data()
    # train_x, train_y, test_x = get_train_test_data(df_label)

    test_pred = DNN_model_prediction()
