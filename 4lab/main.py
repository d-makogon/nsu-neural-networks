import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from preprocess import preprocess_data
from dataset import train_val_test_split
from dataset import cut_df_to_dataset
from rnn_lstm import setup_optimizers, RNN_Model, LSTM_Model
from optimizers import Adam
from metrics import report


def trainer(x_train, y_train, x_val, y_val, epoch_num, learning_rate, model, nn_name):
    mempool = cp.get_default_memory_pool()
    num_train = len(x_train)
    batch_size = len(x_train)
    x_val_gpu = cp.array(x_val)
    y_val_gpu = cp.array(y_val)
    best_val_loss = 9999999
    for epoch in range(epoch_num):
        shuffled_indices = np.arange(num_train)
        np.random.shuffle(shuffled_indices)
        sections = np.arange(batch_size, num_train, batch_size)
        batches_indices = np.array_split(shuffled_indices, sections)
        batch_losses = np.zeros(len(batches_indices))
        for batch_id, batch_indices in enumerate(batches_indices):
            batch_X = x_train[batch_indices]
            batch_y = y_train[batch_indices]
            batch_y_gpu = cp.asarray(batch_y)
            batch_X_gpu = cp.asarray(batch_X)
            out = model.forward(batch_X_gpu)
            train_loss = cp.mean((out - batch_y_gpu) ** 2)
            grad = out - batch_y_gpu
            for param in model.params().values():
                param.grad.fill(0)
            model.backward(grad)
            for param_name, param in model.params().items():
                optimizer = model.optimizers[param_name]
                optimizer.update(param.value, param.grad, learning_rate)
            batch_losses[batch_id] = train_loss.get()
            mempool.free_all_blocks()
        val_out = model.forward(x_val_gpu)
        val_loss = cp.mean((val_out - y_val_gpu) ** 2)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            for param_name, param in model.params().items():
                cp.save(f"best_model_{nn_name}/{param_name}.npy", param.value)
        if epoch % 25 == 0:
            print(
                f"Epoch {epoch}: Loss = {batch_losses.mean():.5f}, val loss = {val_loss:.5f}"
            )


def predict_visualization(true, predict):
    plt.figure(figsize=(30, 10))
    plt.plot(predict, color='y', label="predict")
    plt.plot(true, color='b', label="true")
    plt.legend()
    plt.show()
    plt.savefig('res.png')


def preprocess_data(data):
    week = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }
    data["Day_of_week"] = data["Day_of_week"].map(week)
    data["WeekStatus"] = data["WeekStatus"].map({"Weekday": 0, "Weekend": 1})
    data["Load_Type"] = data["Load_Type"].map(
        {"Light_Load": 0, "Medium_Load": 1, "Maximum_Load": 2}
    )
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    day_len_seconds = 24 * 60 * 60
    year_len_seconds = (365.2425) * day_len_seconds

    data["Time_sin"] = np.sin(
        (data["date"] - data["date"].min()).dt.total_seconds()
        * (2 * np.pi / day_len_seconds)
    )
    data["Date_sin"] = np.sin(
        (data["date"] - data["date"].min()).dt.total_seconds()
        * (2 * np.pi / year_len_seconds)
    )
    data = data.drop(columns=["date"])

    feature_columns = list(data.columns)
    feature_columns.remove("Usage_kWh")
    data[feature_columns] = MinMaxScaler().fit_transform(data[feature_columns])
    return data


data = pd.read_csv('Steel_industry_data.csv')
data.head()
data.describe()
plt.plot(data['Usage_kWh'])
data.columns[data.dtypes == object]
data = preprocess_data(data)
data_train, data_val, data_test = train_val_test_split(data)
x_train, y_train = cut_df_to_dataset(data_train, 20, 'Usage_kWh')
x_val, y_val = cut_df_to_dataset(data_val, 20, 'Usage_kWh')
x_test, y_test = cut_df_to_dataset(data_test, 20, 'Usage_kWh')
model = RNN_Model(12, 100)
model.optim = Adam()
setup_optimizers(model)
epoch_num = 7500
learning_rate = 1e-3
trainer(x_train, y_train, x_val, y_val, epoch_num, learning_rate, model, "RNN")
best_model = RNN_Model(12, 100)
best_model.load_params('best_model_RNN')
test = cp.array(x_test)
predict = best_model.forward(test)
predict = predict.get()
report(predict, y_test)
model = LSTM_Model(12, 20)
model.optim = Adam()
setup_optimizers(model)
trainer(x_train, y_train, x_val, y_val, epoch_num, learning_rate, model, "LSTM")
best_model = LSTM_Model(12, 20)
best_model.load_params('best_model_LSTM')
test = cp.array(x_test)
predict = best_model.forward(test)
predict = predict.get()
print("Finished")
report(predict, y_test)
predict_visualization(y_test, predict)
