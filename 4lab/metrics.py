from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

def multiclass_accuracy(true, predict):
    return (predict == true).sum() / len(predict)

def report(target, prediction):
    metrics = [
        ("MAE", mean_absolute_error(target, prediction)),
        ("MSE", mean_squared_error(target, prediction)),
        ("MAPE", mean_absolute_percentage_error(target, prediction)),
    ]
    for metric, value in metrics:
        print(f"{metric:>25s}: {value: >20.3f}")
