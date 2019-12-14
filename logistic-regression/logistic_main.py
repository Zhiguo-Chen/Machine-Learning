import os
import pandas as pd
from logistic_model import logistic
from sklearn.model_selection import train_test_split

data_file = 'marks.csv'
data_path = os.path.join(os.path.abspath('.'), 'data', data_file)
print(data_path)


def read_data(path):
    data = pd.read_csv(path)
    dataMat = data.values[:, :-1]
    labelMat = data.values[:, -1]
    return dataMat, labelMat


if __name__ == "__main__":
    dataMat, labelMat = read_data(data_path)
    X_train, X_test, Y_train, Y_test = train_test_split(dataMat, labelMat, test_size=0.2)
    model = logistic(X_train, Y_train)
    model.fit()
    Y_pred = model.predit(X_test)
    correct_list = []
    for i in range(len(Y_test)):
        if Y_test[i] == Y_pred[i]:
            correct_list.append(Y_pred[i])
    accuracy = len(correct_list) / len(Y_test)
    print(accuracy)
