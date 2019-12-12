import os
import pandas as pd
from logistic_model import logistic

data_file = 'marks.csv'
data_path = os.path.join(os.path.abspath('..'), 'data', 'logistic_data', data_file)
print(data_path)


def read_data(path):
    data = pd.read_csv(data_file)
    dataMat = data.values[:, :-1]
    labelMat = data.values[:, -1]
    return dataMat, labelMat


if __name__ == "__main__":
    dataMat, labelMat = read_data(data_path)
    # print(dataMat.shape)
    # print(dataMat)
    # print(labelMat)
    model = logistic(dataMat, labelMat)
