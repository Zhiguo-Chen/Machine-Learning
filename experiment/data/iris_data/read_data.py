from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris()
# print(iris['data'])
X = iris['data'][:, [2, 3]]
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
