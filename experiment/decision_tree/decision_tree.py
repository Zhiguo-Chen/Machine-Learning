import os
import sys
import random
import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath('../../')
sys.path.append(ROOT_DIR)
from experiment.data.iris_data.read_data import *
from experiment.plot_decision import plot_decision_regions

# print(X_train)
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
# print(X_combined)
# print(y_combined)
plot_decision_regions(X_combined, y_combined,
                      classifier=tree, test_idx=range(105, 150))
plt.xlabel('petal length[cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()
