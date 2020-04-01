import warnings
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import numpy as np


warnings.filterwarnings("ignore")

csvFile = "students_data.csv"
data = pd.read_csv(csvFile, encoding='utf-8', header=0)

x = data.values[:, 0:-1]
y = data.values[:, -1]

logreg = LogisticRegression()
kf = KFold(n_splits=5, random_state=None, shuffle=True)
kf.get_n_splits(x)

acc = 0
for train_index, test_index in kf.split(x):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    acc = acc + accuracy_score(y_test, y_pred)

print('Train/Test split results:')
print("准确率为" + str(acc / 5))

array = [[3, 99, 3, 94, 3, 99, 3, 72, 3, 85, 4, 86, 4, 86, 3, 87, 3, 86, 3, 88, 1]]
pre = logreg.predict(np.array(array))

print("预测结果为 %2.3f" % logreg.predict(np.array(array)))
