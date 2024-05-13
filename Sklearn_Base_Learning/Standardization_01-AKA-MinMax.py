import sklearn.model_selection
from sklearn.preprocessing import MinMaxScaler
import numpy as np

X = np.arange(15).reshape(5, 3)
X_train, X_test = sklearn.model_selection.train_test_split(X, random_state=42)

scaler = MinMaxScaler()
Nor_X = scaler.fit_transform(X)
print(scaler.data_max_)
