import numpy as np
from sklearn.linear_model import LinearRegression

# データの準備
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # 独立変数
Y = np.array([[2, 3], [3, 5], [5, 7], [7, 9]])  # 従属変数

# モデルの作成と学習
model = LinearRegression()
model.fit(X, Y)

# 予測
Y_pred = model.predict(X)
print(Y_pred)
