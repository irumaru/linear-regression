import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
#from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math
import numpy as np

# データ読み込み
cal_housing = pd.read_csv('datasets/cal_housing/CaliforniaHousing/cal_housing.data', sep=',')
cal_housing.columns = ['longitude', 'latitude', 'housingMedianAge', 'totalRooms', 'totalBedrooms', 'population', 'households', 'medianIncome', 'medianHouseValue']

# 正規化

# cal_housing['longitude'] = (cal_housing['longitude'] - cal_housing['longitude'].min()) * 100
# cal_housing['latitude'] = (cal_housing['latitude'] - cal_housing['latitude'].min()) * 100

cal_housing = cal_housing[cal_housing['population'] / cal_housing['households'] <= 10]
cal_housing['population'] = cal_housing['population'] / cal_housing['households']

cal_housing = cal_housing[cal_housing['totalRooms'] / cal_housing['households'] <= 10]
cal_housing['totalRooms'] = cal_housing['totalRooms'] / cal_housing['households']

cal_housing = cal_housing[cal_housing['totalBedrooms'] / cal_housing['households'] <= 10]
cal_housing['totalBedrooms'] = cal_housing['totalBedrooms'] / cal_housing['households']

print(cal_housing.describe())
#exit()

# cal_housingはあなたのデータフレームと仮定します
# 相関行列を計算します
corr_matrix = cal_housing.corr().abs()

# 相関が0.8以上の列を取得します
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]

print(to_drop)

print(cal_housing)

# 相関が強い列を削除します
#cal_housing.drop(to_drop, axis=1, inplace=True)

#print(cal_housing.describe())

def train(X, y):
  # 訓練データとテストデータに分割
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

  # 学習
  model = LinearRegression()
  #model = KernelRidge(alpha=0.01)
  model.fit(X_train, y_train)

  print(model.coef_)

  # 評価
  mse = mean_squared_error(y_test, model.predict(X_test))
  r2 = r2_score(y_test, model.predict(X_test))
  print('MSE:', mse)
  print('R2:', r2)

  print(model.predict(X_test))
  print(y_test)

  #eva = pd.concat([y_test, prediction], axis=1)

  #print(eva)

  return model

# 抽出
#X = cal_housing[['housingMedianAge', 'totalRooms', 'totalBedrooms', 'population', 'households', 'medianIncome']]
X = cal_housing[['housingMedianAge', 'totalRooms', 'totalBedrooms', 'population', 'households', 'medianIncome']]

#longitude_model = train(X, cal_housing[['longitude', 'latitude']])
longitude_model = train(X, cal_housing['medianHouseValue'])
