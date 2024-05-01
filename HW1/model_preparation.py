import pandas as pd # pandas для работы с данными
import numpy as np # Для работы с числовыми данными
from sklearn.linear_model import LinearRegression # LinearRegression из sklearn.linear_model для создания модели линейной регрессии
import pickle # pickle для сохранения обученной модели в файл
from sklearn.utils import shuffle # Для перемешивания данны
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Метрики


# Загружаем данные в датафрейм
df = pd.read_csv('./train/result_train.csv')


# Перемешаем данные
df = shuffle(df, random_state=42)


# Разделяем датафрейм на признаки и целевую переменную
X_train = df.drop('Target', axis=1)
y_train = df['Target']


# Создаем экземпляр модели и обучаем ее
model = LinearRegression()
model.fit(X_train, y_train)


# Предсказание на данных из пвпки train
y_predict = model.predict(X_train)


# Расчет метрик
mse = mean_squared_error(y_train, y_predict) # Среднеквадратичная ошибка, показывает, насколько хорошо предсказаны значения, но не учитывает размер ошибки
mae = mean_absolute_error(y_train, y_predict) # Средняя абсолютная ошибка (более чувствительна к выбросам)
r2 = r2_score(y_train, y_predict) # коэффициент детерминации показывает, какая доля дисперсии зависимой переменной объясняется моделью
rmse = np.sqrt(mse) # кв корень из MSE, более интерпретируемый, поскольку он выражается в тех же единицах, что и таргет


# Вывод метрик
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Сохраняем обученнуюю модель в файл
pickle.dump(model, open('model.pkl', 'wb'))