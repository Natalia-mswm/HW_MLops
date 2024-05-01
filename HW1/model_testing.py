# Импорт необходимых библиотек
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Метрики
import pandas as pd # pandas для работы с данными
import pickle # pickle для сохранения обученной модели в файл
import numpy as np # Для работы с числовыми данными


# Загружаем модель
model = pickle.load(open('model.pkl', 'rb'))
df_test = pd.read_csv('./test/result_test.csv')


# Выделяем в тестовом датафрейме признаки и целевую переменную
X_test = df_test.drop("Target", axis=1)
y_test = df_test["Target"]


# Предсказание на данных из пвпки test
y_predict = pd.DataFrame(model.predict(X_test))


# Оценка
# Расчет метрик
mse = mean_squared_error(y_test, y_predict) # Среднеквадратичная ошибка, показывает, насколько хорошо предсказаны значения, но не учитывает размер ошибки
mae = mean_absolute_error(y_test, y_predict) # Средняя абсолютная ошибка (более чувствительна к выбросам)
r2 = r2_score(y_test, y_predict) # коэффициент детерминации показывает, какая доля дисперсии зависимой переменной объясняется моделью
rmse = np.sqrt(mse) # кв корень из MSE, более интерпретируемый, поскольку он выражается в тех же единицах, что и таргет


# Вывод полученных метрик
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")
print(f"Root Mean Squared Error (RMSE): {rmse}")