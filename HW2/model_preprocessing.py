# Импорт необходимых библиотек
import pandas as pd # pandas для работы с данными
from sklearn.preprocessing import StandardScaler # для стандартизации признаков
import glob # Для поиска файлов по шаблону в указанных директориях
import os # Для работы с файловой системой


# Функция найдет в папках train и test все файлы по шаблону *.csv и вернет их список
def get_data_path(train_path='./train/', test_path='./test/'):
    train_files = glob.glob(train_path + "*.csv")
    test_files = glob.glob(test_path + "*.csv")
    return train_files, test_files


# Функция создаст 2 датафрейма (test, train) из найденных csv файлов 
def create_df():
    train_files, test_files = get_data_path()
    
    # требуется, чтобы количество тренировочных файлов (train_files) было равно количеству тестовых файлов (test_files),
    # чтобы обеспечить сопоставимость данных для обучения и тестирования модели машинного обучения.
    if len(train_files) == len(test_files):
        if len(train_files) > 1:
            train_df = pd.read_csv(train_files[0])
            test_df = pd.read_csv(test_files[0])
            
            for i in range(1, len(train_files)):
                train_df = pd.concat([train_df, pd.read_csv(train_files[i])], axis=1)
                test_df = pd.concat([test_df, pd.read_csv(test_files[i])], axis=1)
        
        elif len(train_files) == 1:
            return pd.read_csv(train_files[0]), pd.read_csv(test_files[0])
       
    return train_df, test_df


# Завершаем создание датафреймов
def add_target(train_df, test_df):
    target = pd.read_csv('target_df.csv')
    
    train_df = pd.concat([train_df, target], axis=1)
    train_df.columns = ['Year_1','Year_2','Target']

    test_df = pd.concat([test_df, target], axis=1)
    test_df.columns = ['Year_1','Year_2','Target']
    return train_df, test_df


# Создаем тестовый и тренировочный df
train_df, test_df = create_df()
# Завершаем создание, добавляя колонку с целевой переменной
train_df, test_df = add_target(train_df, test_df)


# Разделяем признаки и целевую переменную
X_train = train_df.drop('Target', axis=1)
y_train = train_df['Target']
X_test = test_df.drop('Target', axis=1)
y_test = test_df['Target']


# Стандартизация признаков
scaler = StandardScaler()
scaled_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
scaled_X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# объединяем в единые датафреймы приведенные признаки и целевые значения
scaled_train_df = pd.concat([scaled_X_train, y_train], names=['10', '20', '30', '40', '50'], axis=1)
scaler_test_df = pd.concat([scaled_X_test, y_test], names=['10', '20', '30', '40', '50'], axis=1)


# записываем полученные датафреймы в единые файлы
train_path = os.path.join("./train", "result_train.csv")
test_path = os.path.join("./test", "result_test.csv")

scaled_train_df.to_csv(train_path, index=False)
scaler_test_df.to_csv(test_path, index=False)
