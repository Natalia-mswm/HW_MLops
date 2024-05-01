# Импорт необходимых библиотек
import numpy as np # для работы с числовыми данными
import pandas as pd # pandas для работы с данными
import os # Для работы с файловой системой


# Сгенерируем несколько наборов данных, описывающих количество пользователей в течение x дней на сайте А

# Генерация данных без аномалий и шумов
def generate_normal_data(base, x):
    ##n_data = np.array(base, y, x)
    return np.array([int(base) + int(np.random.randint(25)) for i in np.random.rand(x)])


# Генерация данных с аномалиями
def generate_anomaly_data(base, x):
    anomaly_data = generate_normal_data(base, x)
    
    # Создаем генератор случайных чисел
    rng = np.random.default_rng()
    
    # Выбираем 10 позиций из массива для аномалий без замены индекса 
    anomaly_positions = rng.choice(len(anomaly_data), size=10, replace=False) 

    # Изменяет значения в выбранных позициях на случайное
    for position in anomaly_positions:
        anomaly_data[position] += rng.uniform(0, 10000000)
    
    return anomaly_data


# Генерация данных с шумами
def generate_noise_data(base, x):
    noise_data = generate_normal_data(base, x)
    noise = np.random.randint(1, 100, len(noise_data))
    noise_data = noise_data + noise
    return noise_data


# Генерация данных и с шумами и с аномалиями
def generate_anomaly_and_noise_data(base, x):
    anomaly_data = generate_normal_data(base, x)
    rng = np.random.default_rng()
    anomaly_positions = rng.choice(len(anomaly_data), size=10, replace=False) 
    
    for position in anomaly_positions:
        anomaly_data[position] += rng.uniform(0, 10000000)
    
    noise = np.random.randint(1, 100, len(anomaly_data))
    final_data = anomaly_data + noise
    return final_data


# Сохраняет данные в CSV файл
def save_data(data, filename, directory): 
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f"{filename}.csv")
    data.to_csv(filepath, index=False)
    return filepath


def create_pd(data):
    df = pd.DataFrame()
    df['N'] = data.tolist()
    return df


# Создаем 3 набора данных 
days = 364
base_value = 20
year_2021 = generate_normal_data(base_value, days)
year_2022 = generate_anomaly_data(base_value, days)
year_2023 = generate_noise_data(base_value, days)
year_2024 = generate_anomaly_and_noise_data(base_value, days)
year_2025 = generate_anomaly_and_noise_data(base_value, days)


# Cоздание датафреймов
df2021 = create_pd(year_2021)
df2022 = create_pd(year_2022)
df2023 = create_pd(year_2023)
df2024 = create_pd(year_2024)
target = create_pd(year_2025)


# Сохранение данных
save_data(df2021, '2021', 'train')
save_data(df2024, '2024', 'train')
save_data(df2022, '2022', 'test')
save_data(df2023, '2023', 'test')
save_data(target, 'target_df', './')
