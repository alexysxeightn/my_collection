"""
Консольное приложение для прогноза продолжительности жизни пациента и предсказания выживаемости
"""
import argparse
import pickle
from typing import Tuple

import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import Ridge

# Пути до предобученных моделей и тренировочных данных (они нужны для нормализации)
PATH_TO_REGR_MODEL = 'best_regr_model.pkl'
PATH_TO_CLF_MODEL = 'best_clf_model.pkl'
PATH_TO_TRAINING_DF = 'data.csv'

# Признаки, которые были отобраны для обоих моделей
COLUMNS_FOR_REGR = [
    'Gender', 'Medical ICU', 'Surgical ICU',
    'Cardiac Surgery Recovery Unit', 'Age', 'Height', 'MeanBUN', 'MeanFiO2',
    'MeanGCS', 'MeanHR', 'MeanK', 'MeanMg', 'MeanNa', 'MeanNIDiasABP',
    'MeanNIMAP', 'MeanNISysABP', 'MeanSysABP', 'MeanTemp', 'MeanUrine',
    'MeanWeight'
]
COLUMNS_FOR_CLF = [
    'Gender', 'Medical ICU', 'Coronary Care Unit',
    'Cardiac Surgery Recovery Unit', 'Age', 'MeanBUN', 'MeanCreatinine',
    'MeanFiO2', 'MeanGCS', 'MeanGlucose', 'MeanHR', 'MeanLactate', 'MeanNa',
    'MeanNIDiasABP', 'MeanNIMAP', 'MeanNISysABP', 'MeanPAO2', 'MeanTemp',
    'MeanUrine', 'MeanWeight'
]

# Фичи, которые не нуждаются в нормализации
NOT_NORMALIZE_FEATURES = [
    'RecordID', 'Gender', 'Medical ICU',
    'Surgical ICU', 'Coronary Care Unit',
    'Cardiac Surgery Recovery Unit'
]


def load_models() -> Tuple[Ridge, SVC]:
    """
    Функция для загрузки предобученных моделей

    Возвращает регрессионную и классификационную модель
    """
    with open(PATH_TO_REGR_MODEL, 'rb') as file:
        regr_model = pickle.load(file)

    with open(PATH_TO_CLF_MODEL, 'rb') as file:
        clf_model = pickle.load(file)

    return regr_model, clf_model


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Функция принимает на вход датафрейм и нормализует в нем необходимые фичи
    """
    train_df = pd.read_csv(PATH_TO_TRAINING_DF, sep=',').drop('Unnamed: 0', axis=1)
    train_df = train_df.drop(NOT_NORMALIZE_FEATURES, axis=1)
    df_normalize = df.drop(NOT_NORMALIZE_FEATURES, axis=1)
    df_normalize = (df_normalize - train_df.mean()) / train_df.std()
    data = df[NOT_NORMALIZE_FEATURES].join(df_normalize)
    return data


def get_prediction(path: str) -> pd.DataFrame:
    """
    Предсказывает таргеты по датасету, который находится по пути path

    Возвращает датафрейм с предсказаниями для каждого пациента
    """
    df = pd.read_csv(path, sep=',').drop('Unnamed: 0', axis=1)
    data = normalize_data(df)

    # Создаем отдельные датасеты для задачи классификации и регрессии
    data_regr = data[COLUMNS_FOR_REGR]
    data_clf = data[COLUMNS_FOR_CLF]

    # Загружаем модели
    regr_model, clf_model = load_models()

    # Инференс моделей
    days = regr_model.predict(data_regr)
    death = clf_model.predict(data_clf)
    death_proba = clf_model.predict_proba(data_clf)[:, 1]

    # Подсчитываем индикатор степени тяжести больного
    indicator = []
    # трешхолды в 0.1 и 0.2 подобраны из соображений логики
    for p in death_proba:
        if p < 0.1:
            indicator.append('зеленая')
        elif p < 0.2:
            indicator.append('желтая')
        else:
            indicator.append('красная')

    # Организуем результат в датафрейм
    result = pd.DataFrame()
    result['RecordID'] = df['RecordID']
    result['Продолжительность жизни (в днях)'] = days.astype('int64')
    result['Степень тяжести'] = indicator
    result['Наступит смерть'] = death
    result['Наступит смерть'].replace({0: 'нет', 1: 'да'}, inplace=True)
    return result


def main(path_to_data: str, output_type: str, path_to_result: str):
    """
    Основная функция программы

    Принимает на вход путь до данных, тип вывода результата и (для
    случая вывода в файл) путь до файла с результатом
    """
    # Создаем предсказания по данным
    prediction = get_prediction(path_to_data)

    if output_type == 'output':
        # В случае output выводим все на экран и выходим из программы
        print(prediction)
        return 0
    elif output_type == 'json':
        # В случае json возвращает результат в формате json-файла и выходит из программы
        return prediction.to_json(orient="split")
    elif output_type == 'file':
        if path_to_result is None:
            # Если выбран вывод в файл и не указан путь до файла, то
            # предупреждаем пользователя и выходим из программы
            print('Укажите путь, куда сохранить результат')
        else:
            # Если выбран вывод в файл и указан путь, то сохраняем
            # туда результат и выходим из программы
            prediction.to_csv(path_to_result, sep=',', encoding='utf-8')
            print('Успешно сохранено')
            return 0
    else:
        # Если указан некорректный тип вывода, то говорим об этом пользователю
        print('Некоректное значение параметра --output-type')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='API для формирования заключения по прогнозированию ' \
        'продолжительности жизни и классификации выживаемости пациента'
    )
    parser.add_argument(
        '--path-to-data',
        required=True,
        help='Путь до файла, содержащего данные о пациенте'
    )
    parser.add_argument(
        '--output-type',
        help="""
            output - выводит результат на экран; \n
            json - возвращает json; \n
            file - сохраняет результат в файл (необходимо указать путь в --path-to-result)
        """,
        default='output'
    )
    parser.add_argument(
        '--path-to-result',
        help='Путь, куда сохраняется результат работы моделей',
        default=None
    )
    args = parser.parse_args()
    main(args.path_to_data, args.output_type, args.path_to_result)