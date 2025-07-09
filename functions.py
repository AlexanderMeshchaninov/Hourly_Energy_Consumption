import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

def check_adfuller(df, title=''):
    """
    Проверяет стационарность временного ряда с помощью теста Дики-Фуллера (ADF).
    
    Параметры:
    - df: DataFrame с временным рядом
    - title: имя колонки с рядом

    Выводит ADF-статистику, p-value и критические значения.
    Сообщает, стационарен ли ряд (если p-value < 0.05).
    """
    adf_result = adfuller(df[title])

    print('ADF statistic:', adf_result[0])
    print('p-value:', adf_result[1])
    for key, value in adf_result[4].items():
        print(f'Критическое значение {key}: {value}')

    if adf_result[1] < 0.05:
        print("Ряд стационарен (по ADF)")
    else:
        print("Ряд НЕ стационарен (по ADF)")

def check_acf_pacf(df, lags_acf=10, lags_pacf=5, title=''):
    """
    Строит графики автокорреляции (ACF) и частичной автокорреляции (PACF).

    Параметры:
    - df: DataFrame с временным рядом
    - lags_acf: количество лагов для ACF
    - lags_pacf: количество лагов для PACF
    - title: имя колонки с рядом
    """
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plot_acf(df[title], lags=lags_acf, ax=plt.gca(), title='ACF — автокорреляция')

    plt.subplot(1, 2, 2)
    plot_pacf(df[title], lags=lags_pacf, ax=plt.gca(), method='ywm', title='PACF — частичная автокорреляция')

    plt.tight_layout()
    plt.show()
    
def train_test_split(df, test_size):
    """
    Разделяет временной ряд на обучающую и тестовую выборки.

    Параметры:
    - df: исходный DataFrame
    - test_size: размер тестовой выборки (в часах, днях и т.д.)

    Возвращает:
    - train: обучающая часть
    - test: тестовая часть
    """
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    return train, test

def evaluate(true, pred):
    """
    Вычисляет метрики качества прогноза: MAE, RMSE и MAPE.

    Параметры:
    - true: истинные значения
    - pred: предсказанные значения

    Возвращает:
    - mae: средняя абсолютная ошибка
    - rmse: корень из среднеквадратичной ошибки
    - mape: средняя абсолютная процентная ошибка
    """
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = mean_absolute_percentage_error(true, pred)
    return mae, rmse, mape

def plot_forecast(train, test, forecast, title):
    """
    Визуализирует прогноз модели на фоне обучающей и тестовой выборки.

    Параметры:
    - train: обучающая часть временного ряда
    - test: тестовая часть
    - forecast: прогноз модели
    - title: заголовок графика
    """
    plt.figure(figsize=(16, 5))
    plt.plot(train[-100:], label='Train')  # последние 100 точек для визуального контекста
    plt.plot(test, label='Test', color='orange')
    plt.plot(forecast, label='Forecast', color='green')
    plt.title(title)
    plt.legend(True)
    plt.grid(True)
    plt.show()

def run_sarimax(train, test, S, order=(1, 0, 1), seasonal_order=(1, 0, 1, 24), method='lbfgs', maxiter=200):
    """
    Обучает модель SARIMAX и строит прогноз на S шагов вперёд.

    Параметры:
    - train: обучающий набор данных
    - test: тестовый набор
    - S: горизонт прогноза (количество шагов вперёд)
    - order: параметры ARIMA (p, d, q)
    - seasonal_order: параметры сезонности (P, D, Q, s)
    - method: метод оптимизации (например, 'lbfgs')
    - maxiter: максимальное количество итераций

    Возвращает:
    - имя модели, горизонт, прогноз, MAE, RMSE, MAPE
    """
    model = SARIMAX(train['PJM_Load_MW'],
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    result = model.fit(disp=False, method=method, maxiter=maxiter)
    forecast = result.get_forecast(steps=S).predicted_mean
    mae, rmse, mape = evaluate(test['PJM_Load_MW'], forecast)
    return 'SARIMAX', S, forecast, mae, rmse, mape

def run_prophet(train, test, S, prophet_params=None):
    """
    Обучает модель Prophet и строит прогноз на S шагов вперёд.

    Параметры:
    - train: обучающий набор данных
    - test: тестовый набор
    - S: горизонт прогноза
    - prophet_params: словарь с параметрами Prophet (growth, seasonality и т.д.)

    Возвращает:
    - имя модели, горизонт, прогноз, MAE, RMSE, MAPE
    """
    prophet_train = train.reset_index().rename(columns={'Datetime': 'ds', 'PJM_Load_MW': 'y'})
    model = Prophet(**(prophet_params or {}))
    model.fit(prophet_train)

    future = pd.date_range(start=test.index[0], periods=S, freq='h')
    forecast_df = model.predict(pd.DataFrame({'ds': future}))
    forecast = forecast_df.set_index('ds')['yhat']

    mae, rmse, mape = evaluate(test['PJM_Load_MW'], forecast)
    return 'Prophet', S, forecast, mae, rmse, mape