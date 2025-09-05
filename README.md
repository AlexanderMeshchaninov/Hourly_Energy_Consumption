# Electricity Load Forecasting
![](Images/image_for_readme_project.png)

## Описание проекта

Цель проекта — спрогнозировать почасовое потребление электроэнергии в одном из регионов США на основе исторических данных.

Задача относится к классу регрессионных задач на временных рядах. Реализовано несколько подходов к прогнозированию, включая классические статистические модели (SARIMA), Prophet от Meta, а также нейронную сеть.

## Источник данных

Данные для проекта взяты с Kaggle:

🔗 [Hourly Energy Consumption — Rob Mulla (Kaggle)](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)

Датасет содержит почасовую нагрузку на энергосистему PJM по регионам США.

> Источник данных: Rob Mulla, Kaggle Datasets

## Установка зависимостей

Перед запуском установите необходимые библиотеки:

```bash
    !pip install --upgrade pip
    !pip install prophet
```

## Проект ориентирован на:

- Выявление сезонности и трендов
- Сравнение моделей по точности
- Прогнозирование нагрузки на ближайшие 2 дня с помощью статических моделей
- Прогнозирование нагрузки на ближайшие 2 недели с помощью нейронной сети LSTM
- Практику пайплайна анализа временных рядов

### Расшифровка регионов PJM

| Файл                 | Регион     | Описание |
|----------------------|------------|----------|
| `AEP_hourly.csv`     | AEP        | American Electric Power — крупный поставщик электроэнергии в США |
| `COMED_hourly.csv`   | COMED      | Commonwealth Edison — северо-восток Иллинойса, включая Чикаго |
| `DAYTON_hourly.csv`  | DAYTON     | Dayton Power & Light — обслуживает регион Дейтона, Огайо |
| `DEOK_hourly.csv`    | DEOK       | Duke Energy Ohio and Kentucky |
| `DOM_hourly.csv`     | DOM        | Dominion Energy — Вирджиния и Северная Каролина |
| `DUQ_hourly.csv`     | DUQ        | Duquesne Light Company — Питтсбург, Пенсильвания |
| `EKPC_hourly.csv`    | EKPC       | East Kentucky Power Cooperative — кооператив в Кентукки |
| `FE_hourly.csv`      | FE         | FirstEnergy — многорегиональная компания |
| `NI_hourly.csv`      | NI         | Northern Illinois — подзона в составе COMED |
| `PJM_Load_hourly.csv`| PJM        | Совокупная нагрузка PJM Interconnection |
| `PJME_hourly.csv`    | PJME       | PJM East — агрегированные данные восточного региона |
| `pjm_hourly_est.csv` | PJM (est)  | Оценочная версия нагрузки PJM (estimated) |
| `PJMW_hourly.csv`    | PJMW       | PJM West — агрегированные данные западного региона |

Примечание: В обучении используются совокупные данные нагрузки (PJM_Load_hourly.csv)

## Используемые технологии

- Python 3.10+
- Pandas, Numpy, Matplotlib, Seaborn
- Scikit-learn
- Statsmodels (`SARIMAX`)
- `Prophet` (от Meta)
- TensorFlow/Keras (для LSTM)
- Optuna (подбор гиперпараметров)
- TimeSeriesSplit (кросс-валидация)

## Подходы к моделированию

Проект включает реализацию и сравнение нескольких моделей:

| Модель       | Подход               | Назначение                       |
|--------------|----------------------|----------------------------------|
| **SARIMAX**  | Статистическая       | Краткосрочное прогнозирование    |
| **Prophet**  | Полустатистическая   | Прогноз с учётом сезонности      |
| **LSTM**     | Нейросеть            | Средне- и долгосрочный прогноз   |

Каждая стат. модель тестируется на горизонтах 24, 48 часов. 
Нейронная сеть на горизонтах 168, 336 часов.
Сравнение производится по метрикам: **MAE**, **RMSE**, **MAPE**.

## Сохраненные модели

🔗 [models](https://disk.yandex.ru/d/aSVvymGqdLh-0Q)

## Дополнительные ноутбуки

Основной ноутбук с реализацией LSTM-модели доступен на Kaggle:

🔗 [LSTM Model on Kaggle](https://www.kaggle.com/code/alexanderf85/lstm-model)

В этом ноутбуке реализованы:
- масштабирование данных и подготовка окон
- подбор гиперпараметров с помощью `Optuna`
- кросс-валидация `TimeSeriesSplit`
- финальное обучение модели `LSTM`
- прогноз на 336 часов и визуализация
- сохранение обученной модели (`.h5`)

## TODO

- [x] Построение и сравнение SARIMAX / Prophet
- [x] Обучение LSTM с Optuna и кросс-валидацией
- [x] Прогноз и сохранение моделей
- [ ] Добавить внешние признаки (погода, день недели)
- [ ] Реализация LSTM с attention / Transformer