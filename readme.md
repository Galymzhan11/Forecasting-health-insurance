# Предсказание стоимости медицинской страховки

## Описание проекта

Этот проект посвящён построению моделей машинного обучения для предсказания расходов на медицинскую страховку на основе характеристик клиента (возраст, пол, ИМТ, дети, статус курения и регион). В процессе был проведён подробный анализ данных, инженерия признаков и сравнение моделей.

##  Цели

- Построить регрессионную модель для предсказания `charges`.
- Выявить наиболее важные признаки.
- Сравнить модели: LinearRegression, RandomForest, GradientBoosting, XGBoost.
- Проанализировать поведение модели и метрики качества.

##  Структура проекта

```
├── data/               # Данные (при необходимости)
├── notebooks/          # Jupyter ноутбуки с шагами
├── models/             # Сохранённые модели
├── visuals/            # Графики и изображения
├── src/                # Скрипты (при необходимости)
├── README.md           # Этот файл
├── requirements.txt    # Зависимости проекта
```

## Используемые признаки

Исходные признаки:

- `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`

Инженерия признаков:

- `bmi_age`, `bmi_age_smoker`, `binary_smoker`
- `bmi_group`, `age_group`

## Используемые библиотеки

- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost

## Результаты моделей

### 5‑fold CV (real-space)

| Model            | RMSE (CV) |
| ---------------- | --------- |
| LinearRegression | \~5698    |
| RandomForest     | \~4817    |
| GradientBoosting | \~4706    |
| XGBoost          | \~4982    |

### Hold‑out Metrics

| Model            | RMSE   | MAE    | R²     |
| ---------------- | ------ | ------ | ------ |
| LinearRegression | 5556.1 | 3958.2 | 0.8320 |
| RandomForest     | 4592.9 | 2554.3 | 0.8852 |
| GradientBoosting | 4428.2 | 2606.5 | 0.8933 |
| XGBoost          | 4725.5 | 2780.0 | 0.8785 |

**Лучшая модель:** `GradientBoosting` с RMSE = 4428.2 и R² = 0.8933

## Визуализация и интерпретация

- Построены графики распределений, корреляций и scatter-графики предсказаний.
- Анализ Feature Importance выявил наиболее значимые признаки: `bmi_age_smoker`, `age_squared`, `bmi_smoker`, `bmi_age`, `bmi`.

##  Запуск проекта

```
git clone https://github.com/Galymzhan11/Forecasting-health-insurance.git
cd Forecasting-health-insurance
pip install -r requirements.txt
jupyter notebook
```

## Загрузка модели

```python
import joblib
model = joblib.load("best_model.pkl")
y_pred = model.predict(X_new)
```

## Заключение

Проект успешно решает задачу регрессии и может быть использован в оценке медицинской страховки. Была достигнута высокая точность с помощью Gradient Boosting и тщательно проработанной фичер-инженерии.

---

Автор: [@Galymzhan Beketay](https://github.com/Galymzhan11)

