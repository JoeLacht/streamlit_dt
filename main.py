import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from prophet import Prophet

st.header("Прогноз продаж по неделям")

uploaded_file = st.file_uploader("Загрузите CSV с данными о продажах", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Предварительный просмотр данных")
    st.dataframe(df.head())

try:
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    data = df.groupby('Order Date')['Sales'].sum()
    data = data.to_frame()
    data = data.resample('W').sum()

    data_prophet = data['Sales'].reset_index().rename(columns={'Order Date': 'ds', 'Sales': 'y'})

    train_size = int(len(data_prophet) * 0.8)
    data_train = data_prophet.iloc[:train_size]
    data_test = data_prophet.iloc[train_size:]

    model = Prophet(changepoint_prior_scale=0.001005839566510604, seasonality_prior_scale=1.5257650530224547, seasonality_mode='additive')
    model.fit(data_train)

    seasonality_period = 52
    number_of_future_predicted_points = 3 * seasonality_period # Предскажем на три периода в тесте + пару периодов наперед

    future = model.make_future_dataframe(periods=number_of_future_predicted_points, freq='W')
    forecast = model.predict(future)

    forecast_train = forecast[:-number_of_future_predicted_points] # Трейновый период
    forecast_test = forecast[-number_of_future_predicted_points: -number_of_future_predicted_points + len(data_test)] # Тестовый
    forecast_future = forecast[-number_of_future_predicted_points + len(data_test):] # Будущий период


    prophet_mae_train = np.round(mean_absolute_error(data_train['y'], forecast_train['yhat']), 1)
    prophet_mae_test = np.round(mean_absolute_error(data_test['y'], forecast_test['yhat']), 1)

    plt.figure(figsize=(20, 10))
    plt.plot(data['Sales'], label='true_data', marker='o')

    plt.plot(forecast_train['ds'], forecast_train['yhat'], marker='v', linestyle=':', label=f'forecast_train, mae={prophet_mae_train}')
    plt.plot(forecast_test['ds'], forecast_test['yhat'], marker='v', linestyle=':', label=f'forecast_test = mae={prophet_mae_test}')
    plt.plot(forecast_future['ds'], forecast_future['yhat'], marker='v', linestyle=':', label='forecast_future', color='b')

    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.15)
    plt.xticks(rotation=45)

    plt.title(f'Prophet, mae = {prophet_mae_test}')
    plt.legend()
    st.pyplot(plt)

    # Собираем только нужные колонки
    future_df = forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    future_df = forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
        columns={
            'ds': 'Weekly Date',
            'yhat': 'Predicted Sales',
            'yhat_lower': 'Predicted Sales (Lower Bound)',
            'yhat_upper': 'Predicted Sales (Upper Bound)'
        }
    )

    # Делаем кнопочку для скачивания
    csv = future_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Скачать прогноз (CSV)",
        data=csv,
        file_name="future_forecast.csv",
        mime="text/csv"
    )
    
except Exception as e:
    print('Ожидаются колонки "Order Date" и "Sales"')

