import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import requests_cache
from datetime import datetime
import openmeteo_requests
from retry_requests import retry
import numpy as np
import json
import os
from dotenv import load_dotenv

from helper_functions_qn2 import get_weather_data, process_weather_data, prepare_data_for_sagemaker, get_sagemaker_predictions, prepare_final_df, plot_heatmap, plot_timeseries

# Load environment variables from the .env file
load_dotenv() 

# Open-Meteo API to retrieve weather forecast information
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def get_weather_forecast():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 51.5074,
        "longitude": -0.1278,
        "hourly": ["temperature_2m", "wind_speed_10m", "relative_humidity_2m"],  # Adjusted for hourly data
        "timezone": "Europe/London"
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert("Europe/London"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True).tz_convert("Europe/London"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(2).ValuesAsNumpy()
    }

    return pd.DataFrame(data=hourly_data)

def get_season(date):
    month, day = date.month, date.day
    if (month == 12 and day >= 21) or (month <= 3 and day <= 19):
        return 0 # Winter
    elif (month == 3 and day >= 20) or (month <= 6 and day <= 20):
        return 1 # Spring
    elif (month == 6 and day >= 21) or (month <= 9 and day <= 21):
        return 2 # Summer
    else:
        return 3 # Autumn

##### STREAMLIT #####
# app code
st.title("Energy Analysis & Visualisation Dashbord")

# 2 tabs - Energy Forecast and Dunkelflaute Prediction
forecast_tab, dunkelflaute_tab = st.tabs(["📈 Energy Forecast", "🌑 Dunkelflaute Prediction"])

with forecast_tab:
    st.image("https://www.teriin.org/sites/default/files/2018-01/theme-banner_0.jpg", use_container_width=True)
    st.subheader("Weather Forecast")
    st.write("Current weather data and forecast will be used to predict energy consumption for the next 7 days.")

    # Get the weather data for the next 7 days
    weather_forecast_df = get_weather_forecast()
    weather_forecast_df["is_weekend"] = weather_forecast_df["date"].dt.dayofweek.isin([5, 6]).astype(int)
    weather_forecast_df["season"] = weather_forecast_df["date"].apply(get_season)

    if weather_forecast_df is not None:
        # plot weather forecast data
        fig = go.Figure()

        # temperature
        fig.add_trace(go.Scatter(
            x=weather_forecast_df["date"],
            y=weather_forecast_df["temperature_2m"],
            mode='lines',
            name='Temperature (°C)',
            line=dict(color='red')
        ))

        # humidity
        fig.add_trace(go.Scatter(
            x=weather_forecast_df["date"],
            y=weather_forecast_df["relative_humidity_2m"],
            mode='lines',
            name='Relative Humidity (%)',
            line=dict(color='blue')
        ))

        # wind speed 
        fig.add_trace(go.Scatter(
            x=weather_forecast_df["date"],
            y=weather_forecast_df["wind_speed_10m"],
            mode='lines',
            name='Wind Speed (m/s)',
            line=dict(color='green')
        ))

        fig.update_layout(
            title="Weather Forecast for the Next 7 Days",
            xaxis_title="Date",
            yaxis_title="Values",
            hovermode="x",
            legend_title="Weather Variables"
        )
        
        st.plotly_chart(fig, theme="streamlit")
        st.divider()
        
        ###### CLUSTERING ######
        st.subheader("Clustering Profile")
        st.write("Input your household's energy consumption data to determine which cluster you belong to.")
        
        # Labels for the 18 features
        feature_labels = [
            "Early Morning Peak (kWh)", "Early Morning Average (kWh)", 
            "Late Morning Peak (kWh)", "Late Morning Average (kWh)", 
            "Early Afternoon Peak (kWh)", "Early Afternoon Average (kWh)", 
            "Late Afternoon Peak (kWh)", "Late Afternoon Average (kWh)", 
            "Evening Peak (kWh)", "Evening Average (kWh)", 
            "Night Peak (kWh)", "Night Average (kWh)", 
            "Standard Deviation of Energy (kWh)", "Minimum Energy (kWh)", 
            "Maximum Energy (kWh)", "Q1 Energy (kWh)", 
            "Median Energy (kWh)", "Q3 Energy (kWh)"
        ]

        # Input fields for user data
        input_features = []
        for label in feature_labels:
            input_features.append(st.number_input(label, min_value=0.0, step=0.1))

        if st.button("Determine Cluster"):
            API_URL = "https://m1auibsxn8.execute-api.us-east-1.amazonaws.com/prod/testqn1clustering"

            if len(input_features) != 18:
                st.error("Input must contain exactly 18 numeric values.")
                st.stop()

            if not all(isinstance(f, (int, float)) for f in input_features):
                st.error("All input features must be numeric values.")
                st.stop()

            # request payload
            payload = {
                "features": input_features  
            }

            try:
                # Serialize the payload to JSON
                payload_json = json.dumps(payload)

                # Call the clustering API
                response = requests.post(API_URL, data=payload_json, headers={"Content-Type": "application/json"})

                if response.status_code == 200:
                    # Parse the response
                    result = response.json()
                    cluster_id = result.get("cluster", None)

                    if cluster_id is not None:
                        st.success(f"Your household belongs to Cluster {cluster_id}.")
                    else:
                        st.error("Failed to determine cluster. Please try again.")
                else:
                    st.error(f"Failed to call clustering API. Status code: {response.status_code}, Error: {response.text}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        
        st.divider()
        
        ###### FORECASTING ######
        st.subheader("Energy Foreacast")
        if st.button("Generate Energy Forecast for All Clusters"):
            
            for cluster_id in range(0, 10):
                st.write(f"### Energy Forecast for Cluster {cluster_id}")

                # test data for Lambda
                test_data = []
                for i in range(len(weather_forecast_df)):
                    test_data.append({
                        "datetime": weather_forecast_df["date"].iloc[i].strftime("%Y-%m-%d %H:%M:%S"),  # Match Lambda's expected format
                        "temperature": float(weather_forecast_df["temperature_2m"].iloc[i]),
                        "humidity": float(weather_forecast_df["relative_humidity_2m"].iloc[i]),
                        "windSpeed": float(weather_forecast_df["wind_speed_10m"].iloc[i]),
                        "is_weekend": int(weather_forecast_df["is_weekend"].iloc[i]),
                        "season": int(weather_forecast_df["season"].iloc[i])
                    })

                API_URL = "https://m1auibsxn8.execute-api.us-east-1.amazonaws.com/prod/call-sagemaker-endpoint"

                # request payload
                request_payload = {
                    "periods": 1,
                    "test_data": test_data,
                    "cluster": str(cluster_id)  # Pass the current cluster ID
                }

                # Call Lambda via API Gateway
                response = requests.post(API_URL, json=request_payload)

                if response.status_code == 200:
                    # Parse the forecasted values from the response
                    data = response.json()
                    forecast_data = data.get("forecast", {}).get("predictions", [])

                    # Extract predictions and ensure alignment with weather_forecast_df
                    if len(forecast_data) != len(weather_forecast_df):
                        st.error(f"Mismatch in forecast data length for Cluster {cluster_id}. Please try again later.")
                    elif len(forecast_data) == 0:
                        st.error(f"No forecast data received for Cluster {cluster_id}. Please try again later.")
                    else:
                        # Parse the forecast data into a DataFrame
                        forecast_df = pd.DataFrame(forecast_data)
                        forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"])  # Ensure datetime format

                        # Remove timezone from forecast_df's datetime column
                        forecast_df["datetime"] = forecast_df["datetime"].dt.tz_localize(None)

                        # Remove timezone from weather_forecast_df's date column (if needed)
                        weather_forecast_df["date"] = weather_forecast_df["date"].dt.tz_localize(None)

                        # Merge with weather_forecast_df to ensure alignment
                        forecast_df = forecast_df.merge(
                            weather_forecast_df[["date"]],
                            left_on="datetime",
                            right_on="date",
                            how="right"
                        )

                        # Plot hourly forecast with confidence interval
                        fig = go.Figure()

                        # confidence interval 
                        fig.add_trace(go.Scatter(
                            x=pd.concat([forecast_df["datetime"], forecast_df["datetime"][::-1]]),
                            y=pd.concat([forecast_df["upper"], forecast_df["lower"][::-1]]),
                            fill='toself',
                            fillcolor='rgba(100, 149, 237, 0.3)',  # Light blue fill
                            line=dict(color='rgba(255,255,255,0)'),  # No border line
                            hoverinfo="skip",
                            name='Confidence Interval'
                        ))

                        # hourly predictions line
                        fig.add_trace(go.Scatter(
                            x=forecast_df["datetime"],
                            y=forecast_df["predictions"],
                            mode='lines',
                            name='Hourly Forecast',
                            line=dict(color='blue', dash='solid')
                        ))

                        fig.update_layout(
                            title=f"7-Day Hourly Energy Consumption Forecast for Cluster {cluster_id}",
                            xaxis_title="Date and Time",
                            yaxis_title="Predicted Energy Consumption (kWh)",
                            hovermode="x"
                        )

                        st.plotly_chart(fig)

                        # Display the hourly forecasted values in a table
                        # st.write(f"### Hourly Forecasted Values for Cluster {cluster_id}")
                        # st.table(forecast_df[["datetime", "predictions", "lower", "upper"]])

                else:
                    st.error(f"Failed to get forecast for Cluster {cluster_id}. Error: {response.text}")
        
        # test w fake data first
    #     if st.button("Generate Energy Forecast"):
    #         # fake values
    #         fake_forecast_values = np.random.uniform(low=100, high=300, size=7)  # Daily total energy consumption
    #         lower_bound = fake_forecast_values - np.random.uniform(low=20, high=50, size=7)  # Lower confidence interval
    #         upper_bound = fake_forecast_values + np.random.uniform(low=20, high=50, size=7)  # Upper confidence interval

    #         daily_forecast_df = pd.DataFrame({
    #             "date": pd.date_range(start=weather_forecast_df["date"].iloc[0], periods=7, freq='D'),
    #             "predicted_energy": fake_forecast_values,
    #             "lower_bound": lower_bound,
    #             "upper_bound": upper_bound
    #         })

    #         fig = go.Figure()

    #         # predicted energy
    #         fig.add_trace(go.Scatter(
    #             x=daily_forecast_df["date"],
    #             y=daily_forecast_df["predicted_energy"],
    #             mode='lines',
    #             name='Predicted Energy',
    #             line=dict(color='blue', dash='solid')
    #         ))

    #         # confidence interval 
    #         fig.add_trace(go.Scatter(
    #             x=pd.concat([daily_forecast_df["date"], daily_forecast_df["date"][::-1]]),
    #             y=pd.concat([daily_forecast_df["upper_bound"], daily_forecast_df["lower_bound"][::-1]]),
    #             fill='toself',
    #             fillcolor='rgba(100, 149, 237, 0.3)',  # Darker light blue fill
    #             line=dict(color='rgba(255,255,255,0)'),
    #             hoverinfo="skip",
    #             name='Confidence Interval'
    #         ))


    #         fig.update_layout(
    #             title="7-Day Daily Energy Consumption Forecast",
    #             xaxis_title="Date",
    #             yaxis_title="Predicted Energy Consumption (kWh)",
    #             hovermode="x",
    #             legend_title="Forecast Components"
    #         )

    #         # Display the plot
    #         st.plotly_chart(fig)
    # else:
    #     st.error("Failed to retrieve weather data.")
        
    

with dunkelflaute_tab:
    st.image("https://www.concertoplus.eu/wp-content/uploads/2016/08/cre-page-banner.jpg", use_container_width=True)
    st.subheader("Dunkelflaute Prediction")
    st.write("Analyze the probability of Dunkelflaute for the next 7 days based on forecasted weather data.")
    
    # Set your latitude and longitude
    latitude = 51.5085
    longitude = -0.1257
    variables = ["wind_speed_120m", "cloud_cover"]

    # Access the environment variables
    AWS_ACCESS_KEY_ID_QN2 = os.getenv('AWS_ACCESS_KEY_ID_QN2')
    AWS_SECRET_ACCESS_KEY_QN2 = os.getenv('AWS_SECRET_ACCESS_KEY_QN2')
    
    print(f"AWS_ACCESS_KEY_ID_QN2: {AWS_ACCESS_KEY_ID_QN2}")
    
    # Set the SageMaker endpoint name
    endpoint_name_qn2 = 'Orchestrate-ETL-ML-Pipeline-qn2-xgb-endpoint'

    if st.button("Generate Dunkelflaute Prediction"):
        # Get weather data
        response = get_weather_data(latitude, longitude, variables)

        # Process weather data and prepare dataframe
        hourly_dataframe = process_weather_data(response)

        # Prepare data for SageMaker
        payload, hourly_dataframe = prepare_data_for_sagemaker(hourly_dataframe)

        # Get predictions from SageMaker
        output_df = get_sagemaker_predictions(payload, endpoint_name_qn2, AWS_ACCESS_KEY_ID_QN2, AWS_SECRET_ACCESS_KEY_QN2)

        # Prepare the final dataframe for visualization
        final_df = prepare_final_df(hourly_dataframe, output_df)

        # Plot the heatmap and display it in Streamlit
        heatmap_fig = plot_heatmap(final_df)
        st.plotly_chart(heatmap_fig)

        # Plot the time-series forecast and display it in Streamlit
        timeseries_fig = plot_timeseries(final_df)
        st.plotly_chart(timeseries_fig)
        
        # # fetch weather forecast data for the next 7 days
        # weather_forecast_df = get_weather_forecast()

        # # test w fake data (heatmap, time-series, table form)
        # if weather_forecast_df is not None:
        #     hourly_probabilities = np.random.uniform(low=0, high=1, size=len(weather_forecast_df))

        #     # Simulate daily aggregated Dunkelflaute classification (Yes/No)
        #     daily_aggregated = ["Yes" if np.random.uniform(0, 1) > 0.7 else "No" for _ in range(7)]

        #     # heatmap 
        #     st.write("### Hourly Dunkelflaute Probability Heatmap")
        #     heatmap_data = pd.DataFrame({
        #         "datetime": weather_forecast_df["date"],
        #         "probability": hourly_probabilities
        #     })
        #     heatmap_data["hour"] = heatmap_data["datetime"].dt.hour
        #     heatmap_data["day"] = heatmap_data["datetime"].dt.date

        #     heatmap_pivot = heatmap_data.pivot(index="hour", columns="day", values="probability")
        #     fig = go.Figure(data=go.Heatmap(
        #         z=heatmap_pivot.values,
        #         x=heatmap_pivot.columns,
        #         y=heatmap_pivot.index,
        #         colorscale="Viridis",
        #         colorbar=dict(title="Probability")
        #     ))
        #     fig.update_layout(
        #         title="Hourly Dunkelflaute Probability Heatmap",
        #         xaxis_title="Date",
        #         yaxis_title="Hour of Day"
        #     )
        #     st.plotly_chart(fig)
            
        #     # time-series chart for Dunkelflaute Probability
        #     st.write("### Dunkelflaute Probability Time Series")
        #     time_series_data = pd.DataFrame({
        #         "date": weather_forecast_df["date"].dt.date,
        #         "hour_of_day": weather_forecast_df["date"].dt.hour,
        #         "dunkelflaute_prob": hourly_probabilities
        #     })
        #     time_series_data["datetime"] = pd.to_datetime(
        #         time_series_data["date"].astype(str) + " " + time_series_data["hour_of_day"].astype(str) + ":00"
        #     )

        #     # plot dunkelflaute
        #     fig = go.Figure()
        #     fig.add_trace(go.Scatter(
        #         x=time_series_data["datetime"],
        #         y=time_series_data["dunkelflaute_prob"],
        #         mode='lines',
        #         name='Dunkelflaute Probability',
        #         line=dict(color='blue', dash='solid'),
        #         marker=dict(size=5)
        #     ))
        #     fig.update_layout(
        #         title="Dunkelflaute Probability Time Series (Predicted Probabilities)",
        #         xaxis_title="Date and Hour",
        #         yaxis_title="Dunkelflaute Probability",
        #         xaxis=dict(tickformat="%Y-%m-%d %H:%M", tickangle=45),
        #         hovermode="x"
        #     )
        #     st.plotly_chart(fig)

        #     # display the 7-day aggregated forecast (zoomed-out view)
        #     st.write("### 7-Day Aggregated Dunkelflaute Forecast")
        #     daily_forecast_df = pd.DataFrame({
        #         "date": pd.date_range(start=weather_forecast_df["date"].iloc[0], periods=7, freq='D'),
        #         "dunkelflaute": daily_aggregated
        #     })
        #     st.table(daily_forecast_df)

        # else:
        #     st.error("Failed to retrieve weather data.")
