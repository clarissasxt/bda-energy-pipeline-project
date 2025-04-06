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
# from dotenv import load_dotenv

from helper_functions_qn2 import get_weather_data, process_weather_data, prepare_data_for_sagemaker, get_sagemaker_predictions, get_lambda_predictions, prepare_final_df, plot_heatmap, plot_timeseries

# Load environment variables from the .env file
# load_dotenv() 

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
st.title("UK Energy Analysis & Visualisation Dashboard")

# 2 tabs - Energy Forecast and Dunkelflaute Prediction
forecast_tab, dunkelflaute_tab = st.tabs(["📈 Energy Forecast", "🌑 Dunkelflaute Prediction"])

with forecast_tab:
    st.image("https://www.teriin.org/sites/default/files/2018-01/theme-banner_0.jpg", use_container_width=True)
    st.subheader("🌤️ Weather Forecast")
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
        # Simplified input fields for user data
        st.subheader("🏠 Determine Household Profile Cluster")
        st.write("Want to determine your household profile type? Provide a few key values, and the rest will be calculated automatically.")

        # Key user inputs
        total_energy = st.number_input("Total Energy Consumption (kWh) Per Day", min_value=0.0, step=0.1)
        peak_energy = st.number_input("Peak Energy Consumption (kWh) Per Day", min_value=0.0, step=0.1)
        average_energy = st.number_input("Average Energy Consumption (kWh) Per Day", min_value=0.0, step=0.1)

        # Button to determine cluster
        if st.button("Determine Cluster"):

            # Define weight factors for each period
            # Example: Higher energy usage in the evening and morning
            period_weights = np.array([0.15, 0.10, 0.20, 0.15, 0.25, 0.15])  # Weights for 6 periods
            period_weights = period_weights / period_weights.sum()  # Normalize to ensure they sum to 1

            # Calculate energy values for each period based on weights
            energy_values = total_energy * period_weights

            # Optionally, add random variability to simulate real-world fluctuations
            variability = np.random.uniform(-0.05, 0.05, size=len(energy_values))  # ±5% variability
            energy_values = energy_values * (1 + variability)

            # Calculate peak and average energy for each period
            peak_period_energy = peak_energy * period_weights
            avg_period_energy = average_energy * period_weights

            # Calculate standard deviation dynamically
            sd_energy = np.std(energy_values, ddof=0)  # Population standard deviation

            # Construct the input_features list
            input_features = [
                peak_period_energy[0], avg_period_energy[0],  # Early Morning
                peak_period_energy[1], avg_period_energy[1],  # Late Morning
                peak_period_energy[2], avg_period_energy[2],  # Early Afternoon
                peak_period_energy[3], avg_period_energy[3],  # Late Afternoon
                peak_period_energy[4], avg_period_energy[4],  # Evening
                peak_period_energy[5], avg_period_energy[5],  # Night
                sd_energy,  # Standard Deviation
                avg_period_energy.mean() * 0.8,  # Minimum Energy (e.g., 80% of average)
                avg_period_energy.mean() * 1.2,  # Maximum Energy (e.g., 120% of average)
                avg_period_energy.mean() * 0.25,  # Q1 Energy (e.g., 25% of average)
                avg_period_energy.mean(),  # Median Energy
                avg_period_energy.mean() * 0.75  # Q3 Energy (e.g., 75% of average)
            ]

            # Validate the input_features list
            if len(input_features) != 18:
                st.error("Failed to construct the input features. Please try again.")
                st.stop()

            # Prepare the payload
            payload = {
                "features": input_features  # Pass the list directly
            }

            try:
                # Serialize the payload to JSON
                payload_json = json.dumps(payload)

                # Call the clustering API
                API_URL = "https://m1auibsxn8.execute-api.us-east-1.amazonaws.com/prod/testqn1clustering"
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
        # Add a radio button for user selection
        st.subheader("⚡️ Energy Forecast")
        cluster_option = st.radio(
            "Select a cluster to display the forecast:",
            options=[0, 1, 2, 3, 4, 5, 6, "All"],
            format_func=lambda x: f"Cluster {x}" if x != "All" else "All Clusters"
        )

        # Generate forecast based on user selection
        if st.button("Generate Energy Forecast"):
            if cluster_option == "All":
                # Display forecasts for all clusters
                for cluster_id in range(7):
                    st.write(f"### Energy Forecast for Cluster {cluster_id}")

                    # Test data for Lambda
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

                    # Request payload
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

                            # Confidence interval
                            fig.add_trace(go.Scatter(
                                x=pd.concat([forecast_df["datetime"], forecast_df["datetime"][::-1]]),
                                y=pd.concat([forecast_df["upper"], forecast_df["lower"][::-1]]),
                                fill='toself',
                                fillcolor='rgba(100, 149, 237, 0.3)',  # Light blue fill
                                line=dict(color='rgba(255,255,255,0)'),  # No border line
                                hoverinfo="skip",
                                name='Confidence Interval'
                            ))

                            # Hourly predictions line
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

                    else:
                        st.error(f"Failed to get forecast for Cluster {cluster_id}. Error: {response.text}")

            else:
                # Display forecast for the selected cluster
                cluster_id = cluster_option
                st.write(f"### Energy Forecast for Cluster {cluster_id}")

                # Test data for Lambda
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

                # Request payload
                request_payload = {
                    "periods": 1,
                    "test_data": test_data,
                    "cluster": str(cluster_id)  # Pass the selected cluster ID
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

                        # Confidence interval
                        fig.add_trace(go.Scatter(
                            x=pd.concat([forecast_df["datetime"], forecast_df["datetime"][::-1]]),
                            y=pd.concat([forecast_df["upper"], forecast_df["lower"][::-1]]),
                            fill='toself',
                            fillcolor='rgba(100, 149, 237, 0.3)',  # Light blue fill
                            line=dict(color='rgba(255,255,255,0)'),  # No border line
                            hoverinfo="skip",
                            name='Confidence Interval'
                        ))

                        # Hourly predictions line
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

                else:
                    st.error(f"Failed to get forecast for Cluster {cluster_id}. Error: {response.text}")

with dunkelflaute_tab:
    st.image("https://www.concertoplus.eu/wp-content/uploads/2016/08/cre-page-banner.jpg", use_container_width=True)
    st.subheader("Dunkelflaute Prediction")
    st.write("Analyze the probability of Dunkelflaute for the next 7 days based on forecasted weather data.")
    
    # Set your latitude and longitude
    latitude = 51.5085
    longitude = -0.1257
    variables = ["wind_speed_120m", "cloud_cover"]

    # # Access the environment variables
    # AWS_ACCESS_KEY_ID_QN2 = os.getenv('AWS_ACCESS_KEY_ID_QN2')
    # AWS_SECRET_ACCESS_KEY_QN2 = os.getenv('AWS_SECRET_ACCESS_KEY_QN2')
    
    # print(f"AWS_ACCESS_KEY_ID_QN2: {AWS_ACCESS_KEY_ID_QN2}")
    
    # Set the SageMaker endpoint name
    endpoint_name_qn2 = 'Orchestrate-ETL-ML-Pipeline-qn2-xgb-endpoint'
    
    # Set lambda function name 
    # lambda_function_name_qn2 = 'call-qn2-sagemaker-endpoint'

    if st.button("Generate Dunkelflaute Prediction"):
        # Get weather data
        response = get_weather_data(latitude, longitude, variables)

        # Process weather data and prepare dataframe
        hourly_dataframe = process_weather_data(response)

        # Prepare data for SageMaker
        payload, hourly_dataframe = prepare_data_for_sagemaker(hourly_dataframe)

        # Get predictions from SageMaker
        # API Gateway URL (replace with your actual URL)
        api_url = "https://q85vghv4fa.execute-api.us-east-1.amazonaws.com/prod/qn2-invoke-sagemaker"

        # Call the API Gateway
        response = requests.post(
            api_url,
            json={"data": payload},
            headers={"Content-Type": "application/json"}
        )

        # Process the response
        if response.status_code == 200:
            try:
                result = response.json()
                if isinstance(result, str):
                    # Sometimes API Gateway returns the result as a JSON string
                    result = json.loads(result)
                    
                # Extract predictions based on your response structure
                if 'body' in result:
                    body = result['body']
                    if isinstance(body, str):
                        body = json.loads(body)
                    predictions = body['predictions']
                else:
                    predictions = result['predictions']
                    
                # Create a dataframe with the predictions
                output_df = pd.DataFrame(predictions, columns=['probability'])
                print(output_df.head())
            except Exception as e:
                print(f"Error processing response: {e}")
                print("Response content:", response.text)
        else:
            print(f"Error: HTTP Status {response.status_code}")
            print("Response content:", response.text)

        # Prepare the final dataframe for visualization
        final_df = prepare_final_df(hourly_dataframe, output_df)

        # Plot the heatmap and display it in Streamlit
        heatmap_fig = plot_heatmap(final_df)
        st.plotly_chart(heatmap_fig)

        # Plot the time-series forecast and display it in Streamlit
        timeseries_fig = plot_timeseries(final_df)
        st.plotly_chart(timeseries_fig)
