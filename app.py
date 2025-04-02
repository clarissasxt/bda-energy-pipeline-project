import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import requests_cache
from datetime import datetime
import openmeteo_requests
from retry_requests import retry
import numpy as np

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


# Function to determine the season based on the current date
def get_season(date):
    month, day = date.month, date.day
    if (month == 12 and day >= 21) or (month <= 3 and day <= 19):
        return "Winter"
    elif (month == 3 and day >= 20) or (month <= 6 and day <= 20):
        return "Spring"
    elif (month == 6 and day >= 21) or (month <= 9 and day <= 21):
        return "Summer"
    else:
        return "Fall"

# Streamlit app setup
st.title("Energy Analysis & Visualisation Dashbord")

# Create Tabs
forecast_tab, dunkelflaute_tab = st.tabs(["📈 Energy Forecast", "🌑 Dunkelflaute Prediction"])

with forecast_tab:
    st.image("https://www.teriin.org/sites/default/files/2018-01/theme-banner_0.jpg", use_container_width=True)
    st.subheader("Energy Forecast")
    st.write("Current weather data and forecast will be used to predict energy consumption for the next 7 days.")

    # Get the weather data for the next 7 days
    weather_forecast_df = get_weather_forecast()
    weather_forecast_df["is_weekend"] = weather_forecast_df["date"].dt.dayofweek.isin([5, 6]).astype(int)
    weather_forecast_df["season"] = weather_forecast_df["date"].apply(get_season)

    if weather_forecast_df is not None:
        # Plot the weather forecast data
        fig = go.Figure()

        # Add temperature line
        fig.add_trace(go.Scatter(
            x=weather_forecast_df["date"],
            y=weather_forecast_df["temperature_2m"],
            mode='lines',
            name='Temperature (°C)',
            line=dict(color='red')
        ))

        # Add relative humidity line
        fig.add_trace(go.Scatter(
            x=weather_forecast_df["date"],
            y=weather_forecast_df["relative_humidity_2m"],
            mode='lines',
            name='Relative Humidity (%)',
            line=dict(color='blue')
        ))

        # Add wind speed line
        fig.add_trace(go.Scatter(
            x=weather_forecast_df["date"],
            y=weather_forecast_df["wind_speed_10m"],
            mode='lines',
            name='Wind Speed (m/s)',
            line=dict(color='green')
        ))

        # Update layout
        fig.update_layout(
            title="Weather Forecast for the Next 7 Days",
            xaxis_title="Date",
            yaxis_title="Values",
            hovermode="x",
            legend_title="Weather Variables"
        )
    
        # Display the plot
        st.plotly_chart(fig, theme="streamlit")
        
        # radio button for household cluster selection
        household_cluster = st.radio(
            "Select the type of household profile to view the energy forecast",
            ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "Cluster 6", "Cluster 7", 
                "Cluster 8", "Cluster 9", "Cluster 10"]
        )

        # Display the selected cluster
        st.write(f"You selected: {household_cluster}")
        
        # if st.button("Generate Energy Forecast"):
        #     # Prepare test data for Lambda
        #     test_data = []
        #     for i in range(len(weather_forecast_df)):
        #         test_data.append({
        #             "datetime": weather_forecast_df["date"].iloc[i].strftime("%Y-%m-%d %H:%M:%S"),  # Match Lambda's expected format
        #             "temperature": float(weather_forecast_df["temperature_2m"].iloc[i]),
        #             "humidity": float(weather_forecast_df["relative_humidity_2m"].iloc[i]),
        #             "windSpeed": float(weather_forecast_df["wind_speed_10m"].iloc[i]),
        #             "is_weekend": int(weather_forecast_df["is_weekend"].iloc[i]),
        #             "season": weather_forecast_df["season"].iloc[i],
        #             "household_cluster": household_cluster  
        #         })

        #     # API Gateway URL
        #     API_URL = "https://m1auibsxn8.execute-api.us-east-1.amazonaws.com/prod/call-sagemaker-endpoint"

        #     # Request payload
        #     request_payload = {"periods": 7, "test_data": test_data}

        #     # Call the Lambda function via API Gateway
        #     response = requests.post(API_URL, json=request_payload)

        #     if response.status_code == 200:
        #         # Parse the forecasted values from the response
        #         data = response.json()
        #         forecast_values = data["forecast"]

        #         # Convert to DataFrame
        #         forecast_df = pd.DataFrame(forecast_values, columns=["Predicted Energy"])
        #         forecast_df.index = weather_forecast_df["date"]

        #         # Plot results
        #         fig = go.Figure()
        #         fig.add_trace(go.Scatter(
        #             x=forecast_df.index,
        #             y=forecast_df["Predicted Energy"],
        #             mode='lines',
        #             name='Forecast',
        #             line=dict(color='blue', dash='solid')
        #         ))
        #         fig.update_layout(
        #             title="7-Day Hourly Energy Consumption Forecast",
        #             xaxis_title="Date and Time",
        #             yaxis_title="Predicted Energy Consumption (kWh)",
        #             hovermode="x"
        #         )

        #         st.plotly_chart(fig)
        #     else:
        #         st.error(f"Failed to get forecast. Error: {response.text}")
        
        # test w fake data first
        if st.button("Generate Energy Forecast"):
            # fake values
            fake_forecast_values = np.random.uniform(low=100, high=300, size=7)  # Daily total energy consumption
            lower_bound = fake_forecast_values - np.random.uniform(low=20, high=50, size=7)  # Lower confidence interval
            upper_bound = fake_forecast_values + np.random.uniform(low=20, high=50, size=7)  # Upper confidence interval

            daily_forecast_df = pd.DataFrame({
                "date": pd.date_range(start=weather_forecast_df["date"].iloc[0], periods=7, freq='D'),
                "predicted_energy": fake_forecast_values,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            })

            fig = go.Figure()

            # predicted energy
            fig.add_trace(go.Scatter(
                x=daily_forecast_df["date"],
                y=daily_forecast_df["predicted_energy"],
                mode='lines',
                name='Predicted Energy',
                line=dict(color='blue', dash='solid')
            ))

            # confidence interval 
            fig.add_trace(go.Scatter(
                x=pd.concat([daily_forecast_df["date"], daily_forecast_df["date"][::-1]]),
                y=pd.concat([daily_forecast_df["upper_bound"], daily_forecast_df["lower_bound"][::-1]]),
                fill='toself',
                fillcolor='rgba(100, 149, 237, 0.3)',  # Darker light blue fill
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='Confidence Interval'
            ))


            fig.update_layout(
                title="7-Day Daily Energy Consumption Forecast",
                xaxis_title="Date",
                yaxis_title="Predicted Energy Consumption (kWh)",
                hovermode="x",
                legend_title="Forecast Components"
            )

            # Display the plot
            st.plotly_chart(fig)
    else:
        st.error("Failed to retrieve weather data.")
        
    

with dunkelflaute_tab:
    st.image("https://www.concertoplus.eu/wp-content/uploads/2016/08/cre-page-banner.jpg", use_container_width=True)
    st.subheader("Dunkelflaute Prediction")
    st.write("Analyze the probability of Dunkelflaute for the next 7 days based on forecasted weather data.")

    if st.button("Generate Dunkelflaute Prediction"):
        # Step 1: Fetch weather forecast data for the next 7 days
        weather_forecast_df = get_weather_forecast()

        if weather_forecast_df is not None:
            # Step 2: Generate fake Dunkelflaute probabilities
            # Simulate hourly probabilities (168 hours for 7 days)
            hourly_probabilities = np.random.uniform(low=0, high=1, size=len(weather_forecast_df))

            # Simulate daily aggregated Dunkelflaute classification (Yes/No)
            daily_aggregated = ["Yes" if np.random.uniform(0, 1) > 0.7 else "No" for _ in range(7)]

            # Step 3: Display the heatmap (zoomed-in view)
            st.write("### Hourly Dunkelflaute Probability Heatmap")
            heatmap_data = pd.DataFrame({
                "datetime": weather_forecast_df["date"],
                "probability": hourly_probabilities
            })
            heatmap_data["hour"] = heatmap_data["datetime"].dt.hour
            heatmap_data["day"] = heatmap_data["datetime"].dt.date

            heatmap_pivot = heatmap_data.pivot(index="hour", columns="day", values="probability")
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                colorscale="Viridis",
                colorbar=dict(title="Probability")
            ))
            fig.update_layout(
                title="Hourly Dunkelflaute Probability Heatmap",
                xaxis_title="Date",
                yaxis_title="Hour of Day"
            )
            st.plotly_chart(fig)

            # Step 4: Display the 7-day aggregated forecast (zoomed-out view)
            st.write("### 7-Day Aggregated Dunkelflaute Forecast")
            daily_forecast_df = pd.DataFrame({
                "date": pd.date_range(start=weather_forecast_df["date"].iloc[0], periods=7, freq='D'),
                "dunkelflaute": daily_aggregated
            })
            st.table(daily_forecast_df)

        else:
            st.error("Failed to retrieve weather data.")
