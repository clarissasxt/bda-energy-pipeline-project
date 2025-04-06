# helper_functions.py

import openmeteo_requests
import requests_cache
import pandas as pd
import boto3
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from retry_requests import retry
import plotly.graph_objects as go
import plotly.express as px
import json

# Function to get weather data from Open-Meteo API
def get_weather_data(latitude, longitude, variables):
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": variables,
        "timezone": "Europe/London"
    }

    responses = openmeteo.weather_api(url, params=params)
    return responses[0]

# Function to process the weather data and prepare dataframe
def process_weather_data(response):
    hourly = response.Hourly()
    hourly_wind_speed_120m = hourly.Variables(0).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(1).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}
    hourly_data["wind_speed_120m"] = hourly_wind_speed_120m
    hourly_data["cloud_cover"] = hourly_cloud_cover

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    return hourly_dataframe

# Function to prepare data for SageMaker prediction
def prepare_data_for_sagemaker(hourly_dataframe):
    # Reformat scale of wind speed and cloud cover: 0-100 to 0-1
    hourly_dataframe['wind_speed_120m'] = hourly_dataframe['wind_speed_120m'] / 100
    hourly_dataframe['cloud_cover'] = hourly_dataframe['cloud_cover'] / 100
    hourly_dataframe.rename(columns={'date': 'datetime'}, inplace=True)
    hourly_df_final = hourly_dataframe[['cloud_cover', 'wind_speed_120m']]

    # Drop headers & index
    csv_buffer = io.StringIO()
    hourly_df_final.to_csv(csv_buffer, index=False, header=False)
    payload = csv_buffer.getvalue()

    return payload, hourly_dataframe

# Function to invoke SageMaker endpoint and get prediction results
def get_sagemaker_predictions(payload, endpoint_name, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY):
    sagemaker_runtime = boto3.client(
        "sagemaker-runtime",
        region_name="us-east-1",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Body=payload
    )

    result = response['Body'].read().decode('utf-8')
    output_df = pd.read_csv(io.StringIO(result), header=None, names=['probability'])
    return output_df

# Function to invoke Lambda and get prediction results
def get_lambda_predictions(payload, lambda_function_name, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY):
    lambda_client = boto3.client(
        'lambda',
        region_name='us-east-1',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    # Preparing the input data for the Lambda function
    input_data = {
        "body": json.dumps({
            "test_data": payload,
            "cluster": "1"  # Example cluster name, update as necessary
        })
    }

    # Invoke the Lambda function
    response = lambda_client.invoke(
        FunctionName=lambda_function_name,
        InvocationType='RequestResponse',  # Synchronous invocation
        Payload=json.dumps(input_data)
    )

    # Parse the Lambda response
    result = json.loads(response['Payload'].read().decode('utf-8'))
    # Print the result to understand the structure
    print("Lambda response:", result)

    if response['StatusCode'] == 200:
        # Assuming the result has a "forecast" field with prediction probabilities
        output_df = pd.DataFrame(result['probability'])
        return output_df
    else:
        raise Exception(f"Error invoking Lambda: {result}")

# Function to prepare the final data frame for visualization
def prepare_final_df(hourly_dataframe, output_df):
    hourly_df_with_date = hourly_dataframe[['datetime']].reset_index(drop=True)
    final_df = pd.concat([hourly_df_with_date, output_df], axis=1)

    final_df['date'] = pd.to_datetime(final_df['datetime'])
    final_df['date'] = final_df['datetime'].dt.date
    final_df['hour_of_day'] = final_df['datetime'].dt.hour

    return final_df

# Function to plot the heatmap
def plot_heatmap(final_df):
    # Pivot the dataframe to get the data for the heatmap
    heatmap_data = final_df.pivot_table(index='date', columns='hour_of_day', values='probability', aggfunc=np.mean)
    heatmap_data.to_csv('heatmap_data.csv', index=True)  # Save the heatmap data to a CSV file
    # Replace NaN values with 'NA' or any other placeholder you want
    heatmap_data_filled = heatmap_data.fillna('')
    
    # Create a Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data_filled.values,
        x=heatmap_data_filled.columns,
        y=heatmap_data_filled.index,
        colorscale='reds',  # Red color theme
        colorbar=dict(title='Dunkelflaute Probability'),
        text=heatmap_data_filled.values,  # Add the probability numbers to show inside each cell
        hovertemplate='Date: %{y}<br>Hour: %{x}<br>Probability: %{z:.2f}<extra></extra>',
        showscale=True,
        # Annotate values inside the cells
        texttemplate="%{text:.2f}",  # Display probability with two decimals inside the cells
        textfont=dict(size=12, color="black")  # Change font size and color for readability
    ))

    # Customize the layout
    fig.update_layout(
        title='Dunkelflaute Probability Heatmap (Predicted Probabilities for the Week)',
        xaxis_title='Hour of Day',
        yaxis_title='Date',
        xaxis=dict(tickmode='array', tickvals=list(range(24)), ticktext=[f'{i}:00' for i in range(24)]),
        yaxis=dict(tickmode='array', tickvals=heatmap_data_filled.index),
    )

    # Show the plot
    return fig
    
# Function to plot the time-series forecast
def plot_timeseries(final_df):
    fig = px.line(final_df, x='datetime', y='probability', title='Dunkelflaute Probability Time Series (Predicted Probabilities)', markers=True)

    # Customize the layout
    fig.update_layout(
        xaxis_title='Date and Hour',
        yaxis_title='Dunkelflaute Probability',
        xaxis=dict(tickformat='%Y-%m-%d %H:%M', tickangle=45),
        title_font=dict(size=16),
    )

    # Show the plot
    return fig

# # Function to plot the heatmap
# def plot_heatmap(final_df):
#     heatmap_data = final_df.pivot_table(index='date', columns='hour_of_day', values='probability', aggfunc=np.mean)

#     plt.figure(figsize=(12, 8))
#     sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f', cbar_kws={'label': 'Dunkelflaute Probability'})
#     plt.title('Dunkelflaute Probability Heatmap (Predicted Probabilities for the Week)', fontsize=16)
#     plt.xlabel('Hour of Day', fontsize=14)
#     plt.ylabel('Date', fontsize=14)
#     plt.xticks(ticks=np.arange(24), labels=[f'{i}:00' for i in range(24)], rotation=45)
#     plt.yticks(rotation=0)
#     plt.tight_layout()
#     plt.show()

# # Function to plot the time-series forecast
# def plot_timeseries(final_df):
#     plt.figure(figsize=(14, 8))
#     sns.lineplot(x='datetime', y='probability', data=final_df, marker='o', color='b')

#     plt.title('Dunkelflaute Probability Time Series (Predicted Probabilities)', fontsize=16)
#     plt.xlabel('Date and Hour', fontsize=14)
#     plt.ylabel('Dunkelflaute Probability', fontsize=14)
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.show()
