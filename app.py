import pandas as pd

# Load the combined data
combined_data = pd.read_csv('combined_data.csv')

# Convert 'periodname' to datetime if it's not already
combined_data['periodname'] = pd.to_datetime(combined_data['periodname'], errors='coerce')

# Drop rows with NaN values in 'periodname'
combined_data.dropna(subset=['periodname'], inplace=True)

# Aggregate data to the monthly level
combined_data.set_index('periodname', inplace=True)
monthly_data = combined_data.resample('M').sum()

# Reset index to make 'periodname' a column again
monthly_data.reset_index(inplace=True)

# Debug print: Check the first few rows of the aggregated data
print(monthly_data.head())

# Save the preprocessed data to a new CSV file
monthly_data.to_csv('preprocessed_data.csv', index=False)
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the preprocessed data
monthly_data = pd.read_csv('preprocessed_data.csv', parse_dates=['periodname'])

# Set 'periodname' as the index
monthly_data.set_index('periodname', inplace=True)

# Select a column for forecasting (e.g., 'Male Condoms')
data_column = 'Male Condoms'
series = monthly_data[data_column].dropna()

# Split the data into training and test sets
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# Fit the ARIMA model
model = ARIMA(train, order=(5,1,0))  # Adjust order as necessary
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train, label='Training Data')
plt.plot(test.index, test, label='Actual Data')
plt.plot(test.index, forecast, label='Forecast')
plt.xlabel('Date')
plt.ylabel(data_column)
plt.title(f'Forecast vs Actual for {data_column}')
plt.legend()
plt.show()
import streamlit as st
import pandas as pd
import numpy as np
from plotly import graph_objs as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the data
data_file_path = 'combined_data.csv'
combined_data = pd.read_csv(data_file_path)

# Convert 'periodname' to datetime
combined_data['periodname'] = pd.to_datetime(combined_data['periodname'], format='%b-%y', errors='coerce')

# Set page title
st.title("Family Planning Commodities Analysis")

# Filters for selecting product, organization, and data type
products = [
    'Male Condoms', 'Female Condoms', 'EC Pills', 'COCs', 'POPs',
    'Cycle Beads', 'DMPA-IM', 'DMPA-SC', 'Implanon', '2 Rod',
    'Hormonal IUCD', 'Non-Hormonal IUCD', 'Levoplant', 'Jadelle'
]

organizations = combined_data['organisationunitname'].unique()
data_types = ['Consumption', 'Service', 'Both']

selected_product = st.selectbox("Select Product", products)
selected_organization = st.selectbox("Select Organization", organizations)
selected_data_type = st.selectbox("Select Data Type", data_types)

# Filter data based on selections
if selected_data_type == 'Consumption':
    filtered_data = combined_data[(combined_data[selected_product].notna()) & (combined_data['Type'] == 'Consumption')]
elif selected_data_type == 'Service':
    filtered_data = combined_data[(combined_data[selected_product].notna()) & (combined_data['Type'] == 'Service')]
else:
    filtered_data = combined_data[combined_data[selected_product].notna()]

filtered_data = filtered_data[filtered_data['organisationunitname'] == selected_organization]

# Interactive plot to show data labels
def plot_interactive(data, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['periodname'], y=data[selected_product], mode='lines+markers', name=title, text=data[selected_product], hoverinfo='text'))

    fig.update_layout(
        title=title,
        xaxis_title='Period',
        yaxis_title='Quantity',
        hovermode='closest'
    )
    return fig

# Buttons for showing trends and forecasts
if st.button("Show Trend"):
    st.subheader(f"Trend for {selected_product} in {selected_organization}")
    if selected_data_type == 'Both':
        consumption_data = filtered_data[filtered_data['Type'] == 'Consumption']
        service_data = filtered_data[filtered_data['Type'] == 'Service']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=consumption_data['periodname'], y=consumption_data[selected_product], mode='lines+markers', name='Consumption', text=consumption_data[selected_product], hoverinfo='text'))
        fig.add_trace(go.Scatter(x=service_data['periodname'], y=service_data[selected_product], mode='lines+markers', name='Service', text=service_data[selected_product], hoverinfo='text'))
        fig.update_layout(title=f"Trend for {selected_product} in {selected_organization}", xaxis_title='Period', yaxis_title='Quantity', hovermode='closest')
    else:
        fig = plot_interactive(filtered_data, f"Trend for {selected_product} in {selected_organization}")

    st.plotly_chart(fig)

if st.button("Show Forecast"):
    st.subheader(f"Forecast for {selected_product} in {selected_organization}")
    model = ExponentialSmoothing(filtered_data[selected_product], seasonal='add', seasonal_periods=12).fit()
    forecast = model.forecast(steps=12)
    forecast_index = pd.date_range(start=filtered_data['periodname'].iloc[-1], periods=12, freq='M')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data['periodname'], y=filtered_data[selected_product], mode='lines+markers', name='Actual', text=filtered_data[selected_product], hoverinfo='text'))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines+markers', name='Forecast', text=forecast, hoverinfo='text'))

    fig.update_layout(title=f"Forecast for {selected_product} in {selected_organization}", xaxis_title='Period', yaxis_title='Quantity', hovermode='closest')
    st.plotly_chart(fig)

    # Display forecast table
    forecast_table = pd.DataFrame({'Date': forecast_index, 'Forecasted Quantity': forecast})
    st.subheader("Forecast Table")
    st.table(forecast_table)

