import streamlit as st
import pandas as pd
import numpy as np
np.float_ = np.float64
from plotly import graph_objs as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Ensure page configuration is the first Streamlit command
st.set_page_config(page_title="Trend and Forecast App", page_icon="📈")

# Function to load data
@st.cache_data
def load_data():
    consumption_file_path = 'data (25).csv'
    service_file_path = 'data (27).csv'
    consumption_data = pd.read_csv(consumption_file_path)
    service_data = pd.read_csv(service_file_path)

    # Add a 'type' column to differentiate between consumption and service
    consumption_data['type'] = 'consumption'
    service_data['type'] = 'service'

    # Concatenate the dataframes
    combined_data = pd.concat([consumption_data, service_data], ignore_index=True)

    # Melting the DataFrame to long format
    long_data = combined_data.melt(id_vars=['periodid', 'organisationunitname', 'type'],
                                   var_name='product_description', value_name='quantity')

    # Check for null values in 'quantity' and fill with 0
    long_data['quantity'].fillna(0, inplace=True)

    # Mapping dictionary based on the given data
    COLUMN_MAPPING = {
        'MOH 747A_Male Condoms': 'Male Condoms',
        'MOH 711 Client receiving Male condoms': 'Male Condoms',
        'MOH 747A_Female Condoms': 'Female Condoms',
        'MOH 711 Clients receiving Female Condoms': 'Female Condoms',
        'MOH 747A_Emergency Contraceptive pills': 'EC Pills',
        'MOH 711 Emergency contraceptive pill': 'EC Pills',
        'MOH 747A_Combined Oral contraceptive Pills': 'COCs',
        'MOH 711 Pills Combined oral contraceptive': 'COCs',
        'MOH 747A_Progestin only pills': 'POPs',
        'MOH 711 Pills progestin only': 'POPs',
        'MOH 747A_Cycle Beads': 'Cycle Beads',
        'MOH 711 Rev 2020_Clients given cycle beads': 'Cycle Beads',
        'MOH 747A_DMPA-IM': 'DMPA-IM',
        'MOH 711 Rev 2020_FP Injections DMPA- IM': 'DMPA-IM',
        'MOH 747A_DMPA-SC': 'DMPA-SC',
        'MOH 711 Rev 2020_FP Injections DMPA- SC': 'DMPA-SC',
        'MOH 747A_Implants (1-Rod) – ENG 68mg': 'Implanon',
        'MOH 711 Rev 2020_Implants insertion 1 Rod': 'Implanon',
        'MOH 711 Rev 2020_Implants insertion 2 Rod': '2 Rod',
        'MOH 747A_Hormonal IUCD': 'Hormonal IUCD',
        'MOH 711 Rev 2020_IUCD Insertion Hormonal': 'Hormonal IUCD',
        'MOH 747A_Non-Hormonal IUCD': 'Non-Hormonal IUCD',
        'MOH 711 Rev 2020_IUCD Insertion Non Hormonal': 'Non-Hormonal IUCD',
        'MOH 747A_Implants (2-Rod) - LNG 75mg (3 years)': 'Levoplant',
        'MOH 747A_Implant (2-Rod) – LNG 75mg (5 years)': 'Jadelle'
    }

    # Function to map common names
    def map_common_name(description):
        for key, value in COLUMN_MAPPING.items():
            if key in description:
                return value
        return 'Unknown'

    # Apply the mapping function
    long_data['common_name'] = long_data['product_description'].apply(map_common_name)

    # Remove rows with 'Unknown' common_name if necessary
    long_data = long_data[long_data['common_name'] != 'Unknown']

    # Aggregating the data by common_name, periodid, and organisationunitname
    aggregated_data = long_data.groupby(['common_name', 'periodid', 'organisationunitname', 'type']).agg({'quantity': 'sum'}).reset_index()

    # Ensure 'quantity' is numeric
    aggregated_data['quantity'] = pd.to_numeric(aggregated_data['quantity'], errors='coerce')

    # Fill missing values with 0
    aggregated_data['quantity'].fillna(0, inplace=True)

    return aggregated_data

data = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Trend and Forecast", "Product Comparison"])

if page == "Trend and Forecast":
    st.title('Family Planning Forecast Tool')

    # Add clickable link
    st.sidebar.markdown("[InSupply Health](https://insupplyhealth.com/)")

    # Sidebar filters
    product = st.sidebar.selectbox('Select Product', data['common_name'].unique())
    county = st.sidebar.selectbox('Select County', ['Kenya'] + list(data['organisationunitname'].unique()))
    type_options = ['consumption', 'service', 'both']
    selected_type = st.sidebar.selectbox('Select Type', type_options)

    # Filter data based on selections
    @st.cache_data
    def filter_data(product, county, selected_type, data):
        if county == 'Kenya':
            filtered_data = data[data['common_name'] == product]
        else:
            filtered_data = data[(data['common_name'] == product) & (data['organisationunitname'] == county)]

        if selected_type != 'both':
            filtered_data = filtered_data[filtered_data['type'] == selected_type]

        filtered_data['periodid'] = pd.to_datetime(filtered_data['periodid'], format='%Y%m', errors='coerce')
        
        return filtered_data

    filtered_data = filter_data(product, county, selected_type, data)

    if selected_type == 'both':
        st.subheader(f'Trend for {product} in {county}')
        
        # Separate data for consumption and service
        consumption_data = filtered_data[filtered_data['type'] == 'consumption']
        service_data = filtered_data[filtered_data['type'] == 'service']

        # Aggregate data by period
        aggregated_consumption = consumption_data.groupby('periodid')['quantity'].sum().reset_index()
        aggregated_service = service_data.groupby('periodid')['quantity'].sum().reset_index()

        # Create interactive plots for consumption and service
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=aggregated_consumption['periodid'], y=aggregated_consumption['quantity'],
                                 mode='lines+markers', name='Consumption'))
        fig.add_trace(go.Scatter(x=aggregated_service['periodid'], y=aggregated_service['quantity'],
                                 mode='lines+markers', name='Service'))
        fig.update_layout(title=f'Trend of {product} in {county}', xaxis_title='Period', yaxis_title='Quantity')
        st.plotly_chart(fig)

        # Forecasting for both types separately
        st.subheader(f'Forecast for {product} in {county}')

        for data_type, data_label in zip([aggregated_consumption, aggregated_service], ['Consumption', 'Service']):
            if not data_type.empty:
                model = SARIMAX(data_type['quantity'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                model_fit = model.fit(disp=False)
                forecast = model_fit.get_forecast(steps=12)
                forecast_index = pd.date_range(start=data_type['periodid'].iloc[-1], periods=13, freq='M')[1:]
                forecast_values = forecast.predicted_mean
                conf_int = forecast.conf_int()
                conf_int.columns = ['lower', 'upper']
                forecast_values = np.maximum(forecast_values, 0)
                conf_int['lower'] = np.maximum(conf_int['lower'], 0)
                conf_int['upper'] = np.maximum(conf_int['upper'], 0)

                # Plot forecast
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data_type['periodid'], y=data_type['quantity'],
                                         mode='lines+markers', name='Observed'))
                fig.add_trace(go.Scatter(x=forecast_index, y=forecast_values,
                                         mode='lines+markers', name='Forecast'))
                fig.add_trace(go.Scatter(x=forecast_index, y=conf_int['upper'], fill=None, mode='lines',
                                         line=dict(color='gray'), showlegend=False))
                fig.add_trace(go.Scatter(x=forecast_index, y=conf_int['lower'], fill='tonexty', mode='lines',
                                         line=dict(color='gray'), fillcolor='rgba(0,100,80,0.2)', showlegend=False))
                fig.update_layout(title=f'{data_label} Forecast of {product} in {county}', xaxis_title='Date', yaxis_title='Quantity')
                st.plotly_chart(fig)

                # Display forecast table
                st.subheader(f'{data_label} Forecast Table')
                forecast_table = pd.DataFrame({
                    'Date': forecast_index,
                    'Forecast': forecast_values,
                    'Lower CI': conf_int['lower'],
                    'Upper CI': conf_int['upper']
                })
                st.write(forecast_table)

    else:
        filtered_data = filtered_data[filtered_data['type'] == selected_type]

        # Aggregate the data by period
        aggregated_data = filtered_data.groupby('periodid')['quantity'].sum().reset_index()

        # Display the trend
        st.subheader(f'Trend for {product} in {county}')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=aggregated_data['periodid'], y=aggregated_data['quantity'], mode='lines+markers'))
        fig.update_layout(title=f'Trend of {product} in {county}', xaxis_title='Period', yaxis_title='Quantity')
        st.plotly_chart(fig)

        # Forecasting
        st.subheader(f'Forecast for {product} in {county}')
        model = SARIMAX(aggregated_data['quantity'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)

        # Forecast for the next 12 periods
        forecast = model_fit.get_forecast(steps=12)
        forecast_index = pd.date_range(start=aggregated_data['periodid'].iloc[-1], periods=13, freq='M')[1:]
        forecast_values = forecast.predicted_mean
        conf_int = forecast.conf_int()
        conf_int.columns = ['lower', 'upper']

        # Adjust forecast values to avoid negatives
        forecast_values = np.maximum(forecast_values, 0)
        conf_int['lower'] = np.maximum(conf_int['lower'], 0)
        conf_int['upper'] = np.maximum(conf_int['upper'], 0)

        # Plot the forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=aggregated_data['periodid'], y=aggregated_data['quantity'], name='Observed', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast_values, name='Forecast', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=forecast_index, y=conf_int['upper'], fill=None, mode='lines', line=dict(color='gray'), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_index, y=conf_int['lower'], fill='tonexty', mode='lines', line=dict(color='gray'), fillcolor='rgba(0,100,80,0.2)', showlegend=False))
        fig.update_layout(title=f'Forecast of {product} in {county}', xaxis_title='Date', yaxis_title='Quantity')
        st.plotly_chart(fig)

        # Display forecast table
        st.subheader('Forecast Table')
        forecast_table = pd.DataFrame({
            'Date': forecast_index,
            'Forecast': forecast_values,
            'Lower CI': conf_int['lower'],
            'Upper CI': conf_int['upper']
        })
        st.write(forecast_table)

elif page == "Product Comparison":
    st.title('Product Comparison')

    # Sidebar filters
    selected_products = st.sidebar.multiselect('Select Products', data['common_name'].unique())
    county = st.sidebar.selectbox('Select County', ['Kenya'] + list(data['organisationunitname'].unique()))
    type_options = ['consumption', 'service', 'both']
    selected_type = st.sidebar.selectbox('Select Type', type_options)

    if selected_products:
        fig = go.Figure()
        for product in selected_products:
            if county == 'Kenya':
                filtered_data = data[data['common_name'] == product]
            else:
                filtered_data = data[(data['common_name'] == product) & (data['organisationunitname'] == county)]

            if selected_type != 'both':
                filtered_data = filtered_data[filtered_data['type'] == selected_type]

            # Convert periodid to datetime format
            filtered_data['periodid'] = pd.to_datetime(filtered_data['periodid'], format='%Y%m', errors='coerce')

            # Aggregate the data by period
            aggregated_data = filtered_data.groupby('periodid')['quantity'].sum().reset_index()

            # Plot the trend for each selected product
            fig.add_trace(go.Scatter(x=aggregated_data['periodid'], y=aggregated_data['quantity'], mode='lines+markers', name=product))

        fig.update_layout(title=f'Trend Comparison in {county}', xaxis_title='Period', yaxis_title='Quantity')
        st.plotly_chart(fig)
