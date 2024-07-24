from fastapi import FastAPI, HTTPException
import pandas as pd
from prophet import Prophet
from typing import Dict, Any
from functools import lru_cache

app = FastAPI()

# Load and preprocess the data once at startup
data = pd.read_csv('combined_product_data (1).csv')
data = data.dropna(subset=['periodname'])

def parse_dates(date_str):
    for fmt in ('%B %Y', '%b-%y'):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    return pd.NaT  # Return NaT if no format matches

data['periodname'] = data['periodname'].apply(parse_dates)
data = data.dropna(subset=['periodname'])

@lru_cache(maxsize=32)
def get_filtered_data(org_unit: str, common_name: str, data_type: str) -> pd.DataFrame:
    filtered_data = data[(data['organisationunitname'] == org_unit) & (data['common_name'] == common_name)]
    if data_type != 'Both':
        filtered_data = filtered_data[filtered_data['Type'] == data_type]
    return filtered_data

@app.get("/forecast")
def get_forecast(org_unit: str, common_name: str, data_type: str = 'Both') -> Dict[str, Any]:
    filtered_data = get_filtered_data(org_unit, common_name, data_type)
    if filtered_data.empty:
        raise HTTPException(status_code=404, detail="Data not found")

    data_prophet = filtered_data[['periodname', 'value']].rename(columns={'periodname': 'ds', 'value': 'y'})
    model = Prophet()
    model.fit(data_prophet)
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)
    forecast = forecast[forecast['ds'] >= '2024-07-01']
    forecast_dict = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')
    return {"forecast": forecast_dict}
