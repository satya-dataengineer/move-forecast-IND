from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import pandas as pd
from prophet import Prophet
import pickle
from datetime import datetime, timedelta
import pytz
from typing import List, Dict
import os
import logging
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# MCP API Key
MCP_API_KEY = "YmFzZTY0c3RyaW5nZm9ybWNwYXV0aGVudGljYXRpb24"

# Dependency to validate MCP API key
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != MCP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# Paths to datasets and models (relative to main.py)
DATA_PATH = r"C:\Users\sswain_quantum-i\OneDrive\Desktop"
MODEL_PATH = os.path.join(DATA_PATH, "Prophet_Models")

# Load datasets
try:
    forecasting_data = pd.read_csv(os.path.join(DATA_PATH, "Forecasting_Data.csv"))
    final_data = pd.read_csv(os.path.join(DATA_PATH, "Final_Data.csv"))
except FileNotFoundError as e:
    logger.error(f"Dataset not found: {e}")
    raise HTTPException(status_code=500, detail=f"Dataset not found: {e}")

# Convert date columns to datetime
try:
    forecasting_data['Date'] = pd.to_datetime(forecasting_data['Date'], format="%Y-%m-%d")
    final_data['Date'] = pd.to_datetime(final_data['Date'], format="%Y-%m-%d")
except Exception as e:
    logger.error(f"Error parsing dates: {e}")
    raise HTTPException(status_code=500, detail=f"Error parsing dates: {e}")

# Pydantic models for request/response
class ForecastRequest(BaseModel):
    date: str  # Expected format: DD-MM-YYYY
    branch: str
    move_type: str

class ForecastResponse(BaseModel):
    dates: List[str]
    counts: List[float]
    move_size_counts: Dict[str, float]
    crew_trucks: Dict[str, Dict[str, int]]
    hourly_rates: Dict[str, float]
    historical_counts: Dict[str, float]

# Helper function to load branch-specific Prophet model
def load_prophet_model(branch: str) -> Prophet:
    model_file = os.path.join(MODEL_PATH, f"prophet_model_{branch}.pkl")
    if not os.path.exists(model_file):
        logger.error(f"No model found for branch: {branch}")
        raise HTTPException(status_code=400, detail=f"No model found for branch: {branch}")
    try:
        with open(model_file, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading model for branch {branch}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model for branch {branch}: {e}")

# Helper function to get 7-day window
def get_forecast_window(input_date: datetime) -> tuple[datetime, datetime]:
    current_date = datetime.now(pytz.timezone("Asia/Kolkata")).replace(tzinfo=None)
    last_forecast_date = datetime(2025, 7, 31)
    
    if (input_date - current_date).days < 3:
        start_date = current_date
        end_date = start_date + timedelta(days=6)
    elif (last_forecast_date - input_date).days < 3:
        end_date = last_forecast_date
        start_date = end_date - timedelta(days=6)
    else:
        start_date = input_date - timedelta(days=3)
        end_date = input_date + timedelta(days=3)
    return start_date, end_date

# Calculate move type and move size percentages for last 4 years
def calculate_percentages(branch: str, move_type: str, input_date: datetime) -> tuple[Dict[str, float], float]:
    start_year = input_date.year - 4
    end_year = input_date.year - 1
    historical = final_data[(final_data['Date'].dt.year.between(start_year, end_year)) & 
                           (final_data['Branch'] == branch)]
    
    # Move type percentages
    total_counts = historical.groupby('MoveType')['Count'].sum()
    move_type_percentages = (total_counts / total_counts.sum() * 100).to_dict()
    move_type_pct = move_type_percentages.get(move_type, 0.0) / 100 if total_counts.sum() > 0 else 0.0
    
    # Move size percentages for the given move type
    move_type_data = historical[historical['MoveType'] == move_type]
    move_size_counts = move_type_data.groupby('MoveSize')['Count'].sum()
    move_size_percentages = (move_size_counts / move_size_counts.sum() * 100).to_dict() if move_size_counts.sum() > 0 else {ms: 0.0 for ms in final_data['MoveSize'].unique()}
    move_size_percentages = {ms: move_size_percentages.get(ms, 0.0) / 100 for ms in final_data['MoveSize'].unique()}
    
    return move_size_percentages, move_type_pct

# Forecast endpoint
@app.post("/forecast", response_model=ForecastResponse, dependencies=[Depends(verify_api_key)])
async def forecast(request: ForecastRequest):
    try:
        # Parse input date (DD-MM-YYYY)
        input_date = pd.to_datetime(request.date, format="%d-%m-%Y").replace(tzinfo=None)
        branch = request.branch
        move_type = request.move_type

        # Validate inputs
        if branch not in forecasting_data['Branch'].unique():
            logger.error(f"Invalid branch: {branch}")
            raise HTTPException(status_code=400, detail="Invalid branch")
        if move_type not in final_data['MoveType'].unique():
            logger.error(f"Invalid move type: {move_type}")
            raise HTTPException(status_code=400, detail="Invalid move type")

        # Load branch-specific Prophet model
        model = load_prophet_model(branch)

        # Get 7-day window
        start_date, end_date = get_forecast_window(input_date)
        future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})

        # Forecast counts for branch
        forecast = model.predict(future_df)
        forecast_counts = forecast[['ds', 'yhat']].set_index('ds')['yhat'].clip(lower=0).to_dict()
        forecast_dates = [dt.strftime("%Y-%m-%d") for dt in future_dates]
        forecast_values = [round(forecast_counts.get(dt, 0), 2) for dt in future_dates]

        # Adjust for move type using last 4 years' average percentage
        move_size_percentages, move_type_pct = calculate_percentages(branch, move_type, input_date)
        input_date_count = forecast_counts.get(input_date, 0) * move_type_pct
        move_size_counts = {ms: round(input_date_count * pct, 2) for ms, pct in move_size_percentages.items()}

        # Get crew size, trucks, and hourly rates for input date
        crew_trucks = {}
        hourly_rates = {}
        for ms in final_data['MoveSize'].unique():
            ms_data = final_data[(final_data['Date'].dt.date == input_date.date()) & 
                                (final_data['Branch'] == branch) & 
                                (final_data['MoveType'] == move_type) & 
                                (final_data['MoveSize'] == ms)]
            if not ms_data.empty:
                crew_trucks[ms] = {
                    'CrewSize': int(ms_data['CrewSize'].iloc[0]),
                    'Trucks': int(ms_data['Trucks'].iloc[0])
                }
                hourly_rates[ms] = round(float(ms_data['HourlyRate'].iloc[0]), 2)
            else:
                crew_trucks[ms] = {'CrewSize': 0, 'Trucks': 0}
                hourly_rates[ms] = 0.0

        # Get historical counts for past 3 years
        historical_counts = {}
        for year in range(2022, 2025):
            past_date = input_date.replace(year=year)
            past_data = final_data[(final_data['Date'].dt.date == past_date.date()) & 
                                  (final_data['Branch'] == branch) & 
                                  (final_data['MoveType'] == move_type)]
            historical_counts[str(year)] = round(past_data['Count'].sum(), 2) if not past_data.empty else 0.0

        return ForecastResponse(
            dates=forecast_dates,
            counts=forecast_values,
            move_size_counts=move_size_counts,
            crew_trucks=crew_trucks,
            hourly_rates=hourly_rates,
            historical_counts=historical_counts
        )
    except Exception as e:
        logger.error(f"Error in forecast endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Trend endpoint
@app.post("/trend", response_model=Dict[str, List], dependencies=[Depends(verify_api_key)])
async def get_trend(request: ForecastRequest):
    try:
        input_date = pd.to_datetime(request.date, format="%d-%m-%Y")
        branch = request.branch
        move_type = request.move_type

        # Validate inputs
        if branch not in forecasting_data['Branch'].unique():
            logger.error(f"Invalid branch: {branch}")
            raise HTTPException(status_code=400, detail="Invalid branch")
        if move_type not in final_data['MoveType'].unique():
            logger.error(f"Invalid move type: {move_type}")
            raise HTTPException(status_code=400, detail="Invalid move type")

        # Load branch-specific Prophet model
        model = load_prophet_model(branch)

        # Get historical data for 2022–2024 and forecast for 2025
        trend_data = {'years': [], 'counts': []}
        for year in range(2022, 2026):
            year_date = input_date.replace(year=year)
            if year == 2025:
                future_df = pd.DataFrame({'ds': [year_date]})
                forecast = model.predict(future_df)
                move_size_percentages, move_type_pct = calculate_percentages(branch, move_type, year_date)
                count = round(forecast['yhat'].iloc[0] * move_type_pct, 2)
            else:
                data = final_data[(final_data['Date'].dt.date == year_date.date()) & 
                                 (final_data['Branch'] == branch) & 
                                 (final_data['MoveType'] == move_type)]
                count = round(data['Count'].sum(), 2) if not data.empty else 0.0
            trend_data['years'].append(str(year))
            trend_data['counts'].append(count)
        return trend_data
    except Exception as e:
        logger.error(f"Error in trend endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
