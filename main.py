import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import os
import pickle
import warnings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import random
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')  # Suppress Prophet logging

# Initialize FastAPI app
app = FastAPI(title="Move Forecast API", description="API to forecast move counts using Prophet models")

# PostgreSQL connection configuration
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

# Function to fetch data from PostgreSQL
def fetch_data(query, params=None):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        df = pd.read_sql_query(query, conn, params=params)
        return df
    except Exception as e:
        logger.error(f"Error fetching data from PostgreSQL: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

# Function to initialize PostgreSQL (verify connection)
def init_db():
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        logger.info("PostgreSQL connection verified successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        if conn is not None:
            conn.close()

# Initialize database
init_db()

# Function to fetch precomputed percentages
def fetch_historical_percentages(branch, move_type, month, day):
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Try day-specific percentage
            cursor.execute("""
                SELECT avg_percentage
                FROM historical_percentages
                WHERE branch = %s AND move_type = %s AND month = %s AND day = %s
            """, (branch, move_type, month, day))
            result = cursor.fetchone()
            if result:
                logger.info(f"Found day-specific percentage: {result['avg_percentage']}% for {branch}, {move_type}, month {month}, day {day}")
                return result['avg_percentage']
            
            # Fallback to monthly average
            cursor.execute("""
                SELECT AVG(avg_percentage) as avg_percentage
                FROM historical_percentages
                WHERE branch = %s AND move_type = %s AND month = %s
            """, (branch, move_type, month))
            result = cursor.fetchone()
            if result and result['avg_percentage'] is not None:
                logger.info(f"Using monthly average percentage: {result['avg_percentage']}% for {branch}, {move_type}, month {month}")
                return result['avg_percentage']
            
            # Fallback to minimal percentage
            minimal_percentage = 1.0  # 1% as a logical default
            logger.warning(f"No percentage data for {branch}, {move_type}, month {month}. Using minimal percentage: {minimal_percentage}%")
            return minimal_percentage
    except Exception as e:
        logger.error(f"Error fetching historical percentage: {str(e)}")
        return 1.0  # Minimal percentage on error
    finally:
        if conn is not None:
            conn.close()

# Define input model for request validation
class ForecastInput(BaseModel):
    date: str  # e.g., "2025-06-11"
    branch: str  # e.g., "Columbus"
    move_type: Optional[str] = None  # e.g., "Local"

# Comment phrases
CONSISTENT_PHRASES = [
    "Demand for {move_type} moves aligns closely with historical patterns (historical avg {hist_avg:.1f}%, current {current:.1f}%).",
    "{move_type} move demand is in line with past trends (historical avg {hist_avg:.1f}%, current {current:.1f}%).",
    "Expected {move_type} moves are consistent with historical data (historical avg {hist_avg:.1f}%, current {current:.1f}%)."
]
STRONGER_PHRASES = [
    "Demand for {move_type} moves is higher than historical trends (historical avg {hist_avg:.1f}%, current {current:.1f}%).",
    "{move_type} move demand exceeds past patterns (historical avg {hist_avg:.1f}%, current {current:.1f}%).",
    "Projected {move_type} moves show stronger demand than historical norms (historical avg {hist_avg:.1f}%, current {current:.1f}%)."
]
WEAKER_PHRASES = [
    "Demand for {move_type} moves is lower than historical trends (historical avg {hist_avg:.1f}%, current {current:.1f}%).",
    "{move_type} move demand is below past trends (historical avg {hist_avg:.1f}%, current {current:.1f}%).",
    "Expected {move_type} moves are weaker compared to historical data (historical avg {hist_avg:.1f}%, current {current:.1f}%)."
]
NO_MOVE_TYPE_PHRASE = "Forecast reflects total moves for the branch, with no move type specified."

# Summary comment phrases
SUMMARY_CONSISTENT_PHRASES = [
    "This period's {move_type} move demand in {branch} aligns with historical averages ({current_year:.1f}% in 2025 vs. {hist_avg:.1f}% historically).",
    "{move_type} moves in {branch} for this period are consistent with past trends ({current_year:.1f}% in 2025 vs. {hist_avg:.1f}% historically).",
    "The demand for {move_type} moves in {branch} this period matches historical patterns ({current_year:.1f}% in 2025 vs. {hist_avg:.1f}% historically)."
]
SUMMARY_STRONGER_PHRASES = [
    "This period's {move_type} move demand in {branch} is stronger than historical averages ({current_year:.1f}% in 2025 vs. {hist_avg:.1f}% historically).",
    "{move_type} moves in {branch} show higher demand this period compared to past years ({current_year:.1f}% in 2025 vs. {hist_avg:.1f}% historically).",
    "Demand for {move_type} moves in {branch} is elevated this period relative to historical trends ({current_year:.1f}% in 2025 vs. {hist_avg:.1f}% historically)."
]
SUMMARY_WEAKER_PHRASES = [
    "This period's {move_type} move demand in {branch} is lower than historical averages ({current_year:.1f}% in 2025 vs. {hist_avg:.1f}% historically).",
    "{move_type} moves in {branch} are below historical trends for this period ({current_year:.1f}% in 2025 vs. {hist_avg:.1f}% historically).",
    "Demand for {move_type} moves in {branch} is weaker this period compared to past years ({current_year:.1f}% in 2025 vs. {hist_avg:.1f}% historically)."
]
SUMMARY_NO_MOVE_TYPE = "This forecast reflects total moves for {branch} over the period, with no move type specified."

# In-memory model cache
model_cache = {}

# Load Prophet models at startup
def load_models(model_dir='prophet_models'):
    global model_cache
    os.makedirs(model_dir, exist_ok=True)
    for model_path in glob.glob(os.path.join(model_dir, '*.pkl')):
        branch = os.path.splitext(os.path.basename(model_path))[0].replace("prophet_model_", "")
        try:
            with open(model_path, 'rb') as f:
                model_cache[branch] = pickle.load(f)
            logger.info(f"Loaded model for {branch}")
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {str(e)}")
    if not model_cache:
        raise ValueError("No models loaded from prophet_models/")
    logger.info(f"Loaded {len(model_cache)} Prophet models")

# Load models at startup
load_models()

def forecast_move(input_date, input_branch, input_move_type=None, model_dir='prophet_models'):
    try:
        # Step 1: Convert input date to datetime
        try:
            input_date_dt = pd.to_datetime(input_date, format='%Y-%m-%d')
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD (e.g., '2025-06-11')")
        
        # Validate input date
        if input_date_dt > pd.to_datetime('2025-12-31'):
            raise ValueError("Date must be on or before December 31, 2025")
        
        # Step 2: Validate branch
        if input_branch not in model_cache:
            raise ValueError(f"Branch {input_branch} not found. Valid branches: {list(model_cache.keys())}")
        
        # Step 3: Validate move_type if provided
        if input_move_type is not None:
            move_types = fetch_data('SELECT DISTINCT move_type FROM historical_percentages')
            valid_move_types = move_types['move_type'].unique()
            if input_move_type not in valid_move_types:
                logger.warning(f"Invalid MoveType {input_move_type}. Using minimal percentage.")
                input_move_type = None  # Treat as no move type to use 100%
        
        # Step 4: Load Prophet model
        model = model_cache[input_branch]
        
        # Step 5: Generate forecast for the 15-day window
        today = pd.to_datetime(datetime.now().date())
        days_from_today = (input_date_dt - today).days

        if days_from_today <= 7:
            start_date = today
            end_date = today + timedelta(days=14)
        else:
            start_date = input_date_dt - timedelta(days=7)
            end_date = input_date_dt + timedelta(days=7)

        if end_date > pd.to_datetime('2025-12-31'):
            end_date = pd.to_datetime('2025-12-31')
        
        future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future_df)
        
        forecast = forecast[forecast['ds'] >= today]
        forecast = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Count'})
        forecast['Count'] = forecast['Count'].clip(lower=0).round().astype(int)
        
        # Step 6: Calculate historical percentage
        percentage = 100.0
        if input_move_type is not None:
            month = input_date_dt.month
            day = input_date_dt.day
            percentage = fetch_historical_percentages(input_branch, input_move_type, month, day)
        
        # Step 7: Prepare output
        predicted_summary = []
        total_predicted_moves = 0
        total_branch_forecast = 0
        
        for _, row in forecast.iterrows():
            forecast_date = row['Date']
            branch_forecast = row['Count']
            
            final_forecast = (percentage / 100) * branch_forecast
            final_forecast = int(round(final_forecast))
            
            total_predicted_moves += final_forecast
            total_branch_forecast += branch_forecast
            
            comment = ""
            if input_move_type is not None:
                month = forecast_date.month
                day = forecast_date.day
                hist_avg = fetch_historical_percentages(input_branch, input_move_type, month, day)
                
                implied_percentage = (final_forecast / branch_forecast * 100) if branch_forecast > 0 else 0
                percentage_diff = implied_percentage - hist_avg
                
                if abs(percentage_diff) <= 5:
                    comment = random.choice(CONSISTENT_PHRASES).format(
                        move_type=input_move_type, hist_avg=hist_avg, current=implied_percentage
                    )
                elif percentage_diff > 5:
                    comment = random.choice(STRONGER_PHRASES).format(
                        move_type=input_move_type, hist_avg=hist_avg, current=implied_percentage
                    )
                else:
                    comment = random.choice(WEAKER_PHRASES).format(
                        move_type=input_move_type, hist_avg=hist_avg, current=implied_percentage
                    )
            else:
                comment = NO_MOVE_TYPE_PHRASE
            
            predicted_summary.append({
                "date": forecast_date.strftime('%Y-%m-%d'),
                "predicted_moves": final_forecast,
                "comment": comment
            })
        
        average_daily_moves = int(round(total_predicted_moves / len(predicted_summary))) if predicted_summary else 0
        
        # Step 8: Calculate summary comment
        summary_comment = ""
        if input_move_type is not None:
            current_percentage = (total_predicted_moves / total_branch_forecast * 100) if total_branch_forecast > 0 else 0
            historical_percentages = []
            for forecast_date in pd.date_range(start=start_date, end=end_date, freq='D'):
                if forecast_date < today:
                    continue
                hist_avg = fetch_historical_percentages(input_branch, input_move_type, forecast_date.month, forecast_date.day)
                historical_percentages.append(hist_avg)
            
            historical_period_avg = sum(historical_percentages) / len(historical_percentages) if historical_percentages else percentage
            period_percentage_diff = current_percentage - historical_period_avg
            
            if abs(period_percentage_diff) <= 5:
                summary_comment = random.choice(SUMMARY_CONSISTENT_PHRASES).format(
                    move_type=input_move_type, branch=input_branch, current_year=current_percentage, hist_avg=historical_period_avg
                )
            elif period_percentage_diff > 5:
                summary_comment = random.choice(SUMMARY_STRONGER_PHRASES).format(
                    move_type=input_move_type, branch=input_branch, current_year=current_percentage, hist_avg=historical_period_avg
                )
            else:
                summary_comment = random.choice(SUMMARY_WEAKER_PHRASES).format(
                    move_type=input_move_type, branch=input_branch, current_year=current_percentage, hist_avg=historical_period_avg
                )
        else:
            summary_comment = SUMMARY_NO_MOVE_TYPE.format(branch=input_branch)
        
        result = {
            "branch": input_branch,
            "move_type": input_move_type,
            "forecast_window": {
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d')
            },
            "predicted_summary": predicted_summary,
            "total_predicted_moves": total_predicted_moves,
            "average_daily_moves": average_daily_moves,
            "summary_comment": summary_comment
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        raise ValueError(str(e))

# API endpoints
@app.get("/", response_model=dict)
@app.head("/")
async def root():
    return {"message": "Welcome to the Move Forecast API. Visit /docs for API documentation."}

@app.post("/forecast/")
async def forecast_endpoint(input_data: ForecastInput):
    try:
        logger.info(f"Received request: {input_data.dict()}")
        result = forecast_move(
            input_date=input_data.date,
            input_branch=input_data.branch,
            input_move_type=input_data.move_type
        )
        return result
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
