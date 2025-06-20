import pandas as pd
import psycopg2
from prophet import Prophet
import os
import pickle
 
DATABASE_URL = os.environ.get("DATABASE_URL")
conn = psycopg2.connect(DATABASE_URL)
df_agg = pd.read_sql_query('SELECT "Date", "Branch", "Count" FROM forecasting_data', conn)
conn.close()
 
df_agg['Date'] = pd.to_datetime(df_agg['Date'])
df_agg = df_agg.groupby(['Date', 'Branch'])['Count'].sum().reset_index()
 
os.makedirs('prophet_models', exist_ok=True)
unique_branches = df_agg['Branch'].unique()
 
for branch in unique_branches:
    branch_data = df_agg[df_agg['Branch'] == branch][['Date', 'Count']].rename(columns={'Date': 'ds', 'Count': 'y'})
    train_data = branch_data[branch_data['ds'] <= '2023-12-31']
    if len(train_data) >= 2:
        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=False,  # Disabled to reduce computation
            changepoint_prior_scale=0.01,
            seasonality_prior_scale=15.0,
            seasonality_mode='multiplicative'
        )
        model.fit(train_data)
        with open(f'prophet_models/prophet_model_{branch}.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model for {branch}")
