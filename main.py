import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": -5.0,
	"longitude": -82.0,
	"start_date": "1980-01-01",
	"end_date": "2026-10-08",
	"daily": ["precipitation_sum", "rain_sum", "snowfall_sum", "temperature_2m_mean", "surface_pressure_mean"],
	"timezone": "America/Chicago",
}
responses = openmeteo.weather_api(url, params = params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process daily data. The order of variables needs to be the same as requested.
daily = response.Daily()
daily_precipitation_sum = daily.Variables(0).ValuesAsNumpy()
daily_rain_sum = daily.Variables(1).ValuesAsNumpy()
daily_snowfall_sum = daily.Variables(2).ValuesAsNumpy()
daily_temperature_2m_mean = daily.Variables(3).ValuesAsNumpy()
daily_surface_pressure_mean = daily.Variables(4).ValuesAsNumpy()

daily_data = {
	"time": pd.date_range(
		start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
		end =  pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = daily.Interval()),
		inclusive = "left"
	).tz_convert(response.Timezone().decode())
}

daily_data["precipitation_sum"] = daily_precipitation_sum
daily_data["rain_sum"] = daily_rain_sum
daily_data["snowfall_sum"] = daily_snowfall_sum
daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
daily_data["surface_pressure_mean"] = daily_surface_pressure_mean

daily_dataframe = pd.DataFrame(data = daily_data)

daily_dataframe['time'] = daily_dataframe['time'].dt.to_period('D').dt.to_timestamp()
daily_dataframe['monthyear'] = daily_dataframe['time'].dt.to_period('M').dt.to_timestamp()
monthly_dataframe = daily_dataframe[['precipitation_sum','temperature_2m_mean','surface_pressure_mean','monthyear']].groupby('monthyear').mean()
monthly_dataframe['lat'] = response.Latitude()
monthly_dataframe['long'] = response.Longitude()
daily_dataframe.set_index('monthyear', inplace=True)

## 1.2. Captura de datos - Dataset 2 (ONI data de NOAA.gov)
df_sst = pd.read_csv("https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii", sep='\s+',
                     names=["YR", "MON", "NINO12","NINO12ANOM", "NINO3","NINO3ANOM", "NINO4","NINO4ANOM", "NINO34","NINO34ANOM"], header=0)
df_sst_final = df_sst[['YR','MON','NINO12',"NINO12ANOM","NINO34","NINO34ANOM"]]
df_sst_final['monthyear'] = df_sst_final['YR'].astype(str) + '-' + df_sst_final['MON'].astype(str)
df_sst_final = df_sst_final[df_sst_final['monthyear']>= "1980-01-01"]
df_sst_final['monthyear'] = pd.to_datetime(df_sst_final['monthyear'], format='%Y-%m')
df_sst_final.set_index('monthyear', inplace=True)
df_sst_final.drop(columns=['YR','MON'], inplace=True)

# 1.3 Captura de datos - ICEN
df_icen = pd.read_csv("http://met.igp.gob.pe/datos/ICEN.txt", sep=r"\s+", skiprows=5 ,header=None,names=["YR","MON","ICEN"])
df_icen['monthyear'] = df_icen['YR'].astype(str) + '-' + df_icen['MON'].astype(str)
df_icen = df_icen[df_icen['monthyear']>= "1980-01-01"]
df_icen['monthyear'] = pd.to_datetime(df_icen['monthyear'], format='%Y-%m')
df_icen.set_index('monthyear', inplace=True)
df_icen.drop(columns=['YR','MON'], inplace=True)

df_final = (
    monthly_dataframe
    .merge(df_sst_final, on="monthyear", how="outer")
    .merge(df_icen, on="monthyear", how="outer")
    )

# EDA

### PRCP
max_prcp = df_final['precipitation_sum'].quantile(0.999)
max_prcp
min_prcp = df_final['precipitation_sum'].quantile(0.001)
min_prcp
df2 = df_final[df_final['precipitation_sum'] > min_prcp]
prcp_avg = df2['precipitation_sum'].mean().round(2)
df_final.loc[df_final['precipitation_sum'].isna(), 'precipitation_sum'] = prcp_avg

### TEMP
max_Temp = df_final['temperature_2m_mean'].quantile(0.999)
max_Temp
min_Temp = df_final['temperature_2m_mean'].quantile(0.001)
min_Temp
df2 = df_final[df_final['temperature_2m_mean'] > min_Temp]
Temp_avg = df2['temperature_2m_mean'].mean().round(2)
df_final.loc[df_final['temperature_2m_mean'].isna(), 'temperature_2m_mean'] = Temp_avg

### PRES
max_pres = df_final['surface_pressure_mean'].quantile(0.999)
max_pres
min_pres = df_final['surface_pressure_mean'].quantile(0.001)
min_pres
df2 = df_final[df_final['surface_pressure_mean'] > min_pres]
pres_avg = df2['surface_pressure_mean'].mean().round(2)
df_final.loc[df_final['surface_pressure_mean'].isna(), 'surface_pressure_mean'] = pres_avg

currentdate = datetime.now()
currentdate = currentdate.strftime("%Y-%m-%d")
datecheck = datetime.strptime(currentdate, "%Y-%m-%d") - pd.DateOffset(months=4)
datecheck = datecheck.strftime("%Y-%m-%d")

critical_cols = [
    "temperature_2m_mean","precipitation_sum","surface_pressure_mean", "long","lat"
]
critical_cols_2 = ["NINO34ANOM","ICEN"]

df_recent = df_final[df_final.index >= datecheck]
df_recent_clean = df_recent.dropna(subset=critical_cols)
df_recent_clean = df_recent.dropna(subset=critical_cols_2)

df_finalcleaned = pd.concat([
    df_final[df_final.index < datecheck],   # histórico intacto
    df_recent_clean                         # reciente limpio
])

## salvar en .csv
whichfields = ['precipitation_sum','temperature_2m_mean','surface_pressure_mean','lat','long','NINO12','NINO12ANOM','NINO34','NINO34ANOM','ICEN']
# Define csv filename
csvfile = 'coords-5-82.csv'

df_finalcleaned.to_csv(csvfile)
print(f"Guardado: {csvfile}")
