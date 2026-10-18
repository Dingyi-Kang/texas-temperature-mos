"""Station <-> load-project city mapping (nearest major ASOS to the city point used
for the Open-Meteo forecasts in ~/ML_forecast)."""
CITY_STATION = {"houston": "HOU", "dallas": "DAL", "fortworth": "FTW", "sanantonio": "SAT", "austin": "AUS",
                "corpus": "CRP", "mcallen": "MFE", "midland": "MAF", "abilene": "ABI", "tyler": "TYR",
                "wichitafalls": "SPS"}
STATION_CITY = {v: k for k, v in CITY_STATION.items()}
