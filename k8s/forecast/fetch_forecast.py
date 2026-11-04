"""Fetch the next 24 complete UTC hours and publish one atomic forecast batch."""
import json
import os
from datetime import datetime, timedelta, timezone
import urllib.parse
import urllib.request

API_URL = os.getenv("API_URL", "http://wind-api").rstrip("/")
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


def select_hours(hourly, now):
    start = now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    # Unix timestamps avoid ambiguous or duplicated local hours at DST transitions.
    indexed = dict(zip(hourly["time"], zip(hourly["wind_speed_10m"], hourly["wind_direction_10m"])))
    result = []
    for i in range(24):
        target = start + timedelta(hours=i)
        weather = indexed.get(int(target.timestamp()))
        if weather is None or any(value is None for value in weather):
            raise ValueError(f"Missing weather for {target.isoformat()}; batch not published")
        result.append({"timestamp": target.isoformat(), "wind_speed": weather[0], "wind_direction": weather[1]})
    return result


def main():
    params = {"latitude": 58.98, "longitude": -2.96,
              "hourly": "wind_speed_10m,wind_direction_10m", "wind_speed_unit": "ms",
              "timezone": "UTC", "timeformat": "unixtime", "forecast_hours": 26}
    # Includes the current hour plus a spare hour if the request crosses an hour boundary.
    with urllib.request.urlopen(WEATHER_URL + "?" + urllib.parse.urlencode(params), timeout=30) as response:
        hourly = json.load(response)["hourly"]
    hours = select_hours(hourly, datetime.now(timezone.utc))
    request = urllib.request.Request(API_URL + "/predict/24h",
                                     data=json.dumps({"hours": hours}).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    for point in result["predictions"]:
        print(f"{point['timestamp']} -> {point['power_kw']} kW", flush=True)
    print(f"Published 24 hourly predictions; generated_at={result['generated_at']}", flush=True)


if __name__ == "__main__":
    main()
