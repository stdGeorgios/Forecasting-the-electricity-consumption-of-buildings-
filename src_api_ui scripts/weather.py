# C:/IDEAL_Programming/src/weather.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
import requests

import pandas as pd

@dataclass
class CityLocation:
    name: str
    latitude: float
    longitude: float
    country: str | None = None

def geocode_city(city: str) -> CityLocation:
    # Open-Meteo Geocoding API
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": city, "count": 1, "language": "en", "format": "json"}, timeout=20)
    r.raise_for_status()
    data = r.json()
    if "results" not in data or not data["results"]:
        raise ValueError(f"Δεν βρέθηκε πόλη στο geocoding: {city}")
    x = data["results"][0]
    return CityLocation(
        name=x.get("name", city),
        latitude=float(x["latitude"]),
        longitude=float(x["longitude"]),
        country=x.get("country"),
    )

def fetch_hourly_temperature_24h(city: str, target_date: str) -> list[float]:
    """
    Επιστρέφει 24 ωριαίες θερμοκρασίες (°C) για την ημερομηνία target_date (YYYY-MM-DD).
    Προσπαθεί από Open-Meteo forecast endpoint.
    """
    loc = geocode_city(city)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "hourly": "temperature_2m",
        "timezone": "auto",
        "start_date": target_date,
        "end_date": target_date,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    times = data.get("hourly", {}).get("time", [])
    temps = data.get("hourly", {}).get("temperature_2m", [])

    if len(times) < 24 or len(temps) < 24:
        raise ValueError(f"Δεν επέστρεψε 24 ωριαίες τιμές για {city} στο {target_date}")

    # Κρατάμε τις πρώτες 24 (αν δώσει παραπάνω)
    return [float(x) for x in temps[:24]]

def build_profile_from_minmax(t_min: float, t_max: float) -> list[float]:
    """
    Δημιουργεί απλό 24ωρο προφίλ:
    min ~ 05:00, max ~ 15:00 (ρεαλιστικό “ημερήσιο” pattern).
    """
    import math
    # phase shift ώστε peak ~ 15:00
    vals = []
    for h in range(24):
        # sin in [0..pi] roughly for day shape
        # map hour to angle
        angle = (h - 5) / 24 * 2 * math.pi
        # normalize sin to [0..1]
        w = (math.sin(angle) + 1) / 2
        vals.append(t_min + (t_max - t_min) * w)
    return [round(v, 1) for v in vals]