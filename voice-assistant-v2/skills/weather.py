# Weather via Open-Meteo with wttr.in fallback.

import json
import logging
import urllib.request
import urllib.parse

from config.settings import CONFIG
from skills.base import ToolRegistry

logger = logging.getLogger(__name__)


@ToolRegistry.register("get_weather", "Get current weather (requires internet)", mode="both")
def get_weather(location: str = None) -> dict:
    # Try Open-Meteo first, then fallback to wttr.in.

    def _try_open_meteo(loc: str) -> dict:
        if loc and loc != "auto":
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(loc)}&count=1&language=en&format=json"
            with urllib.request.urlopen(geo_url, timeout=8) as r:
                geo = json.loads(r.read())
            if not geo.get("results"):
                return None
            result = geo["results"][0]
            lat = result["latitude"]
            lon = result["longitude"]
            city = result.get("name", loc)
            country = result.get("country", "")
        else:
            # Use IP geolocation only when enabled.
            if not CONFIG.allow_ip_geolocation:
                return {
                    "display": (
                        "Please specify a location — e.g., 'weather in London'.\n"
                        "Auto-detection is disabled for privacy. "
                        "Enable it in config (allow_ip_geolocation = True)."
                    )
                }
            with urllib.request.urlopen("https://ipapi.co/json/", timeout=5) as r:
                ip_data = json.loads(r.read())
            lat = ip_data["latitude"]
            lon = ip_data["longitude"]
            city = ip_data.get("city", "Unknown")
            country = ip_data.get("country_name", "")

        wx_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
            f"weather_code,wind_speed_10m"
            f"&wind_speed_unit=kmh&timezone=auto"
        )
        with urllib.request.urlopen(wx_url, timeout=8) as r:
            wx = json.loads(r.read())

        cur = wx["current"]
        temp_c = round(cur["temperature_2m"])
        feels_c = round(cur["apparent_temperature"])
        temp_f = round(temp_c * 9/5 + 32)
        humidity = cur["relative_humidity_2m"]
        wind = round(cur["wind_speed_10m"])
        code = cur["weather_code"]

        # WMO weather code map.
        WMO = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Icy fog", 51: "Light drizzle", 53: "Drizzle",
            55: "Heavy drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
            71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Showers",
            81: "Rain showers", 82: "Heavy showers", 95: "Thunderstorm",
            96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
        }
        desc = WMO.get(code, f"Weather code {code}")

        display = (
            f"Weather in {city}, {country}:\n"
            f"  {desc}, {temp_c}\u00b0C / {temp_f}\u00b0F\n"
            f"  Feels like: {feels_c}\u00b0C\n"
            f"  Humidity: {humidity}%  |  Wind: {wind} km/h"
        )
        return {
            "location": f"{city}, {country}",
            "temp_c": temp_c, "temp_f": temp_f, "feels_c": feels_c,
            "description": desc, "humidity": humidity, "wind_kmph": wind,
            "display": display
        }

    def _try_wttr(loc: str) -> dict:
        # Fallback weather source.
        loc_param = loc if loc and loc != "auto" else ""
        url = f"https://wttr.in/{urllib.parse.quote(loc_param)}?format=j1"
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read())
        cur = data["current_condition"][0]
        area = data["nearest_area"][0]
        city = area["areaName"][0]["value"]
        country = area["country"][0]["value"]
        temp_c = int(cur["temp_C"])
        temp_f = int(cur["temp_F"])
        feels_c = int(cur["FeelsLikeC"])
        desc = cur["weatherDesc"][0]["value"]
        humidity = cur["humidity"]
        wind = cur["windspeedKmph"]
        display = (
            f"Weather in {city}, {country}:\n"
            f"  {desc}, {temp_c}\u00b0C / {temp_f}\u00b0F\n"
            f"  Feels like: {feels_c}\u00b0C\n"
            f"  Humidity: {humidity}%  |  Wind: {wind} km/h"
        )
        return {"location": f"{city},{country}", "temp_c": temp_c, "temp_f": temp_f,
                "description": desc, "humidity": humidity, "display": display}

    loc = (location or "").strip()
    errors = []

    try:
        return _try_open_meteo(loc if loc else "auto")
    except Exception as e:
        errors.append(f"Open-Meteo: {e}")

    try:
        return _try_wttr(loc)
    except Exception as e:
        errors.append(f"wttr.in: {e}")

    logger.warning("All weather APIs failed: %s", errors)
    return {
        "error": str(errors),
        "display": f"Couldn't fetch weather for {'''+loc+''' if loc else 'your location'}. Error: {errors[0]}"
    }
