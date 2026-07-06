# Time and date tools.

import datetime

from skills.base import ToolRegistry


@ToolRegistry.register("get_time", "Get current time", mode="both")
def get_time() -> dict:
    now = datetime.datetime.now()
    return {
        "time_12h": now.strftime("%I:%M %p"),
        "time_24h": now.strftime("%H:%M:%S"),
        "display": f"It's {now.strftime('%I:%M %p')}",
    }


@ToolRegistry.register("get_date", "Get current date and day", mode="both")
def get_date() -> dict:
    now = datetime.datetime.now()
    return {
        "date": now.strftime("%B %d, %Y"),
        "day": now.strftime("%A"),
        "display": f"Today is {now.strftime('%A, %B %d, %Y')}",
        "iso": now.date().isoformat(),
    }
