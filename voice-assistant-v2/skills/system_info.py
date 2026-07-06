# System info via psutil.

from skills.base import ToolRegistry


@ToolRegistry.register("get_device_info", "Get system info: CPU, RAM, disk, battery", mode="both")
def get_device_info() -> dict:
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        info = {
            "cpu_percent": cpu,
            "ram_used_gb": round(ram.used / 1e9, 1),
            "ram_total_gb": round(ram.total / 1e9, 1),
            "ram_percent": ram.percent,
            "disk_used_gb": round(disk.used / 1e9, 1),
            "disk_total_gb": round(disk.total / 1e9, 1),
            "disk_percent": disk.percent,
        }

        try:
            battery = psutil.sensors_battery()
            if battery:
                info["battery_percent"] = round(battery.percent, 1)
                info["battery_plugged"] = battery.power_plugged
        except Exception:
            pass  # not all platforms support battery info

        lines = [
            f"CPU: {cpu}%",
            f"RAM: {info['ram_used_gb']}GB / {info['ram_total_gb']}GB ({ram.percent}%)",
            f"Disk: {info['disk_used_gb']}GB / {info['disk_total_gb']}GB ({disk.percent}%)",
        ]
        if "battery_percent" in info:
            status = "plugged in" if info["battery_plugged"] else "on battery"
            lines.append(f"Battery: {info['battery_percent']}% ({status})")

        info["display"] = "\n".join(lines)
        return info

    except ImportError:
        return {"display": "Install psutil for system info: pip install psutil", "error": "psutil not installed"}
