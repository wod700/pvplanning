# -*- coding: utf-8 -*-
import os
import json
import math
import yaml
import requests
import datetime as dt
import argparse

import matplotlib.pyplot as plt

from google.oauth2 import service_account
from googleapiclient.discovery import build
from typing import Optional, Dict, Any, List, Tuple

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None


# ======================
# CONFIG LADEN
# ======================
CONFIG_PATH = os.environ.get("PVPLANNING_CONFIG", "pvplanning.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

TIMEZONE = cfg.get("timezone", "Europe/Amsterdam")
CALENDAR_ID = cfg.get("calendar_id", "primary")
SERVICE_ACCOUNT_FILE = cfg.get("service_account_file", "service_account.json")
PVPLANNING_CONFIG=/secrets/pvplanning.yaml


# Output dirs
OUTPUT_DIR = cfg.get("output_dir", "output")
ARCHIVE_DIR = os.path.join(OUTPUT_DIR, "archive")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)
BASE_DIR = os.environ.get("PVPLANNING_BASE", ".")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


PLANNING_JSON = os.path.join(OUTPUT_DIR, "planning.json")
STATE_JSON = os.path.join(OUTPUT_DIR, "state.json")

# Net / PV
MAX_NET_KW = float(cfg["net"]["max_veilig_kw"])
PV_KWP = (cfg["pv"]["panelen"] * cfg["pv"]["wp_per_paneel"] / 1000.0) * float(cfg["pv"].get("rendement_factor", 1.0))

# Grafiek
GRAFIEK_ACTIEF = bool(cfg.get("grafiek", {}).get("actief", True))
GRAFIEK_BESTAND = cfg.get("grafiek", {}).get("bestand", "energieplanning_morgen.png")

# Locatie & API keys
LAT = float(cfg.get("locatie", {}).get("lat", 52.3489))
LON = float(cfg.get("locatie", {}).get("lon", 5.3125))
OPENWEATHER_API_KEY = cfg.get("openweather", {}).get("api_key", "")

# Kosten
NET_TARIEF_EUR_PER_KWH = float(cfg.get("kosten", {}).get("net_tarief_eur_per_kwh", 0.30))
EXPORT_KOST_EUR_PER_KWH = float(cfg.get("kosten", {}).get("export_kost_eur_per_kwh", 0.00))

# Forecast tuning
CLOUD_SENS = float(cfg.get("forecast", {}).get("cloud_sens", 0.55))
DIFFUSE_FLOOR = float(cfg.get("forecast", {}).get("diffuse_floor", 0.30))

# Tijdmodel
SLOT_MINUTEN = int(cfg.get("slot_minuten", 30))
SLOTS_PER_DAG = int(24 * 60 / SLOT_MINUTEN)

# Apparaten uit YAML (leidend)
APPARATEN_CFG = cfg.get("apparaten", {}) or {}

# Jacuzzi filters
JACUZZI_FILTER_KW = float(cfg.get("jacuzzi", {}).get("filter_vermogen_kw", 0.4))

# Warmtepomp (optioneel)
WARMTEPOMP_CFG = cfg.get("warmtepomp", {}) or {}
WARMTEPOMP_ENABLED = bool(WARMTEPOMP_CFG.get("enabled", False))


# ======================
# TZ helpers
# ======================
def epoch_utc_to_local_hour(ts_utc: int, tz_name: str) -> float:
    """
    Converteer epoch (UTC seconds) naar lokale uurwaarde (float) in tz_name (CET/CEST).
    """
    ts_int = int(ts_utc)
    if ZoneInfo is None:
        dtu = dt.datetime.fromtimestamp(ts_int, tz=dt.timezone.utc)
        return float(dtu.hour) + float(dtu.minute) / 60.0

    tz = ZoneInfo(tz_name)
    dt_local = dt.datetime.fromtimestamp(ts_int, tz=dt.timezone.utc).astimezone(tz)
    return float(dt_local.hour) + float(dt_local.minute) / 60.0


# ======================
# EMAIL (optioneel)
# ======================
def send_email_with_graph(subject: str, body: str, graph_path: str) -> None:
    email_cfg = cfg.get("email", {}) or {}
    if not email_cfg.get("enabled", False):
        return

    import smtplib
    from email.message import EmailMessage

    app_password = email_cfg.get("app_password") or os.environ.get("PVPLANNING_EMAIL_APP_PASSWORD", "")
    if not app_password:
        print("Email enabled, maar app_password ontbreekt. Sla email over.")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = email_cfg["from"]
    msg["To"] = email_cfg["to"]
    msg.set_content(body)

    if email_cfg.get("attach_graph", True) and graph_path and os.path.exists(graph_path):
        with open(graph_path, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="image",
                subtype="png",
                filename=os.path.basename(graph_path),
            )

    with smtplib.SMTP_SSL(email_cfg["smtp_host"], int(email_cfg["smtp_port"])) as smtp:
        smtp.login(email_cfg["from"], app_password)
        smtp.send_message(msg)


# ======================
# Google Calendar
# ======================
def calendar_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/calendar"],
    )
    return build("calendar", "v3", credentials=creds)


def rfc3339_dag_utc(d: dt.date) -> str:
    return f"{d.isoformat()}T00:00:00Z"


def rfc3339_dt_utc(ts: dt.datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def verwijder_events(service, start_rfc3339: str, end_rfc3339: str) -> None:
    resp = (
        service.events()
        .list(
            calendarId=CALENDAR_ID,
            timeMin=start_rfc3339,
            timeMax=end_rfc3339,
            q="Energieadvies",
            singleEvents=True,
        )
        .execute()
    )
    for e in resp.get("items", []):
        service.events().delete(calendarId=CALENDAR_ID, eventId=e["id"]).execute()


def maak_event(service, naam: str, start_iso: str, end_iso: str, vermogen_kw: float, extra: str = "") -> None:
    event = {
        "summary": f"Energieadvies – {naam}",
        "description": (
            "Handmatig inschakelen\n"
            f"Adviesvermogen: {vermogen_kw:.1f} kW\n"
            f"{extra}".strip()
        ),
        "start": {"dateTime": start_iso, "timeZone": TIMEZONE},
        "end": {"dateTime": end_iso, "timeZone": TIMEZONE},
        "colorId": "5",
    }
    service.events().insert(calendarId=CALENDAR_ID, body=event).execute()


# ======================
# State (weekquota)
# ======================
def week_id(d: dt.date) -> str:
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def load_state(dag: dt.date) -> Dict[str, Any]:
    if not os.path.exists(STATE_JSON):
        return {"week": week_id(dag), "used": {}}
    with open(STATE_JSON, "r", encoding="utf-8") as f:
        state = json.load(f)
    if state.get("week") != week_id(dag):
        return {"week": week_id(dag), "used": {}}
    state.setdefault("used", {})
    return state


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ======================
# Helpers tijd
# ======================
def slot_to_datetime_local(dag: dt.date, slot_index: int) -> dt.datetime:
    return dt.datetime.combine(dag, dt.time(0, 0)) + dt.timedelta(minutes=slot_index * SLOT_MINUTEN)


def slot_to_time_str(slot_index: int) -> str:
    total_minutes = (slot_index * SLOT_MINUTEN) % (24 * 60)
    hh = total_minutes // 60
    mm = total_minutes % 60
    return f"{hh:02d}:{mm:02d}"


# ======================
# Weer (OpenWeather)
# ======================
def openweather_onecall() -> Optional[Dict[str, Any]]:
    if not OPENWEATHER_API_KEY:
        return None
    url = (
        "https://api.openweathermap.org/data/3.0/onecall"
        f"?lat={LAT}&lon={LON}&exclude=minutely,alerts&units=metric&appid={OPENWEATHER_API_KEY}"
    )
    resp = requests.get(url, timeout=25)
    resp.raise_for_status()
    return resp.json()


def openweather_hourly_weather_for_date(target_date: dt.date) -> Dict[str, Any]:
    """
    Runtime structuur (voor planner):
      - temp_c: dict {0..23: float}
      - wind_bft: dict {0..23: float}
      - rain_mmph: dict {0..23: float}
      - clouds_frac: dict {0..23: float} (0..1)
      - sunrise_local: Optional[int] epoch UTC
      - sunset_local: Optional[int] epoch UTC
      - sunrise_hour: Optional[float] lokale uren (CET/CEST)
      - sunset_hour: Optional[float] lokale uren (CET/CEST)
    """
    if not OPENWEATHER_API_KEY:
        return {}

    url = (
        "https://api.openweathermap.org/data/3.0/onecall"
        f"?lat={LAT}&lon={LON}&exclude=minutely,alerts&units=metric&appid={OPENWEATHER_API_KEY}"
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    tz_offset = int(data.get("timezone_offset", 0))

    sunrise_ts = None
    sunset_ts = None
    for dday in data.get("daily", [])[:8]:
        ts = dday.get("dt")
        if ts is None:
            continue
        local_day = dt.datetime.fromtimestamp(int(ts) + tz_offset, tz=dt.timezone.utc).date()
        if local_day == target_date:
            sr = dday.get("sunrise")
            ss = dday.get("sunset")
            sunrise_ts = int(sr) if sr is not None else None
            sunset_ts = int(ss) if ss is not None else None
            break

    sunrise_hour: Optional[float] = None
    sunset_hour: Optional[float] = None
    if sunrise_ts is not None:
        sunrise_hour = epoch_utc_to_local_hour(int(sunrise_ts), TIMEZONE)
    if sunset_ts is not None:
        sunset_hour = epoch_utc_to_local_hour(int(sunset_ts), TIMEZONE)

    temp: Dict[int, float] = {h: 0.0 for h in range(24)}
    wind_bft: Dict[int, float] = {h: 0.0 for h in range(24)}
    rain: Dict[int, float] = {h: 0.0 for h in range(24)}
    clouds: Dict[int, float] = {h: 0.0 for h in range(24)}

    def ms_to_bft(ms: float) -> int:
        """
        Beaufort 0..12 op basis van m/s (WMO/KNMI grenzen).
        """
        v = float(ms)
        # grenzen: ondergrens per bft; we bepalen de klasse waarin v valt
        # 0: <0.5
        # 1: 0.5–1.5
        # 2: 1.6–3.3
        # 3: 3.4–5.4
        # 4: 5.5–7.9
        # 5: 8.0–10.7
        # 6: 10.8–13.8
        # 7: 13.9–17.1
        # 8: 17.2–20.7
        # 9: 20.8–24.4
        # 10: 24.5–28.4
        # 11: 28.5–32.6
        # 12: >=32.7
        bounds = [0.5, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]
        for bft, upper in enumerate(bounds):
            if v < upper:
                return bft
        return 12


    for hour in data.get("hourly", [])[:48]:
        ts = hour.get("dt")
        if ts is None:
            continue

        local_dt = dt.datetime.fromtimestamp(int(ts) + tz_offset, tz=dt.timezone.utc)
        if local_dt.date() != target_date:
            continue

        h = int(local_dt.hour)
        temp[h] = float(hour.get("temp", temp[h]))

        wind_ms = float(hour.get("wind_speed", 0.0))
        wind_bft[h] = max(wind_bft[h], ms_to_bft(wind_ms))

        r = hour.get("rain", {})
        if isinstance(r, dict):
            rain[h] = max(rain[h], float(r.get("1h", 0.0)))

        c_pct = hour.get("clouds", 0)
        try:
            c_frac = max(0.0, min(1.0, float(c_pct) / 100.0))
        except Exception:
            c_frac = 0.0
        clouds[h] = max(clouds[h], c_frac)

    return {
        "temp_c": temp,
        "wind_bft": wind_bft,
        "rain_mmph": rain,
        "clouds_frac": clouds,
        "sunrise_local": sunrise_ts,
        "sunset_local": sunset_ts,
        "sunrise_hour": sunrise_hour,
        "sunset_hour": sunset_hour,
    }


# ======================
# Zonnehoogte (simple astronomy)
# ======================
def solar_altitude_deg(date_local: dt.date, hour_float: float, lat_deg: float, lon_deg: float) -> float:
    hh = int(hour_float)
    mm = int(round((hour_float - hh) * 60))
    t = dt.datetime(date_local.year, date_local.month, date_local.day, hh, mm, 0)

    n = t.timetuple().tm_yday
    frac_hour = t.hour + t.minute / 60.0

    gamma = 2.0 * math.pi / 365.0 * (n - 1 + (frac_hour - 12.0) / 24.0)
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )

    tz_hours = 1.0  # curve-shape; niet exact
    time_offset = eqtime + 4.0 * lon_deg - 60.0 * tz_hours

    true_solar_minutes = (t.hour * 60.0 + t.minute + time_offset) % 1440.0
    hour_angle_deg = (true_solar_minutes / 4.0) - 180.0
    hour_angle = math.radians(hour_angle_deg)

    lat = math.radians(lat_deg)
    cos_zenith = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(hour_angle)
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.acos(cos_zenith)

    altitude = 90.0 - math.degrees(zenith)
    return altitude


def solar_altitude_curve_slots(target_date: dt.date) -> List[float]:
    out: List[float] = []
    for s in range(SLOTS_PER_DAG):
        hour = (s * SLOT_MINUTEN) / 60.0
        out.append(solar_altitude_deg(target_date, hour, LAT, LON))
    return out


# ======================
# PV forecast (PVGIS + clouds)
# ======================
def pvgis_hourly_kw_for_date(target_date: dt.date) -> Dict[int, float]:
    p = cfg.get("pvgis", {}) or {}
    base = f"https://re.jrc.ec.europa.eu/api/{p.get('api_version', 'v5_3')}/seriescalc"

    def fetch_for_year(ref_year: int) -> Optional[Dict[int, float]]:
        params = {
            "lat": LAT,
            "lon": LON,
            "outputformat": "json",
            "pvcalculation": 1,
            "peakpower": PV_KWP,
            "loss": float(p.get("loss_percent", 14)),
            "angle": float(p.get("angle", 30)),
            "aspect": float(p.get("aspect", 0)),
            "pvtechchoice": p.get("pvtechchoice", "crystSi"),
            "mountingplace": p.get("mountingplace", "free"),
            "raddatabase": p.get("raddatabase", "PVGIS-SARAH3"),
            "startyear": int(ref_year),
            "endyear": int(ref_year),
            "usehorizon": 1,
        }

        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        hourly = data.get("outputs", {}).get("hourly", [])
        if not hourly:
            return None

        out: Dict[int, float] = {}
        for row in hourly:
            t = row.get("time")  # "YYYYMMDD:HHMM"
            if not t or len(t) < 13:
                continue
            mm = int(t[4:6])
            dd = int(t[6:8])
            hh = int(t[9:11])
            if mm == target_date.month and dd == target_date.day:
                out[hh] = float(row.get("P", 0.0)) / 1000.0  # W -> kW

        if not out:
            return None

        for h in range(24):
            out.setdefault(h, 0.0)
        return out

    candidates: List[int] = []
    if p.get("ref_year") is not None:
        try:
            candidates.append(int(p.get("ref_year")))
        except Exception:
            pass

    this_year = dt.date.today().year
    for y in [this_year - 1, this_year - 2, 2020]:
        if y not in candidates:
            candidates.append(y)

    last_exc: Optional[Exception] = None
    for y in candidates:
        try:
            result = fetch_for_year(y)
            if result is not None:
                return result
        except Exception as e:
            last_exc = e
            continue

    if last_exc:
        print(f"PVGIS fetch mislukt voor ref_years={candidates}: {last_exc}")
    return {h: 0.0 for h in range(24)}


def clear_sky_fallback_slots(target_date: dt.date) -> List[float]:
    alt = solar_altitude_curve_slots(target_date)
    slots: List[float] = [0.0] * SLOTS_PER_DAG
    peak = 0.85 * PV_KWP
    for i, a in enumerate(alt):
        if a <= 0:
            slots[i] = 0.0
        else:
            frac = math.sin(math.radians(min(90.0, a)))
            slots[i] = peak * (frac ** 1.3)
    return slots


def pv_curve_slots(target_date: dt.date, weather: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    pvgis_hour = pvgis_hourly_kw_for_date(target_date)
    clouds_hour = weather.get("clouds_frac", {h: 0.0 for h in range(24)})

    pv_slots: List[float] = [0.0] * SLOTS_PER_DAG
    cloud_factor_slots: List[float] = [1.0] * SLOTS_PER_DAG

    for h in range(24):
        c = float(clouds_hour.get(h, 0.0) or 0.0)

        factor_cloud = 1.0 - CLOUD_SENS * (c ** 1.6)
        factor = max(DIFFUSE_FLOOR, factor_cloud)

        pv_kw = float(pvgis_hour.get(h, 0.0)) * factor
        if pv_kw < 0.05:
            pv_kw = 0.0

        pv_slots[h * 2] = pv_kw
        pv_slots[h * 2 + 1] = pv_kw
        cloud_factor_slots[h * 2] = factor
        cloud_factor_slots[h * 2 + 1] = factor

    if max(pv_slots) < 0.10:
        fb = clear_sky_fallback_slots(target_date)
        pv_slots = [fb[i] * cloud_factor_slots[i] for i in range(SLOTS_PER_DAG)]

    return pv_slots, cloud_factor_slots


# ======================
# Kostenmodel
# ======================
def slot_cost_eur(load_kw: float, pv_kw: float) -> float:
    net_import_kw = max(0.0, load_kw - pv_kw)
    export_kw = max(0.0, pv_kw - load_kw)

    slot_hours = SLOT_MINUTEN / 60.0
    cost = net_import_kw * slot_hours * NET_TARIEF_EUR_PER_KWH
    cost += export_kw * slot_hours * EXPORT_KOST_EUR_PER_KWH
    return cost


# ======================
# Apparaten normaliseren uit YAML
# ======================
def apparaten_from_yaml() -> List[Dict[str, Any]]:
    apparaten: List[Dict[str, Any]] = []

    for key, a in (APPARATEN_CFG or {}).items():
        if key == "jacuzzi_temp":
            continue

        apparaten.append({
            "key": key,
            "naam": a.get("naam", key),
            "vermogen_kw": float(a.get("vermogen_kw", 0.0)),
            "slots": int(a.get("slots", 0)),
            "prio": int(a.get("prio", 99)),
            "max_per_week": int(a.get("max_per_week", 0)),
            "window_start_slot": int(a.get("window_start_slot", int((9 * 60) / SLOT_MINUTEN))),
            "window_end_slot": int(a.get("window_end_slot", int((16 * 60) / SLOT_MINUTEN))),
        })

    jac = APPARATEN_CFG.get("jacuzzi_temp")
    if isinstance(jac, dict):
        apparaten.append({
            "key": "jacuzzi_temp",
            "naam": jac.get("naam", "Jacuzzi Verwarmen"),
            "vermogen_kw": float(jac.get("vermogen_kw", 0.0)),
            "slots": int(jac.get("slots", 0)),
            "prio": int(jac.get("prio", 99)),
            "max_per_week": int(jac.get("max_per_week", 99)),
            "window_start_slot": int(jac.get("window_start_slot", int((9 * 60) / SLOT_MINUTEN))),
            "window_end_slot": int(jac.get("window_end_slot", int((16 * 60) / SLOT_MINUTEN))),
        })

    return sorted(apparaten, key=lambda x: x["prio"])


def jacuzzi_filter_blocks_from_yaml() -> List[Tuple[int, int]]:
    jac = APPARATEN_CFG.get("jacuzzi_temp", {}) or {}
    blocks: List[Tuple[int, int]] = []
    for b in (jac.get("filter_blokken") or []):
        try:
            blocks.append((int(b["start"]), int(b["duur"])))
        except Exception:
            pass

    if not blocks:
        blocks = [
            (int((9 * 60) / SLOT_MINUTEN), int((7 * 60) / SLOT_MINUTEN)),
            (int((21 * 60) / SLOT_MINUTEN), int((7 * 60) / SLOT_MINUTEN)),
        ]
    return blocks


# ======================
# Warmtepomp (optioneel)
# ======================
def warmtepomp_slots(weather: Dict[str, Any]) -> List[float]:
    slots: List[float] = [0.0] * SLOTS_PER_DAG
    if not WARMTEPOMP_ENABLED:
        return slots

    mode = (WARMTEPOMP_CFG.get("mode") or "baseload").lower()
    base_kw = float(WARMTEPOMP_CFG.get("baseload_kw", 0.0))

    if mode == "baseload":
        for i in range(SLOTS_PER_DAG):
            slots[i] = base_kw
        return slots

    setpoint = float(WARMTEPOMP_CFG.get("setpoint_c", 20.0))
    slope = float(WARMTEPOMP_CFG.get("slope_kw_per_c", 0.3))
    min_kw = float(WARMTEPOMP_CFG.get("min_kw", 0.0))
    max_kw = float(WARMTEPOMP_CFG.get("max_kw", 5.0))

    temp_c = weather.get("temp_c", {h: None for h in range(24)})
    for h in range(24):
        t = temp_c.get(h)
        if t is None:
            kw = base_kw
        else:
            kw = max(0.0, (setpoint - float(t)) * slope)
            kw = min(max_kw, max(min_kw, kw))
        slots[h * 2] = kw
        slots[h * 2 + 1] = kw

    return slots


# ======================
# Planner
# ======================
def apply_fixed_loads(used_slots: List[float], fixed_tasks_out: List[Dict[str, Any]]) -> List[float]:
    for start, length in jacuzzi_filter_blocks_from_yaml():
        for s in range(start, min(SLOTS_PER_DAG, start + length)):
            used_slots[s] += JACUZZI_FILTER_KW

        fixed_tasks_out.append({
            "name": "Jacuzzi filters (vast blok)",
            "key": "jacuzzi_filter",
            "start_slot": int(start),
            "slots": int(length),
            "power_kw": float(JACUZZI_FILTER_KW),
            "cost_eur": None,
            "fixed": True,
        })

    return used_slots


def find_best_window_start(
    pv_slots: List[float],
    used_slots: List[float],
    vermogen_kw: float,
    slots_needed: int,
    start_slot: int,
    end_slot: int,
    min_start_slot: int
) -> Tuple[Optional[int], Optional[float]]:
    best: Optional[Tuple[float, int]] = None  # (cost, start)
    start_begin = max(start_slot, min_start_slot)

    for start in range(start_begin, end_slot - slots_needed + 1):
        ok = True
        cost = 0.0
        for s in range(start, start + slots_needed):
            load = used_slots[s] + vermogen_kw
            if load > MAX_NET_KW:
                ok = False
                break
            cost += slot_cost_eur(load, pv_slots[s])
        if ok:
            if best is None or cost < best[0]:
                best = (cost, start)

    if best is None:
        return None, None
    return int(best[1]), float(best[0])


def plan_day(
    dag: dt.date,
    pv_slots: List[float],
    weather: Dict[str, Any],
    state: Dict[str, Any],
    min_start_slot: int
) -> Tuple[List[Dict[str, Any]], List[float], Dict[str, Any]]:
    used_slots: List[float] = [0.0] * SLOTS_PER_DAG

    planning: List[Dict[str, Any]] = []
    fixed_tasks: List[Dict[str, Any]] = []

    used_slots = apply_fixed_loads(used_slots, fixed_tasks)

    hp_slots = warmtepomp_slots(weather)
    used_slots = [used_slots[i] + hp_slots[i] for i in range(SLOTS_PER_DAG)]

    apparaten = apparaten_from_yaml()

    for app in apparaten:
        used_count = int(state["used"].get(app["key"], 0))
        if used_count >= int(app["max_per_week"]):
            continue

        start, cost = find_best_window_start(
            pv_slots=pv_slots,
            used_slots=used_slots,
            vermogen_kw=float(app["vermogen_kw"]),
            slots_needed=int(app["slots"]),
            start_slot=int(app["window_start_slot"]),
            end_slot=int(app["window_end_slot"]),
            min_start_slot=min_start_slot,
        )
        if start is None:
            continue

        for s in range(start, start + int(app["slots"])):
            used_slots[s] += float(app["vermogen_kw"])

        state["used"][app["key"]] = used_count + 1

        planning.append({
            "name": app["naam"],
            "key": app["key"],
            "start_slot": int(start),
            "slots": int(app["slots"]),
            "power_kw": float(app["vermogen_kw"]),
            "cost_eur": float(round(cost, 2)) if cost is not None else None,
            "fixed": False,
        })

    return (fixed_tasks + planning), used_slots, state


# ======================
# Output
# ======================
def save_planning_json(
    dag: dt.date,
    pv_slots: List[float],
    used_slots: List[float],
    planning: List[Dict[str, Any]],
    min_start_slot: int,
    weather: Dict[str, Any],
    solar_alt_slots: List[float],
    cloud_factor_slots: List[float],
    args_tag: str
) -> str:
    payload: Dict[str, Any] = {
        "timezone": TIMEZONE,
        "date": dag.isoformat(),
        "slot_minutes": SLOT_MINUTEN,
        "max_net_kw": float(MAX_NET_KW),
        "min_start_slot": int(min_start_slot),

        "pv_slots_kw": [float(x) for x in pv_slots],
        "used_slots_kw": [float(x) for x in used_slots],

        "planning": planning,

        # Dashboard-vriendelijke structuur: lists met 24 waarden
        "weather": {
            "temp_c_hourly": [weather.get("temp_c", {}).get(h) for h in range(24)],
            "wind_bft_hourly": [weather.get("wind_bft", {}).get(h) for h in range(24)],
            "rain_mmph_hourly": [float(weather.get("rain_mmph", {}).get(h, 0.0)) for h in range(24)],
            "clouds_frac_hourly": [float(weather.get("clouds_frac", {}).get(h, 0.0)) for h in range(24)],
            "sunrise_hour": weather.get("sunrise_hour"),
            "sunset_hour": weather.get("sunset_hour"),
            "sunrise_local_ts": weather.get("sunrise_local"),
            "sunset_local_ts": weather.get("sunset_local"),
        },

        "solar_altitude_deg_slots": [float(x) for x in solar_alt_slots],
        "cloud_factor_slots": [float(x) for x in cloud_factor_slots],

        "warmtepomp": {
            "enabled": bool(WARMTEPOMP_ENABLED),
            "mode": (WARMTEPOMP_CFG.get("mode") or "baseload"),
        },
    }

    with open(PLANNING_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = (args_tag or "run").replace(" ", "_")
    archive_path = os.path.join(ARCHIVE_DIR, "planning_{}_{}_{}.json".format(dag.isoformat(), stamp, safe_tag))
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return archive_path


# ======================
# Grafieken
# ======================
def maak_grafiek_energy(pv_slots: List[float], used_slots: List[float], bestand: str) -> None:
    n = min(len(pv_slots), len(used_slots))
    x = [i * (SLOT_MINUTEN / 60.0) for i in range(n)]

    plt.figure(figsize=(12, 5))
    plt.plot(x, pv_slots[:n], label="PV-opwek (kW)")
    plt.plot(x, used_slots[:n], label="Gepland verbruik (kW)")
    plt.axhline(y=MAX_NET_KW, linestyle="--", label="Netlimiet {:.0f} kW".format(MAX_NET_KW))

    plt.xlabel("Uur van de dag")
    plt.ylabel("Vermogen (kW)")
    plt.title("Energieplanning ({}-min slots)".format(SLOT_MINUTEN))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(bestand)
    plt.close()


def _weather_hour_dict_or_list(weather: Dict[str, Any], key: str) -> List[Optional[float]]:
    """
    Maakt plotting robuust:
      - runtime: weather[key] is dict 0..23
      - saved json: weather[key+"_hourly"] is list 24
    """
    if not isinstance(weather, dict):
        return [None] * 24

    if key in weather and isinstance(weather.get(key), dict):
        d = weather.get(key) or {}
        return [d.get(h) for h in range(24)]

    hourly_key = f"{key}_hourly"
    if hourly_key in weather and isinstance(weather.get(hourly_key), list):
        arr = weather.get(hourly_key) or []
        arr = (arr + [None] * 24)[:24]
        return arr

    if key in weather and isinstance(weather.get(key), list):
        arr = weather.get(key) or []
        arr = (arr + [None] * 24)[:24]
        return arr

    return [None] * 24


def maak_grafiek_weer(weather: Dict[str, Any], dag: dt.date, bestand: str,
                      commute_start_h: int = 7, commute_end_h: int = 10) -> None:
    # werkt zowel met runtime weather (dicts) als saved weather (lists)
    temp_list = _weather_hour_dict_or_list(weather, "temp_c")
    wind_list = _weather_hour_dict_or_list(weather, "wind_bft")
    rain_list = _weather_hour_dict_or_list(weather, "rain_mmph")

    x = list(range(24))
    fig = plt.figure(figsize=(12, 5))
    ax = plt.gca()

    y_temp = [(float(v) if v is not None else float("nan")) for v in temp_list]
    ax.plot(x, y_temp, label="Temperatuur (°C)")

    ax2 = ax.twinx()
    y_wind = [(float(v) if v is not None else float("nan")) for v in wind_list]
    ax2.plot(x, y_wind, linestyle="--", label="Wind (Bft)")

    ax2.bar(x, [float(v or 0.0) for v in rain_list], alpha=0.25, label="Regen (mm/u)")

    ax.axvspan(commute_start_h, commute_end_h, alpha=0.10)

    # Zon op/onder markers (naar lokale tijd CET/CEST)
    sr = weather.get("sunrise_local")
    ss = weather.get("sunset_local")
    if sr:
        try:
            sr_hour = epoch_utc_to_local_hour(int(sr), TIMEZONE)
            ax.axvline(sr_hour, linestyle=":", linewidth=2)
            ax.text(sr_hour, ax.get_ylim()[1], "Zon op", rotation=90, va="top")
        except Exception:
            pass

    if ss:
        try:
            ss_hour = epoch_utc_to_local_hour(int(ss), TIMEZONE)
            ax.axvline(ss_hour, linestyle=":", linewidth=2)
            ax.text(ss_hour, ax.get_ylim()[1], "Zon onder", rotation=90, va="top")
        except Exception:
            pass

    ax.set_title("Weer (temperatuur, wind, regen) + zon op/onder + commute window")
    ax.set_xlabel("Uur")
    ax.set_ylabel("°C")
    ax2.set_ylabel("Bft / mm/u")
    ax.grid(True)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")

    plt.tight_layout()
    plt.savefig(bestand)
    plt.close(fig)


# ======================
# MAIN
# ======================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="Herplan vanaf huidige tijd (rest van vandaag)")
    parser.add_argument(
        "--day",
        choices=["today", "tomorrow"],
        default="tomorrow",
        help="Plan voor today of tomorrow (default: tomorrow)",
    )
    args = parser.parse_args()

    dag_start = dt.date.today() if args.day == "today" else (dt.date.today() + dt.timedelta(days=1))
    weekday = dag_start.weekday()

    if weekday in (cfg.get("afwezig", {}) or {}).get("dagen", []):
        print("Geen planning: afwezig op deze dag")
        return

    min_start_slot = 0
    if args.update and args.day == "today":
        now = dt.datetime.now()
        current_slot = int((now.hour * 60 + now.minute) / SLOT_MINUTEN)
        min_start_slot = min(SLOTS_PER_DAG - 1, current_slot + 1)
        print("Update-modus: herplanning vanaf slot {} ({})".format(min_start_slot, slot_to_time_str(min_start_slot)))

    state = load_state(dag_start)

    weather = openweather_hourly_weather_for_date(dag_start)
    solar_alt_slots = solar_altitude_curve_slots(dag_start)
    pv_slots, cloud_factor_slots = pv_curve_slots(dag_start, weather)

    planning, used_slots, state = plan_day(
        dag=dag_start,
        pv_slots=pv_slots,
        weather=weather,
        state=state,
        min_start_slot=min_start_slot,
    )

    save_state(state)
    archive_path = save_planning_json(
        dag_start,
        pv_slots,
        used_slots,
        planning,
        min_start_slot,
        weather,
        solar_alt_slots,
        cloud_factor_slots,
        args_tag=("today_update" if args.day == "today" and args.update else args.day),
    )

    # Grafieken
    if GRAFIEK_ACTIEF:
        maak_grafiek_energy(pv_slots, used_slots, GRAFIEK_BESTAND)
        weer_png = os.path.join(OUTPUT_DIR, "weer_{}.png".format(dag_start.isoformat()))
        maak_grafiek_weer(weather, dag_start, weer_png)

    # Calendar
    service = calendar_service()
    verwijder_events(service, rfc3339_dag_utc(dag_start), rfc3339_dag_utc(dag_start + dt.timedelta(days=1)))

    for item in planning:
        if args.update and args.day == "today" and int(item.get("start_slot", 0)) < min_start_slot:
            continue

        start_dt = slot_to_datetime_local(dag_start, int(item["start_slot"]))
        end_dt = slot_to_datetime_local(dag_start, int(item["start_slot"]) + int(item["slots"]))

        extra = "PV-gestuurd (slot {}–{})".format(item["start_slot"], item["start_slot"] + item["slots"])
        if item.get("cost_eur") is not None:
            extra = "Geschatte kosten: €{:.2f}\n{}".format(float(item["cost_eur"]), extra)

        maak_event(
            service=service,
            naam=item["name"],
            start_iso=start_dt.isoformat(),
            end_iso=end_dt.isoformat(),
            vermogen_kw=float(item["power_kw"]),
            extra=extra,
        )

    print("Planning gemaakt voor {}. Taken: {}. Output: {}".format(dag_start, len(planning), PLANNING_JSON))
    for item in planning:
        if item.get("fixed"):
            continue
        if item.get("cost_eur") is not None:
            print("{}: kost €{:.2f} (start {})".format(item["name"], float(item["cost_eur"]), item["start_slot"]))

    # Email (optioneel)
    if cfg.get("email", {}).get("enabled", False):
        subject = "PV planning {}".format(dag_start.isoformat())
        body = "Nieuwe planning gemaakt. Taken: {}.\nArchive: {}".format(len(planning), os.path.basename(archive_path))
        send_email_with_graph(subject, body, GRAFIEK_BESTAND if GRAFIEK_ACTIEF else "")


if __name__ == "__main__":
    main()
