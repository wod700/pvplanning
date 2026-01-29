# -*- coding: utf-8 -*-
import os
import json
import glob
import datetime as dt
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None


OUTPUT_DIR = "output"
PLANNING_JSON = os.path.join(OUTPUT_DIR, "planning.json")
ARCHIVE_DIR = os.path.join(OUTPUT_DIR, "archive")
BASE_DIR = os.environ.get("PVPLANNING_BASE", ".")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")



# ======================
# Helpers
# ======================
def slot_to_time_str(slot_index: int, slot_minutes: int) -> str:
    total_minutes = (slot_index * slot_minutes) % (24 * 60)
    hh = total_minutes // 60
    mm = total_minutes % 60
    return f"{hh:02d}:{mm:02d}"


def list_archives() -> List[str]:
    if not os.path.isdir(ARCHIVE_DIR):
        return []
    return sorted(glob.glob(os.path.join(ARCHIVE_DIR, "*.json")), reverse=True)


def load_planning(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _weather_get_hourly_list(weather: Dict[str, Any], key: str) -> List[Any]:
    """
    Ondersteunt:
      - planning.json: key+"_hourly" = lijst (24)
      - runtime dict: key = dict {0..23: ...}
      - legacy: key = lijst
    """
    if not isinstance(weather, dict):
        return [None] * 24

    hourly_key = f"{key}_hourly"
    if hourly_key in weather and isinstance(weather.get(hourly_key), list):
        arr = weather.get(hourly_key) or []
        return (arr + [None] * 24)[:24]

    v = weather.get(key)
    if isinstance(v, dict):
        return [v.get(h) for h in range(24)]

    if isinstance(v, list):
        return (v + [None] * 24)[:24]

    return [None] * 24


def _epoch_to_local_hour(ts_utc: Any, tz_name: str) -> Optional[float]:
    try:
        ts_int = int(ts_utc)
    except Exception:
        return None

    if ZoneInfo is None:
        dtu = dt.datetime.fromtimestamp(ts_int, tz=dt.timezone.utc)
        return float(dtu.hour) + float(dtu.minute) / 60.0

    try:
        tz = ZoneInfo(tz_name)
        dt_local = dt.datetime.fromtimestamp(ts_int, tz=dt.timezone.utc).astimezone(tz)
        return float(dt_local.hour) + float(dt_local.minute) / 60.0
    except Exception:
        return None


def _weather_get_sun_hour(weather: Dict[str, Any], which: str, tz_name: str) -> Optional[float]:
    """
    Ondersteunt:
      - sunrise_hour / sunset_hour direct (float)
      - sunrise_local_ts / sunset_local_ts epoch (int)
      - sunrise_local / sunset_local epoch (int)
    """
    key_hour = f"{which}_hour"
    if weather.get(key_hour) is not None:
        try:
            return float(weather.get(key_hour))
        except Exception:
            return None

    ts = weather.get(f"{which}_local_ts")
    if ts is None:
        ts = weather.get(f"{which}_local")
    if ts is None:
        return None

    return _epoch_to_local_hour(ts, tz_name)


def _slice_hours(start_h: float, end_h: float) -> List[int]:
    """
    Geeft uren (int) die overlappen met het venster [start_h, end_h].
    Geen math-import nodig.
    """
    a = int(start_h)
    b = int(end_h) if float(end_h).is_integer() else int(end_h) + 1
    return [h for h in range(24) if a <= h < b]


def _avg_finite(arr: np.ndarray) -> float:
    v = arr[np.isfinite(arr)]
    return float(np.mean(v)) if v.size > 0 else float("nan")


def _max_finite(arr: np.ndarray) -> float:
    v = arr[np.isfinite(arr)]
    return float(np.max(v)) if v.size > 0 else float("nan")


# ======================
# Business logic
# ======================
def compute_energy_totals(data: Dict[str, Any]) -> Dict[str, float]:
    slot_minutes = int(data.get("slot_minutes", 30))
    slot_hours = slot_minutes / 60.0

    pv = data.get("pv_slots_kw", []) or []
    used = data.get("used_slots_kw", []) or []
    n = min(len(pv), len(used))

    pv_kwh = used_kwh = import_kwh = export_kwh = 0.0
    for i in range(n):
        pv_kw = float(pv[i])
        used_kw = float(used[i])

        pv_kwh += pv_kw * slot_hours
        used_kwh += used_kw * slot_hours

        net_import_kw = max(0.0, used_kw - pv_kw)
        export_kw = max(0.0, pv_kw - used_kw)
        import_kwh += net_import_kw * slot_hours
        export_kwh += export_kw * slot_hours

    return {
        "pv_kwh": pv_kwh,
        "used_kwh": used_kwh,
        "import_kwh": import_kwh,
        "export_kwh": export_kwh,
    }


def make_tasks_df(data: Dict[str, Any]) -> pd.DataFrame:
    slot_minutes = int(data.get("slot_minutes", 30))
    slot_hours = slot_minutes / 60.0
    tasks = data.get("planning", []) or []

    rows: List[Dict[str, Any]] = []
    for t in tasks:
        start = int(t.get("start_slot", 0))
        slots = int(t.get("slots", 0))
        end = start + slots
        power_kw = float(t.get("power_kw", 0.0))

        duration_minutes = slots * slot_minutes
        rows.append({
            "Taak": t.get("name", ""),
            "Start": slot_to_time_str(start, slot_minutes),
            "Eind": slot_to_time_str(end, slot_minutes),
            "Duur": f"{duration_minutes // 60:d}u {duration_minutes % 60:02d}m",
            "Vermogen (kW)": round(power_kw, 2),
            "Energie (kWh)": round(power_kw * float(slots) * slot_hours, 2),
            "Kosten (€)": "n.v.t." if t.get("cost_eur") is None else f"{float(t.get('cost_eur', 0.0)):.2f}",
            "Vast": bool(t.get("fixed", False)),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Start", "Taak"]).reset_index(drop=True)
    return df


def group_tasks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = (
        df.groupby(["Taak", "Vast"], dropna=False, as_index=False)
        .agg({"Energie (kWh)": "sum", "Vermogen (kW)": "mean"})
    )
    g["Energie (kWh)"] = g["Energie (kWh)"].round(2)
    g["Vermogen (kW)"] = g["Vermogen (kW)"].round(2)
    return g.sort_values(["Vast", "Taak"]).reset_index(drop=True)


def make_overview_df(data: Dict[str, Any], show_kw: bool) -> pd.DataFrame:
    slot_minutes = int(data.get("slot_minutes", 30))
    slot_hours = slot_minutes / 60.0

    pv = data.get("pv_slots_kw", []) or []
    used = data.get("used_slots_kw", []) or []
    max_net = float(data.get("max_net_kw", 0.0))
    n = min(len(pv), len(used))

    rows: List[Dict[str, Any]] = []
    for i in range(n):
        pv_kw = float(pv[i])
        used_kw = float(used[i])
        net_import_kw = max(0.0, used_kw - pv_kw)
        export_kw = max(0.0, pv_kw - used_kw)

        row: Dict[str, Any] = {
            "Tijd": slot_to_time_str(i, slot_minutes),
            "PV (kWh/slot)": round(pv_kw * slot_hours, 3),
            "Verbruik (kWh/slot)": round(used_kw * slot_hours, 3),
            "Net import (kWh/slot)": round(net_import_kw * slot_hours, 3),
            "Export (kWh/slot)": round(export_kw * slot_hours, 3),
        }
        if show_kw:
            row.update({
                "PV (kW)": round(pv_kw, 2),
                "Verbruik (kW)": round(used_kw, 2),
                "Net import (kW)": round(net_import_kw, 2),
                "Export (kW)": round(export_kw, 2),
                "Netlimiet (kW)": round(max_net, 0),
            })
        rows.append(row)

    return pd.DataFrame(rows)


# ======================
# Plots
# ======================
def plot_energy(data: Dict[str, Any]) -> None:
    slot_minutes = int(data.get("slot_minutes", 30))
    pv = data.get("pv_slots_kw", []) or []
    used = data.get("used_slots_kw", []) or []
    max_net = float(data.get("max_net_kw", 0.0))
    n = min(len(pv), len(used))

    x = [i * (slot_minutes / 60.0) for i in range(n)]
    fig = plt.figure(figsize=(12, 5))
    plt.plot(x, pv[:n], label="PV-opwek (kW)")
    plt.plot(x, used[:n], label="Gepland verbruik (kW)")
    plt.axhline(y=max_net, linestyle="--", label=f"Netlimiet {max_net:.0f} kW")
    plt.xlabel("Uur van de dag")
    plt.ylabel("Vermogen (kW)")
    plt.title(f"Energieplanning ({slot_minutes}-min slots)")
    plt.grid(True)
    plt.legend()
    st.pyplot(fig)
    plt.close(fig)


def plot_weather_panel(
    weather: Dict[str, Any],
    timezone_name: str,
    commute_out_start: float = 7.0,
    commute_out_end: float = 10.0,
    commute_back_start: float = 16.0,
    commute_back_end: float = 19.0,
) -> None:
    temp_list = _weather_get_hourly_list(weather, "temp_c")

    # wind: expliciet zowel *_hourly als legacy ondersteunen
    wind_list = (
        weather.get("wind_bft_hourly")
        or weather.get("wind_bft")
        or _weather_get_hourly_list(weather, "wind_bft")
    )

    rain_list = _weather_get_hourly_list(weather, "rain_mmph")
    clouds_list = _weather_get_hourly_list(weather, "clouds_frac")

    def to_float_arr(arr: List[Any], default: float = np.nan) -> np.ndarray:
        out = []
        for v in (list(arr) + [None] * 24)[:24]:
            try:
                out.append(float(v) if v is not None else default)
            except Exception:
                out.append(default)
        return np.array(out, dtype=float)

    temp = to_float_arr(temp_list, default=np.nan)
    wind = to_float_arr(wind_list, default=np.nan)
    rain = to_float_arr(rain_list, default=0.0)
    clouds = to_float_arr(clouds_list, default=np.nan)

    sunrise_hour = _weather_get_sun_hour(weather, "sunrise", timezone_name)
    sunset_hour = _weather_get_sun_hour(weather, "sunset", timezone_name)

    hours = np.arange(24)
    fig, ax_temp = plt.subplots(figsize=(12, 5))

    ax_temp.plot(hours, temp, label="Temperatuur (°C)")
    ax_temp.set_xlabel("Uur")
    ax_temp.set_ylabel("°C")
    ax_temp.grid(True, alpha=0.25)
    ax_temp.set_xlim(0, 23)

    # Commute heen/terug shading
    ax_temp.axvspan(float(commute_out_start), float(commute_out_end), alpha=0.10)
    ax_temp.axvspan(float(commute_back_start), float(commute_back_end), alpha=0.06)

    # Labels (na y-lim fix komen ze netjes bovenin)
    # -> we zetten ze later nogmaals met de uiteindelijke y_top

    # Wind: tweede as links
    ax_wind = ax_temp.twinx()
    ax_wind.spines["right"].set_visible(False)
    ax_wind.spines["left"].set_position(("axes", -0.08))
    ax_wind.yaxis.set_label_position("left")
    ax_wind.yaxis.set_ticks_position("left")
    ax_wind.plot(hours, wind, linestyle="--", marker="o", label="Wind (Bft)")
    ax_wind.set_ylabel("Bft")
    ax_wind.set_ylim(0, 12)

    # Regen: rechter as
    ax_rain = ax_temp.twinx()
    ax_rain.bar(hours, rain, alpha=0.25, label="Regen (mm/u)")
    ax_rain.set_ylabel("mm/u")
    rmax = float(np.nanmax(rain)) if np.isfinite(rain).any() else 0.0
    ax_rain.set_ylim(0, max(1.0, rmax * 1.4))

    # Bewolking: extra rechter as
    ax_cloud = ax_temp.twinx()
    ax_cloud.spines["right"].set_position(("axes", 1.10))
    ax_cloud.plot(hours, clouds, linestyle=":", marker="o", label="Bewolking (0..1)")
    ax_cloud.set_ylim(0.0, 1.0)
    ax_cloud.set_ylabel("Bewolking (0..1)")

    # Temp y-lim netjes (voor labels/markers)
    temp_vals = temp[np.isfinite(temp)]
    if temp_vals.size > 0:
        tmin, tmax = float(np.min(temp_vals)), float(np.max(temp_vals))
        if abs(tmax - tmin) < 0.1:
            ax_temp.set_ylim(tmin - 2, tmax + 2)
        else:
            pad = max(1.0, 0.15 * (tmax - tmin))
            ax_temp.set_ylim(tmin - pad, tmax + pad)

    y_top = ax_temp.get_ylim()[1]

    # Commute labels
    ax_temp.text(float(commute_out_start) + 0.05, y_top, "Heen", rotation=90, va="top", ha="left")
    ax_temp.text(float(commute_back_start) + 0.05, y_top, "Terug", rotation=90, va="top", ha="left")

    # Zon op/onder
    if sunrise_hour is not None:
        ax_temp.axvline(sunrise_hour, linestyle=":", linewidth=2)
        ax_temp.text(sunrise_hour + 0.05, y_top, "Zon op", rotation=90, va="top", ha="left")
    if sunset_hour is not None:
        ax_temp.axvline(sunset_hour, linestyle=":", linewidth=2)
        ax_temp.text(sunset_hour + 0.05, y_top, "Zon onder", rotation=90, va="top", ha="left")

    ax_temp.set_title("Weer: temperatuur, wind (Bft links), regen, bewolking + zon + commute (heen/terug)")

    # Legend combineren
    h1, l1 = ax_temp.get_legend_handles_labels()
    h2, l2 = ax_wind.get_legend_handles_labels()
    h3, l3 = ax_rain.get_legend_handles_labels()
    h4, l4 = ax_cloud.get_legend_handles_labels()
    ax_temp.legend(h1 + h2 + h3 + h4, l1 + l2 + l3 + l4, loc="upper left")

    st.pyplot(fig)
    plt.close(fig)


# ======================
# Fietsadvies
# ======================
def bicycle_advice_for_window(
    weather: Dict[str, Any],
    timezone_name: str,
    start_h: float,
    end_h: float,
) -> Dict[str, Any]:
    temp_list = _weather_get_hourly_list(weather, "temp_c")

    # wind: zelfde fallback als plot, anders heb je kans op “lege” wind in advies
    wind_list = (
        weather.get("wind_bft_hourly")
        or weather.get("wind_bft")
        or _weather_get_hourly_list(weather, "wind_bft")
    )

    rain_list = _weather_get_hourly_list(weather, "rain_mmph")

    def to_float_arr(arr: List[Any], default: float) -> np.ndarray:
        out = []
        for v in (list(arr) + [None] * 24)[:24]:
            try:
                out.append(float(v) if v is not None else default)
            except Exception:
                out.append(default)
        return np.array(out, dtype=float)

    temp = to_float_arr(temp_list, default=np.nan)
    wind = to_float_arr(wind_list, default=np.nan)
    rain = to_float_arr(rain_list, default=0.0)

    hours = _slice_hours(start_h, end_h)
    idx = np.array(hours, dtype=int)

    t_avg = _avg_finite(temp[idx])
    w_max = _max_finite(wind[idx])
    r_sum = float(np.nansum(rain[idx]))

    score = 0.0

    # Regen: zwaarwegend
    if r_sum >= 4.0:
        score += 45
    elif r_sum >= 1.5:
        score += 30
    elif r_sum >= 0.5:
        score += 15

    # Wind: vooral impact vanaf 5 Bft
    if np.isfinite(w_max):
        if w_max >= 8:
            score += 35
        elif w_max >= 6:
            score += 25
        elif w_max >= 5:
            score += 15
        elif w_max >= 4:
            score += 7

    # Temperatuur: comfort / gladheid-risico (simpel)
    if np.isfinite(t_avg):
        if t_avg <= 0:
            score += 15
        elif t_avg <= 3:
            score += 10
        elif t_avg <= 7:
            score += 5

    score = max(0.0, min(100.0, score))

    if score >= 65:
        label = "Overweeg OV/auto"
    elif score >= 40:
        label = "Niet ideaal (kan, maar reken op gedoe)"
    elif score >= 15:
        label = "Fiets (regenpak / extra kleding)"
    else:
        label = "Fiets (prima)"

    return {
        "label": label,
        "score": round(score, 1),
        "details": {
            "temp_avg_c": None if not np.isfinite(t_avg) else round(t_avg, 1),
            "wind_max_bft": None if not np.isfinite(w_max) else int(round(w_max)),
            "rain_sum_mm_window": round(r_sum, 1),
            "hours": hours,
        },
    }


# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="PV Energieplanning", layout="wide")
st.title("PV Energieplanning (lokaal)")

archives = list_archives()
options = ["(latest) " + PLANNING_JSON] + archives

selected = st.selectbox("Kies planning", options, index=0, key="planning_select_local")
path = PLANNING_JSON if selected.startswith("(latest)") else selected

data = load_planning(path)
if not data:
    st.warning("Geen planning gevonden. Draai eerst pv_planning.py.")
    st.stop()

timezone_name = str(data.get("timezone", "Europe/Amsterdam"))

dag = data.get("date", "")
slot_minutes = int(data.get("slot_minutes", 30))
min_start_slot = int(data.get("min_start_slot", 0))

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Datum", dag)
with c2:
    st.metric("Slotduur", f"{slot_minutes} min")
with c3:
    st.metric("Update-modus", "Ja" if min_start_slot > 0 else "Nee")

st.divider()

weather = data.get("weather") or {}

st.subheader("Weer (temperatuur, wind, regen) + zon op/onder + commute (heen/terug)")
if not weather:
    st.info("Geen weerdata in planning.json. Draai eerst pv_planning.py met OpenWeather ingeschakeld.")
else:
    plot_weather_panel(
        weather=weather,
        timezone_name=timezone_name,
        commute_out_start=7.0,
        commute_out_end=10.0,
        commute_back_start=16.0,
        commute_back_end=19.0,
    )

st.subheader("Fietsadvies (woon-werk)")
if weather:
    adv_out = bicycle_advice_for_window(weather, timezone_name, start_h=7.0, end_h=10.0)
    adv_back = bicycle_advice_for_window(weather, timezone_name, start_h=16.0, end_h=19.0)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Heen (07:00–10:00)", adv_out["label"], f"score {adv_out['score']}/100")
        st.caption(
            f"Gem temp: {adv_out['details']['temp_avg_c']}°C • "
            f"Max wind: {adv_out['details']['wind_max_bft']} Bft • "
            f"Regen (som): {adv_out['details']['rain_sum_mm_window']} mm"
        )
    with c2:
        st.metric("Terug (16:00–19:00)", adv_back["label"], f"score {adv_back['score']}/100")
        st.caption(
            f"Gem temp: {adv_back['details']['temp_avg_c']}°C • "
            f"Max wind: {adv_back['details']['wind_max_bft']} Bft • "
            f"Regen (som): {adv_back['details']['rain_sum_mm_window']} mm"
        )
else:
    st.info("Geen weerdata beschikbaar voor fietsadvies.")

st.divider()
plot_energy(data)

st.divider()

totals = compute_energy_totals(data)
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Totaal PV (kWh)", f"{totals['pv_kwh']:.1f}")
with m2:
    st.metric("Totaal verbruik (kWh)", f"{totals['used_kwh']:.1f}")
with m3:
    st.metric("Totaal net-import (kWh)", f"{totals['import_kwh']:.1f}")
with m4:
    st.metric("Totaal export (kWh)", f"{totals['export_kwh']:.1f}")

st.divider()

st.subheader("Geplande taken")
tasks_df = make_tasks_df(data)
if tasks_df.empty:
    st.info("Geen taken gepland.")
else:
    st.metric("Totaal energie geplande taken (kWh)", f"{float(tasks_df['Energie (kWh)'].sum()):.2f}")
    group_view = st.checkbox("Groepeer taken (sommeer energie per taak)", value=False, key="group_tasks_local")
    if group_view:
        st.dataframe(group_tasks(tasks_df), use_container_width=True, hide_index=True)
    else:
        st.dataframe(tasks_df, use_container_width=True, hide_index=True)

st.divider()

st.subheader("Slots overzicht")
show_kw = st.checkbox("Toon ook kW-kolommen (vermogen)", value=False, key="show_kw_local")
overview_df = make_overview_df(data, show_kw=show_kw)

with st.expander("Filters", expanded=False):
    show_only_day = st.checkbox("Toon alleen 08:00–20:00", value=True, key="only_day_local")
    show_only_nonzero = st.checkbox("Verberg rijen met 0 PV en 0 verbruik", value=False, key="only_nonzero_local")

df_show = overview_df.copy()
if show_only_day:
    start_slot = int((8 * 60) / slot_minutes)
    end_slot = int((20 * 60) / slot_minutes)
    df_show = df_show.iloc[start_slot:end_slot].reset_index(drop=True)

if show_only_nonzero:
    df_show = df_show[(df_show["PV (kWh/slot)"] > 0) | (df_show["Verbruik (kWh/slot)"] > 0)].reset_index(drop=True)

st.dataframe(df_show, use_container_width=True, hide_index=True)

st.divider()
st.download_button(
    "Download planning.json",
    data=json.dumps(data, ensure_ascii=False, indent=2),
    file_name=os.path.basename(path),
    mime="application/json",
)
st.download_button(
    "Download slots.csv",
    data=df_show.to_csv(index=False).encode("utf-8"),
    file_name="slots.csv",
    mime="text/csv",
)
