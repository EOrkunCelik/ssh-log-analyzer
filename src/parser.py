"""
parser.py
=========
Parses SSH auth logs into structured DataFrames.
Extracts features for ML anomaly detection.
"""

import re
import os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Optional

# ── Regex patterns ────────────────────────────────────────────────────────────

RE_FAILED   = re.compile(
    r"(\w+\s+\d+\s+\d+:\d+:\d+).*Failed password for (\S+) from (\d+\.\d+\.\d+\.\d+) port (\d+)")
RE_ACCEPTED = re.compile(
    r"(\w+\s+\d+\s+\d+:\d+:\d+).*Accepted (\S+) for (\S+) from (\d+\.\d+\.\d+\.\d+) port (\d+)")
RE_INVALID  = re.compile(
    r"(\w+\s+\d+\s+\d+:\d+:\d+).*Invalid user (\S+) from (\d+\.\d+\.\d+\.\d+)")
RE_DISCONNECT = re.compile(
    r"(\w+\s+\d+\s+\d+:\d+:\d+).*Disconnected.*from (\d+\.\d+\.\d+\.\d+)")

MONTH_MAP = {m: i+1 for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun",
     "Jul","Aug","Sep","Oct","Nov","Dec"])}


def _parse_ts(ts_str: str, year: int = 2024) -> Optional[datetime]:
    try:
        parts = ts_str.split()
        month = MONTH_MAP.get(parts[0], 1)
        day   = int(parts[1])
        h, m, s = map(int, parts[2].split(":"))
        return datetime(year, month, day, h, m, s)
    except Exception:
        return None


def parse_log(path: str) -> pd.DataFrame:
    """
    Parses a raw auth.log file into a structured DataFrame.

    Returns columns:
      timestamp, event_type, username, ip, port, auth_method
    """
    records = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = RE_FAILED.search(line)
            if m:
                ts = _parse_ts(m.group(1))
                records.append({
                    "timestamp":   ts,
                    "event_type":  "FAILED",
                    "username":    m.group(2),
                    "ip":          m.group(3),
                    "port":        int(m.group(4)),
                    "auth_method": "password",
                })
                continue

            m = RE_ACCEPTED.search(line)
            if m:
                ts = _parse_ts(m.group(1))
                records.append({
                    "timestamp":   ts,
                    "event_type":  "ACCEPTED",
                    "username":    m.group(3),
                    "ip":          m.group(4),
                    "port":        int(m.group(5)),
                    "auth_method": m.group(2),
                })
                continue

            m = RE_INVALID.search(line)
            if m:
                ts = _parse_ts(m.group(1))
                records.append({
                    "timestamp":   ts,
                    "event_type":  "INVALID_USER",
                    "username":    m.group(2),
                    "ip":          m.group(3),
                    "port":        None,
                    "auth_method": None,
                })
                continue

            m = RE_DISCONNECT.search(line)
            if m:
                ts = _parse_ts(m.group(1))
                records.append({
                    "timestamp":  ts,
                    "event_type": "DISCONNECT",
                    "username":   None,
                    "ip":         m.group(2),
                    "port":       None,
                    "auth_method":None,
                })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = df.dropna(subset=["timestamp", "ip"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"]    = df["timestamp"].dt.hour
    df["minute"]  = df["timestamp"].dt.minute
    df["day"]     = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday
    return df


def extract_ip_features(df: pd.DataFrame,
                          window_minutes: int = 10) -> pd.DataFrame:
    """
    Aggregates per-IP features for ML.

    Features per IP:
      total_attempts     — total login attempts
      failed_attempts    — failed attempts
      success_attempts   — successful logins
      unique_usernames   — number of distinct usernames tried
      unique_ports       — number of distinct ports used
      failure_rate       — failed / total
      avg_interval_sec   — avg seconds between attempts
      min_interval_sec   — min seconds between attempts (burst detection)
      attempts_per_hour  — hourly attempt rate
      night_ratio        — fraction of attempts between 00:00–06:00
      invalid_user_count — attempts with non-existent usernames
      burst_score        — count of intervals < 5 seconds (rapid fire)
    """
    records = []

    for ip, grp in df.groupby("ip"):
        grp = grp.sort_values("timestamp")

        total      = len(grp)
        failed     = (grp["event_type"] == "FAILED").sum()
        success    = (grp["event_type"] == "ACCEPTED").sum()
        invalid    = (grp["event_type"] == "INVALID_USER").sum()
        disconnect = (grp["event_type"] == "DISCONNECT").sum()

        usernames  = grp["username"].dropna().nunique()
        ports      = grp["port"].dropna().nunique()

        # Time-based features
        ts_vals = grp["timestamp"].values
        if len(ts_vals) > 1:
            diffs = np.diff(ts_vals.astype("int64")) / 1e9  # nanoseconds → seconds
            avg_interval = float(np.mean(diffs))
            min_interval = float(np.min(diffs))
            burst_score  = int(np.sum(diffs < 5))
        else:
            avg_interval = 0.0
            min_interval = 0.0
            burst_score  = 0

        # Duration of activity
        if len(ts_vals) > 1:
            duration_hours = (ts_vals[-1] - ts_vals[0]).astype("timedelta64[s]").astype(float) / 3600
        else:
            duration_hours = 0.0

        attempts_per_hour = total / max(duration_hours, 0.01)
        failure_rate      = failed / max(total, 1)
        night_ratio       = ((grp["hour"] < 6).sum()) / max(total, 1)

        records.append({
            "ip":                ip,
            "total_attempts":    total,
            "failed_attempts":   failed,
            "success_attempts":  success,
            "invalid_user_count":invalid,
            "disconnect_count":  disconnect,
            "unique_usernames":  usernames,
            "unique_ports":      ports,
            "failure_rate":      round(failure_rate, 4),
            "avg_interval_sec":  round(avg_interval, 2),
            "min_interval_sec":  round(min_interval, 2),
            "burst_score":       burst_score,
            "attempts_per_hour": round(attempts_per_hour, 2),
            "night_ratio":       round(night_ratio, 4),
            "duration_hours":    round(duration_hours, 2),
        })

    return pd.DataFrame(records)


def label_ips(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic labeling for Logistic Regression training.
    Labels IPs as ATTACK (1) or NORMAL (0) based on rule-based thresholds.

    Rules for ATTACK:
      - failed_attempts > 10, OR
      - burst_score > 5 (many attempts < 5 sec apart), OR
      - unique_usernames > 3 AND failure_rate > 0.9, OR
      - attempts_per_hour > 50
    """
    df = feat_df.copy()

    attack_mask = (
        (df["failed_attempts"]   > 10)  |
        (df["burst_score"]       > 5)   |
        ((df["unique_usernames"] > 3)   & (df["failure_rate"] > 0.9)) |
        (df["attempts_per_hour"] > 50)
    )
    df["label"] = attack_mask.astype(int)
    return df
