"""
generate_logs.py
================
Generates a realistic SSH auth log for ML training and demo.
Includes:
  - Normal login patterns
  - Brute force attacks (rapid successive failures from same IP)
  - Dictionary attacks (many usernames, same IP)
  - Distributed attacks (many IPs, low frequency)
  - Successful logins
  - Port scans
"""

import random
import os
from datetime import datetime, timedelta

OUTPUT = os.path.join(os.path.dirname(__file__), "auth.log")

# Realistic IP pools
ATTACKER_IPS_BRUTE    = ["192.168.1.101", "10.0.0.45", "203.0.113.77",
                          "198.51.100.5",  "172.16.0.9"]
ATTACKER_IPS_DIST     = [f"45.33.{r}.{c}" for r in range(10, 30) for c in range(1, 5)]
LEGIT_IPS             = ["88.99.100.1", "212.45.67.89", "91.108.4.1",
                          "185.76.23.44", "178.62.194.10"]

USERNAMES_COMMON = ["root", "admin", "ubuntu", "user", "deploy", "git",
                    "oracle", "postgres", "mysql", "test", "guest"]
LEGIT_USERS      = ["orkun", "alice", "bob", "carol"]

MONTHS  = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

random.seed(42)


def _ts(dt: datetime) -> str:
    return f"{MONTHS[dt.month-1]} {dt.day:2d} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"


def _failed(dt, user, ip, port=None):
    port = port or random.randint(40000, 65000)
    return f"{_ts(dt)} server sshd[{random.randint(1000,9999)}]: Failed password for {user} from {ip} port {port} ssh2"


def _accepted(dt, user, ip, method="publickey"):
    port = random.randint(40000, 65000)
    return f"{_ts(dt)} server sshd[{random.randint(1000,9999)}]: Accepted {method} for {user} from {ip} port {port} ssh2"


def _invalid(dt, user, ip):
    port = random.randint(40000, 65000)
    return f"{_ts(dt)} server sshd[{random.randint(1000,9999)}]: Invalid user {user} from {ip} port {port}"


def _disconnect(dt, ip):
    port = random.randint(40000, 65000)
    return f"{_ts(dt)} server sshd[{random.randint(1000,9999)}]: Disconnected from invalid user from {ip} port {port} [preauth]"


def generate(n_days: int = 30) -> list[str]:
    lines   = []
    base    = datetime(2024, 1, 1, 0, 0, 0)

    for day in range(n_days):
        day_start = base + timedelta(days=day)

        # ── 1. Brute force attacks (2-4 per day, burst of 20-80 attempts) ──
        for _ in range(random.randint(2, 4)):
            ip    = random.choice(ATTACKER_IPS_BRUTE)
            user  = random.choice(USERNAMES_COMMON)
            hour  = random.randint(0, 5)          # mostly at night
            start = day_start.replace(hour=hour, minute=random.randint(0, 59))
            n_att = random.randint(20, 80)
            for i in range(n_att):
                dt = start + timedelta(seconds=i * random.uniform(0.5, 3))
                # Mix usernames occasionally
                if random.random() < 0.2:
                    user = random.choice(USERNAMES_COMMON)
                lines.append(_failed(dt, user, ip))

        # ── 2. Dictionary attacks (varied usernames, slower pace) ──
        if random.random() < 0.5:
            ip    = random.choice(ATTACKER_IPS_DIST)
            hour  = random.randint(8, 20)
            start = day_start.replace(hour=hour, minute=random.randint(0, 59))
            for i, user in enumerate(USERNAMES_COMMON):
                dt = start + timedelta(seconds=i * random.uniform(5, 30))
                lines.append(_failed(dt, user, ip))
                if random.random() < 0.3:
                    lines.append(_invalid(dt + timedelta(seconds=1), user, ip))

        # ── 3. Distributed low-frequency attacks ──
        n_dist = random.randint(5, 15)
        for _ in range(n_dist):
            ip   = random.choice(ATTACKER_IPS_DIST)
            user = random.choice(USERNAMES_COMMON)
            hour = random.randint(0, 23)
            dt   = day_start.replace(hour=hour,
                                      minute=random.randint(0,59),
                                      second=random.randint(0,59))
            lines.append(_failed(dt, user, ip))

        # ── 4. Legitimate logins ──
        for _ in range(random.randint(2, 6)):
            ip   = random.choice(LEGIT_IPS)
            user = random.choice(LEGIT_USERS)
            hour = random.randint(7, 22)
            dt   = day_start.replace(hour=hour,
                                      minute=random.randint(0,59),
                                      second=random.randint(0,59))
            lines.append(_accepted(dt, user, ip,
                         method=random.choice(["publickey", "password"])))

        # ── 5. Occasional port scan signatures ──
        if random.random() < 0.2:
            ip    = f"192.0.2.{random.randint(1,254)}"
            start = day_start.replace(hour=random.randint(0,23))
            for i in range(random.randint(3, 8)):
                dt = start + timedelta(seconds=i * 0.1)  # very rapid
                lines.append(_disconnect(dt, ip))

    # Sort by timestamp (approximate — by line content)
    lines.sort()
    return lines


def write(path: str = OUTPUT) -> None:
    lines = generate()
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[✓] Generated {len(lines):,} log lines → {path}")


if __name__ == "__main__":
    write()
