# SSH Log Analyzer

ML-powered SSH intrusion detection system with a FastAPI REST backend.
Analyzes auth logs using Isolation Forest (anomaly detection) and Logistic Regression (intrusion classification).

## Features

### Machine Learning
| Model | Type | Task | Performance |
|-------|------|------|-------------|
| Isolation Forest | Unsupervised | Anomaly detection | contamination=0.15 |
| Logistic Regression | Supervised | Intrusion classification | ROC-AUC ≈ 0.99 |

**Isolation Forest** — no labels needed. Detects outlier behavior by measuring how quickly an IP can be "isolated" from normal traffic. Lower anomaly score = more suspicious.

**Logistic Regression** — trained on heuristically-labeled data. Features: failed attempts, burst score, unique usernames, failure rate, night ratio, etc.

### FastAPI REST API (8 endpoints)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/analyze` | Upload + analyze log file |
| POST | `/predict/ip` | Classify single IP in real-time |
| GET | `/anomalies` | List Isolation Forest anomalies |
| GET | `/top-attackers` | Top attacking IPs by probability |
| GET | `/stats` | Full analysis summary |
| GET | `/events` | Raw parsed log events |
| POST | `/geoip/{ip}` | GeoIP lookup |

### GeoIP Enrichment
- Uses ip-api.com (free, no API key needed)
- Results cached locally to avoid repeated requests
- Attack origin chart by country

### Feature Engineering (per IP)
- `failed_attempts`, `burst_score` (rapid-fire detection)
- `unique_usernames` (dictionary attack detection)
- `failure_rate`, `attempts_per_hour`
- `night_ratio` (attacks at 00:00–06:00)
- `avg_interval_sec`, `min_interval_sec`

## Quick Start

```bash
git clone https://github.com/your-username/ssh-log-analyzer.git
cd ssh-log-analyzer
pip install -r requirements.txt

# Generate a sample log (30 days, 5,757 events)
python main.py --generate

# Basic analysis
python main.py auth.log

# Full ML pipeline
python main.py auth.log --ml

# ML + GeoIP enrichment (requires internet)
python main.py auth.log --ml --geoip

# Start FastAPI server → http://localhost:8000/docs
python main.py --api
```

## API Usage

```bash
# Analyze a log file
curl -X POST http://localhost:8000/analyze \
  -F "file=@auth.log"

# Classify an IP in real-time
curl -X POST http://localhost:8000/predict/ip \
  -H "Content-Type: application/json" \
  -d '{
    "total_attempts": 150,
    "failed_attempts": 148,
    "burst_score": 45,
    "unique_usernames": 5,
    "failure_rate": 0.987,
    "attempts_per_hour": 300,
    "night_ratio": 0.9
  }'

# Response:
# {"verdict": "ATTACK", "probability": 0.986, "risk_level": "🔴 CRITICAL"}

# Get anomalies
curl http://localhost:8000/anomalies?limit=10

# Get stats
curl http://localhost:8000/stats
```

## Example Output

```
====================================================
  SSH Log ML Analysis Pipeline
====================================================
[1/5] Parsing log: auth.log
  Parsed 5,757 events  |  101 unique IPs

[2/5] Extracting IP features...
  Feature matrix: 101 IPs × 15 features

[3/5] Running Isolation Forest anomaly detection...
  Detected: 14 anomalies / 101 IPs (13.9%)

  Top anomalies:
  172.16.0.9   score=-0.1833  failed=1166  burst=1145  risk=HIGH
  192.168.1.101 score=-0.1130  failed=939  burst=922  risk=HIGH

[4/5] Training Intrusion Classifier...
  CV ROC-AUC: 0.9852 ± 0.0181
  Test ROC-AUC: 0.9875

  DEMO: Classify IPs
  Brute forcer  → ATTACK  prob=0.986  🔴 CRITICAL
  Normal user   → NORMAL  prob=0.000  🟢 LOW
  Dict attacker → ATTACK  prob=0.997  🔴 CRITICAL
```

## Project Structure

```
ssh-log-analyzer/
├── main.py              # CLI entry point
├── main_original.py     # Original basic analyzer
├── generate_logs.py     # Realistic log generator (30 days)
├── requirements.txt
├── auth.log             # Sample log (git-ignored)
├── src/
│   ├── parser.py        # Log parser + feature extractor
│   └── models.py        # Isolation Forest + Logistic Regression + GeoIP
├── api/
│   └── app.py           # FastAPI REST API (8 endpoints)
└── output/              # Generated charts
    ├── anomaly_detection.png
    └── intrusion_classifier.png
```

## Author

Ercan Orkun Celik — Computer Engineering Student, University of Duisburg-Essen  
Coursework: Security in Communication Networks, Cryptography  
[GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/your-profile)
