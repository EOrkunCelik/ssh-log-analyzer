"""
api/app.py
==========
FastAPI REST API for SSH Log Analyzer.

Endpoints:
  GET  /                    — health check
  POST /analyze             — analyze uploaded log file
  POST /predict/ip          — classify a single IP's behavior
  GET  /anomalies           — get anomalies from last analysis
  GET  /stats               — summary statistics
  GET  /top-attackers       — top attacking IPs
  POST /geoip/{ip}          — GeoIP lookup for an IP
  GET  /docs                — Swagger UI (auto-generated)

Run:
  uvicorn api.app:app --reload --port 8000
"""

import os
import sys
import io
import tempfile
from typing import Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from fastapi             import FastAPI, UploadFile, File, HTTPException, Query
    from fastapi.responses   import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic            import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from src.parser import parse_log, extract_ip_features, label_ips
from src.models import AnomalyDetector, IntrusionClassifier, geoip_lookup

# ── App setup ─────────────────────────────────────────────────────────────────

if not FASTAPI_AVAILABLE:
    print("[WARNING] FastAPI not installed. Run: pip install fastapi uvicorn")
    print("          API code is ready — install packages to launch.")
    # Export a dummy for import compatibility
    app = None
else:
    app = FastAPI(
        title="SSH Log Analyzer API",
        description="""
## SSH Log Analyzer — ML-powered Intrusion Detection

Analyze SSH authentication logs using:
- **Isolation Forest** for unsupervised anomaly detection
- **Logistic Regression** for supervised intrusion classification
- **GeoIP** for attack origin mapping

### Quick Start
1. `POST /analyze` — upload your auth.log
2. `GET /anomalies` — see detected threats
3. `POST /predict/ip` — classify any IP in real-time
        """,
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── In-memory state (production: use Redis/DB) ─────────────────────────────

    _state = {
        "log_df":       None,
        "feat_df":      None,
        "anomaly_df":   None,
        "attack_df":    None,
        "detector":     None,
        "classifier":   None,
        "last_analyzed": None,
        "log_path":     None,
    }

    def _ensure_analyzed():
        if _state["feat_df"] is None:
            raise HTTPException(
                status_code=400,
                detail="No log analyzed yet. POST to /analyze first."
            )

    # ── Request/Response models ────────────────────────────────────────────────

    class IPFeatures(BaseModel):
        total_attempts:     int   = Field(0, description="Total login attempts")
        failed_attempts:    int   = Field(0, description="Failed attempts")
        success_attempts:   int   = Field(0, description="Successful logins")
        invalid_user_count: int   = Field(0, description="Invalid user attempts")
        unique_usernames:   int   = Field(1, description="Distinct usernames tried")
        unique_ports:       int   = Field(1, description="Distinct ports used")
        failure_rate:       float = Field(0.0, description="Failed/total ratio")
        avg_interval_sec:   float = Field(0.0, description="Avg seconds between attempts")
        min_interval_sec:   float = Field(0.0, description="Min seconds between attempts")
        burst_score:        int   = Field(0,   description="Count of attempts < 5s apart")
        attempts_per_hour:  float = Field(0.0, description="Hourly attempt rate")
        night_ratio:        float = Field(0.0, description="Fraction of night attempts")

        class Config:
            json_schema_extra = {
                "example": {
                    "total_attempts": 150, "failed_attempts": 148,
                    "success_attempts": 2, "invalid_user_count": 10,
                    "unique_usernames": 5, "unique_ports": 1,
                    "failure_rate": 0.987, "avg_interval_sec": 2.0,
                    "min_interval_sec": 0.5, "burst_score": 45,
                    "attempts_per_hour": 300, "night_ratio": 0.9,
                }
            }

    class PredictionResponse(BaseModel):
        ip_features:  dict
        is_attack:    bool
        probability:  float
        risk_level:   str
        verdict:      str
        timestamp:    str

    class AnalyzeResponse(BaseModel):
        status:          str
        total_events:    int
        unique_ips:      int
        anomalies_found: int
        attacks_found:   int
        roc_auc:         float
        analyzed_at:     str

    # ── Endpoints ──────────────────────────────────────────────────────────────

    @app.get("/", tags=["Health"])
    def health():
        """Health check endpoint."""
        return {
            "status":  "online",
            "service": "SSH Log Analyzer API",
            "version": "2.0.0",
            "models":  ["IsolationForest", "LogisticRegression"],
            "docs":    "/docs",
        }

    @app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
    async def analyze_log(
        file: UploadFile = File(..., description="SSH auth.log file to analyze"),
        contamination: float = Query(0.15, description="Isolation Forest contamination (0.05–0.4)"),
    ):
        """
        Upload and analyze an SSH auth.log file.

        Runs the full ML pipeline:
        1. Parse log into structured events
        2. Extract per-IP behavioral features
        3. Isolation Forest anomaly detection
        4. Logistic Regression intrusion classification

        Returns summary statistics. Use /anomalies and /top-attackers for details.
        """
        if not file.filename.endswith((".log", ".txt", "")):
            raise HTTPException(400, "Please upload a .log or .txt file.")

        content = await file.read()
        if len(content) > 50 * 1024 * 1024:  # 50 MB limit
            raise HTTPException(413, "File too large (max 50MB).")

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".log",
                                          delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            log_df   = parse_log(tmp_path)
            if log_df.empty:
                raise HTTPException(422, "No parseable SSH log entries found.")

            feat_df  = extract_ip_features(log_df)

            detector   = AnomalyDetector(contamination=min(max(contamination, 0.05), 0.4))
            anomaly_df = detector.fit_predict(feat_df)

            classifier = IntrusionClassifier()
            lr_metrics = classifier.train(feat_df, verbose=False)
            attack_df  = classifier.classify_all(feat_df)

            # Store state
            _state.update({
                "log_df":       log_df,
                "feat_df":      feat_df,
                "anomaly_df":   anomaly_df,
                "attack_df":    attack_df,
                "detector":     detector,
                "classifier":   classifier,
                "last_analyzed":datetime.utcnow().isoformat(),
                "log_path":     tmp_path,
            })

            return AnalyzeResponse(
                status          = "success",
                total_events    = len(log_df),
                unique_ips      = len(feat_df),
                anomalies_found = int(anomaly_df["is_anomaly"].sum()),
                attacks_found   = int(attack_df["is_attack"].sum()),
                roc_auc         = round(lr_metrics["test_auc"], 4),
                analyzed_at     = _state["last_analyzed"],
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Analysis failed: {str(e)}")

    @app.post("/predict/ip", response_model=PredictionResponse, tags=["Prediction"])
    def predict_ip(features: IPFeatures):
        """
        Classify a single IP's behavior in real-time.

        Provide behavioral features extracted from logs for that IP.
        Returns attack probability and risk level.

        **Risk levels:**
        - 🔴 CRITICAL (prob > 0.8)
        - 🟠 HIGH (prob > 0.6)
        - 🟡 MEDIUM (prob > 0.4)
        - 🟢 LOW (prob ≤ 0.4)
        """
        _ensure_analyzed()
        result = _state["classifier"].predict_ip(features.model_dump())
        return PredictionResponse(
            ip_features = features.model_dump(),
            is_attack   = result["is_attack"],
            probability = result["probability"],
            risk_level  = result["risk_level"],
            verdict     = result["verdict"],
            timestamp   = datetime.utcnow().isoformat(),
        )

    @app.get("/anomalies", tags=["Analysis"])
    def get_anomalies(
        limit: int = Query(20, description="Max results to return"),
        risk:  Optional[str] = Query(None, description="Filter by risk: HIGH or MEDIUM"),
    ):
        """Returns anomalous IPs detected by Isolation Forest."""
        _ensure_analyzed()
        df = _state["anomaly_df"]
        df = df[df["is_anomaly"] == 1].copy()
        if risk:
            df = df[df["risk_level"] == risk.upper()]
        df = df.nsmallest(limit, "anomaly_score")

        return {
            "total_anomalies": int(_state["anomaly_df"]["is_anomaly"].sum()),
            "shown":           len(df),
            "anomalies": df[[
                "ip","failed_attempts","burst_score","unique_usernames",
                "failure_rate","anomaly_score","risk_level"
            ]].to_dict(orient="records")
        }

    @app.get("/top-attackers", tags=["Analysis"])
    def top_attackers(limit: int = Query(10, description="Number of top attackers")):
        """Returns top attacking IPs by attack probability."""
        _ensure_analyzed()
        df = _state["attack_df"]
        df = df[df["is_attack"] == 1].nlargest(limit, "attack_probability")
        return {
            "total_attacks": int(df["is_attack"].sum()),
            "attackers": df[[
                "ip","failed_attempts","burst_score","unique_usernames",
                "failure_rate","attack_probability"
            ]].to_dict(orient="records")
        }

    @app.get("/stats", tags=["Analysis"])
    def get_stats():
        """Returns summary statistics from the last analysis."""
        _ensure_analyzed()
        df     = _state["feat_df"]
        log_df = _state["log_df"]
        anom   = _state["anomaly_df"]
        atk    = _state["attack_df"]

        event_counts = log_df["event_type"].value_counts().to_dict()

        return {
            "last_analyzed":  _state["last_analyzed"],
            "total_events":   len(log_df),
            "unique_ips":     len(df),
            "event_breakdown":event_counts,
            "anomaly_detection": {
                "total_anomalies": int(anom["is_anomaly"].sum()),
                "high_risk":       int((anom["risk_level"] == "HIGH").sum()),
                "medium_risk":     int((anom["risk_level"] == "MEDIUM").sum()),
            },
            "intrusion_detection": {
                "total_attacks":      int(atk["is_attack"].sum()),
                "avg_attack_prob":    round(float(atk["attack_probability"].mean()), 4),
                "max_failed_attempts":int(df["failed_attempts"].max()),
                "max_burst_score":    int(df["burst_score"].max()),
            },
            "top_usernames_targeted": (
                log_df[log_df["event_type"] == "FAILED"]["username"]
                .value_counts().head(5).to_dict()
            ),
        }

    @app.get("/events", tags=["Analysis"])
    def get_events(
        event_type: Optional[str] = Query(None, description="Filter: FAILED, ACCEPTED, INVALID_USER"),
        ip:         Optional[str] = Query(None, description="Filter by IP"),
        limit:      int           = Query(50, description="Max events to return"),
    ):
        """Returns raw parsed log events."""
        _ensure_analyzed()
        df = _state["log_df"].copy()
        if event_type:
            df = df[df["event_type"] == event_type.upper()]
        if ip:
            df = df[df["ip"] == ip]
        df = df.head(limit)
        df["timestamp"] = df["timestamp"].astype(str)
        return {
            "total_matching": len(df),
            "events": df[["timestamp","event_type","username","ip","port"]].to_dict(orient="records")
        }

    @app.post("/geoip/{ip}", tags=["GeoIP"])
    def lookup_geoip(ip: str):
        """
        Looks up GeoIP information for an IP address.
        Uses ip-api.com (requires internet connection).
        Results are cached locally.
        """
        result = geoip_lookup(ip)
        return {"ip": ip, **result}


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("\n  FastAPI not installed.")
        print("  Install with: pip install fastapi uvicorn")
        print("\n  API code is complete and ready to run.")
    else:
        try:
            import uvicorn
            print("\n  Starting SSH Log Analyzer API...")
            print("  Docs: http://localhost:8000/docs\n")
            uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
        except ImportError:
            print("  uvicorn not installed. Run: pip install uvicorn")
