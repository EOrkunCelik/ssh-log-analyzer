"""
models.py
=========
ML-based intrusion detection for SSH logs.

Models:
  1. IsolationForest    — unsupervised anomaly detection
                          no labels needed, detects outlier behavior
  2. LogisticRegression — supervised intrusion classifier
                          trained on heuristically-labeled data

GeoIP:
  Uses ip-api.com (free, no API key) to resolve IPs to countries.
  Results are cached to avoid redundant requests.
"""

import os
import sys
import json
import time
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble        import IsolationForest
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics         import (classification_report,
                                     confusion_matrix, ConfusionMatrixDisplay,
                                     roc_auc_score, roc_curve)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.parser import parse_log, extract_ip_features, label_ips

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "geoip_cache.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DARK = {
    "bg":     "#1a1a2e", "panel":  "#16213e",
    "accent": "#0f3460", "text":   "#ecf0f1",
    "grid":   "#2c3e50", "red":    "#e94560",
    "green":  "#27ae60", "gold":   "#f5a623",
    "blue":   "#3498db", "purple": "#9b59b6",
}

ML_FEATURES = [
    "total_attempts", "failed_attempts", "success_attempts",
    "invalid_user_count", "unique_usernames", "unique_ports",
    "failure_rate", "avg_interval_sec", "min_interval_sec",
    "burst_score", "attempts_per_hour", "night_ratio",
]


def _save(fig, name: str) -> str:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK["bg"])
    plt.close(fig)
    return path


def _styled(fig, ax):
    ax.set_facecolor(DARK["panel"])
    ax.tick_params(colors=DARK["text"])
    ax.spines[:].set_color(DARK["accent"])
    ax.xaxis.label.set_color(DARK["text"])
    ax.yaxis.label.set_color(DARK["text"])
    ax.title.set_color(DARK["text"])
    return ax


# ── GeoIP ─────────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def geoip_lookup(ip: str, cache: dict = None) -> dict:
    """
    Looks up GeoIP info for an IP using ip-api.com (free, no key needed).
    Falls back to a placeholder if offline or rate-limited.
    """
    if cache is None:
        cache = _load_cache()

    if ip in cache:
        return cache[ip]

    # Private/reserved ranges → local
    private_prefixes = ("10.", "172.16.", "192.168.", "127.", "0.")
    if any(ip.startswith(p) for p in private_prefixes):
        result = {"country": "Local Network", "countryCode": "LO",
                  "city": "LAN", "lat": 0.0, "lon": 0.0, "isp": "Private"}
        cache[ip] = result
        _save_cache(cache)
        return result

    try:
        url = f"http://ip-api.com/json/{ip}?fields=status,country,countryCode,city,lat,lon,isp"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
        if data.get("status") == "success":
            result = {
                "country":     data.get("country",     "Unknown"),
                "countryCode": data.get("countryCode", "??"),
                "city":        data.get("city",        "Unknown"),
                "lat":         data.get("lat",         0.0),
                "lon":         data.get("lon",         0.0),
                "isp":         data.get("isp",         "Unknown"),
            }
        else:
            result = {"country": "Unknown", "countryCode": "??",
                      "city": "Unknown", "lat": 0.0, "lon": 0.0, "isp": "Unknown"}
        time.sleep(0.15)  # respect rate limit
    except Exception:
        result = {"country": "Offline", "countryCode": "??",
                  "city": "N/A", "lat": 0.0, "lon": 0.0, "isp": "N/A"}

    cache[ip] = result
    _save_cache(cache)
    return result


def enrich_with_geoip(feat_df: pd.DataFrame,
                       max_ips: int = 50) -> pd.DataFrame:
    """Adds country/city columns to the feature DataFrame."""
    cache  = _load_cache()
    df     = feat_df.copy()
    ips    = df["ip"].tolist()[:max_ips]

    countries, cities, lats, lons = [], [], [], []
    for ip in ips:
        info = geoip_lookup(ip, cache)
        countries.append(info.get("country", "Unknown"))
        cities.append(info.get("city",    "Unknown"))
        lats.append(info.get("lat", 0.0))
        lons.append(info.get("lon", 0.0))

    df["country"] = countries
    df["city"]    = cities
    df["lat"]     = lats
    df["lon"]     = lons
    return df


# ── Model 1: Isolation Forest ─────────────────────────────────────────────────

class AnomalyDetector:
    """
    Unsupervised anomaly detection using Isolation Forest.

    Isolation Forest works by randomly partitioning data;
    anomalies are isolated in fewer splits → shorter path length.
    contamination = expected fraction of anomalies in data.
    """

    def __init__(self, contamination: float = 0.15):
        self.model       = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler      = StandardScaler()
        self.trained     = False
        self.contamination = contamination

    def fit(self, feat_df: pd.DataFrame) -> "AnomalyDetector":
        X = feat_df[ML_FEATURES].fillna(0)
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        self.trained = True
        return self

    def predict(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns feat_df with added columns:
          anomaly_score  — raw anomaly score (lower = more anomalous)
          is_anomaly     — 1 if anomaly, 0 if normal
          risk_level     — "HIGH" | "MEDIUM" | "LOW"
        """
        if not self.trained:
            raise RuntimeError("Call fit() first.")

        df = feat_df.copy()
        X  = df[ML_FEATURES].fillna(0)
        Xs = self.scaler.transform(X)

        # decision_function: negative = more anomalous
        scores = self.model.decision_function(Xs)
        preds  = self.model.predict(Xs)          # -1 = anomaly, 1 = normal

        df["anomaly_score"] = np.round(scores, 4)
        df["is_anomaly"]    = (preds == -1).astype(int)

        # Risk levels based on score percentile
        low_thresh  = np.percentile(scores, 30)
        high_thresh = np.percentile(scores, 70)

        def risk(score, is_anom):
            if not is_anom:
                return "LOW"
            return "HIGH" if score < low_thresh else "MEDIUM"

        df["risk_level"] = [
            risk(s, a) for s, a in zip(df["anomaly_score"], df["is_anomaly"])
        ]
        return df

    def fit_predict(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(feat_df).predict(feat_df)

    def plot_anomaly_scores(self, result_df: pd.DataFrame) -> str:
        """Scatter: attempts vs failure_rate, colored by anomaly."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK["bg"])

        # Plot 1: score distribution
        ax = _styled(fig, axes[0])
        normal  = result_df[result_df["is_anomaly"] == 0]["anomaly_score"]
        anomaly = result_df[result_df["is_anomaly"] == 1]["anomaly_score"]
        ax.hist(normal,  bins=20, color=DARK["green"], alpha=0.7,
                label=f"Normal ({len(normal)})",  zorder=3)
        ax.hist(anomaly, bins=20, color=DARK["red"],   alpha=0.7,
                label=f"Anomaly ({len(anomaly)})", zorder=3)
        ax.set_xlabel("Anomaly Score (lower = more suspicious)")
        ax.set_ylabel("Count")
        ax.set_title("Isolation Forest — Score Distribution",
                     fontsize=12, fontweight="bold")
        ax.legend(facecolor=DARK["panel"], labelcolor=DARK["text"])
        ax.grid(color=DARK["grid"], alpha=0.4, zorder=0)

        # Plot 2: failed_attempts vs burst_score
        ax2 = _styled(fig, axes[1])
        for label_, color, marker in [
            (0, DARK["green"], "o"),
            (1, DARK["red"],   "^"),
        ]:
            sub = result_df[result_df["is_anomaly"] == label_]
            lbl = "Normal" if label_ == 0 else "Anomaly"
            ax2.scatter(sub["failed_attempts"], sub["burst_score"],
                        color=color, alpha=0.7, s=60, marker=marker,
                        edgecolors=DARK["bg"], label=lbl, zorder=3)

        ax2.set_xlabel("Failed Attempts")
        ax2.set_ylabel("Burst Score (rapid-fire attempts)")
        ax2.set_title("Feature Space — Failed Attempts vs Burst",
                      fontsize=12, fontweight="bold")
        ax2.legend(facecolor=DARK["panel"], labelcolor=DARK["text"])
        ax2.grid(color=DARK["grid"], alpha=0.4, zorder=0)

        plt.tight_layout()
        return _save(fig, "anomaly_detection.png")


# ── Model 2: Logistic Regression Intrusion Classifier ────────────────────────

class IntrusionClassifier:
    """
    Supervised binary classifier: ATTACK (1) vs NORMAL (0).
    Trained on heuristically-labeled data from label_ips().

    Outputs probability scores → actionable alerts.
    """

    def __init__(self):
        self.model   = LogisticRegression(C=1.0, max_iter=1000,
                                           random_state=42,
                                           class_weight="balanced")
        self.scaler  = StandardScaler()
        self.trained = False

    def train(self, feat_df: pd.DataFrame, verbose: bool = True) -> dict:
        df = label_ips(feat_df)
        X  = df[ML_FEATURES].fillna(0)
        y  = df["label"]

        print(f"  Class balance: NORMAL={int((y==0).sum())}  ATTACK={int((y==1).sum())}")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        cv = cross_val_score(self.model, X_tr_s, y_tr, cv=5,
                              scoring="roc_auc")
        self.model.fit(X_tr_s, y_tr)

        y_pred  = self.model.predict(X_te_s)
        y_proba = self.model.predict_proba(X_te_s)[:, 1]
        auc     = roc_auc_score(y_te, y_proba)

        self.trained = True

        if verbose:
            print(f"\n  IntrusionClassifier — Logistic Regression")
            print(f"  CV ROC-AUC (5-fold): {cv.mean():.4f} ± {cv.std():.4f}")
            print(f"  Test ROC-AUC       : {auc:.4f}")
            print(f"\n  Classification Report:")
            print(classification_report(y_te, y_pred,
                                         target_names=["NORMAL","ATTACK"],
                                         zero_division=0))

        return {"cv_auc": cv.mean(), "test_auc": auc,
                "y_te": y_te, "y_proba": y_proba}

    def predict_ip(self, ip_features: dict) -> dict:
        """
        Classify a single IP's behavior.
        Returns probability of being an attack + risk level.
        """
        if not self.trained:
            raise RuntimeError("Call train() first.")

        row = pd.DataFrame([{f: ip_features.get(f, 0) for f in ML_FEATURES}])
        Xs  = self.scaler.transform(row)
        prob_attack = float(self.model.predict_proba(Xs)[0][1])
        is_attack   = prob_attack > 0.5

        if prob_attack > 0.8:
            risk = "🔴 CRITICAL"
        elif prob_attack > 0.6:
            risk = "🟠 HIGH"
        elif prob_attack > 0.4:
            risk = "🟡 MEDIUM"
        else:
            risk = "🟢 LOW"

        return {
            "is_attack":    is_attack,
            "probability":  round(prob_attack, 4),
            "risk_level":   risk,
            "verdict":      "ATTACK" if is_attack else "NORMAL",
        }

    def classify_all(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        if not self.trained:
            raise RuntimeError("Call train() first.")
        df = feat_df.copy()
        X  = df[ML_FEATURES].fillna(0)
        Xs = self.scaler.transform(X)
        df["attack_probability"] = np.round(self.model.predict_proba(Xs)[:, 1], 4)
        df["is_attack"]          = (df["attack_probability"] > 0.5).astype(int)
        return df

    def plot_roc_and_features(self, metrics: dict) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK["bg"])

        # ROC curve
        ax = _styled(fig, axes[0])
        fpr, tpr, _ = roc_curve(metrics["y_te"], metrics["y_proba"])
        ax.plot(fpr, tpr, color=DARK["blue"], linewidth=2.5,
                label=f"ROC (AUC={metrics['test_auc']:.3f})")
        ax.plot([0, 1], [0, 1], color=DARK["grid"],
                linewidth=1, linestyle="--", label="Random")
        ax.fill_between(fpr, tpr, alpha=0.15, color=DARK["blue"])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Intrusion Classifier",
                     fontsize=12, fontweight="bold")
        ax.legend(facecolor=DARK["panel"], labelcolor=DARK["text"])
        ax.grid(color=DARK["grid"], alpha=0.4)

        # Feature coefficients
        ax2 = _styled(fig, axes[1])
        coef = self.model.coef_[0]
        sorted_idx = np.argsort(np.abs(coef))
        names  = [ML_FEATURES[i] for i in sorted_idx]
        vals   = coef[sorted_idx]
        colors = [DARK["red"] if v > 0 else DARK["blue"] for v in vals]
        ax2.barh(names, vals, color=colors, edgecolor=DARK["bg"], zorder=3)
        ax2.axvline(0, color=DARK["text"], linewidth=0.8)
        ax2.set_xlabel("Coefficient (positive = attack indicator)")
        ax2.set_title("Feature Weights — Intrusion Classifier",
                      fontsize=12, fontweight="bold")
        ax2.grid(axis="x", color=DARK["grid"], alpha=0.4, zorder=0)

        plt.tight_layout()
        return _save(fig, "intrusion_classifier.png")


# ── GeoIP attack map ──────────────────────────────────────────────────────────

def plot_attack_origins(result_df: pd.DataFrame) -> str:
    """
    Bar chart of attack origins by country.
    (Full geo map requires folium/plotly which need network install)
    """
    if "country" not in result_df.columns:
        return ""

    attack_df = result_df[result_df.get("is_anomaly", result_df.get("is_attack", 0)) == 1]
    if attack_df.empty:
        return ""

    country_counts = attack_df["country"].value_counts().head(15)

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=DARK["bg"])
    _styled(fig, ax)

    colors = [DARK["red"] if i < 3 else DARK["gold"] if i < 8
              else DARK["blue"] for i in range(len(country_counts))]
    bars = ax.barh(country_counts.index[::-1],
                   country_counts.values[::-1],
                   color=colors[::-1], edgecolor=DARK["bg"], zorder=3)

    for bar, val in zip(bars, country_counts.values[::-1]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                str(val), va="center", color=DARK["text"],
                fontsize=9, fontweight="bold")

    ax.set_xlabel("Number of Attacks")
    ax.set_title("Attack Origins by Country (GeoIP)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", color=DARK["grid"], alpha=0.4, zorder=0)
    plt.tight_layout()
    return _save(fig, "attack_origins.png")


# ── Full ML pipeline ──────────────────────────────────────────────────────────

def run_ml_pipeline(log_path: str,
                    geoip: bool = False,
                    verbose: bool = True) -> dict:
    """
    Full pipeline:
      1. Parse log → DataFrame
      2. Extract IP features
      3. Isolation Forest anomaly detection
      4. Logistic Regression intrusion classification
      5. GeoIP enrichment (optional, needs internet)
      6. Generate all plots
    """
    print("\n" + "=" * 52)
    print("  SSH Log ML Analysis Pipeline")
    print("=" * 52)

    # Parse
    print(f"\n[1/5] Parsing log: {log_path}")
    df = parse_log(log_path)
    print(f"  Parsed {len(df):,} events  |  {df['ip'].nunique()} unique IPs")

    # Feature extraction
    print("\n[2/5] Extracting IP features...")
    feat_df = extract_ip_features(df)
    print(f"  Feature matrix: {feat_df.shape[0]} IPs × {feat_df.shape[1]} features")

    # Isolation Forest
    print("\n[3/5] Running Isolation Forest anomaly detection...")
    detector  = AnomalyDetector(contamination=0.15)
    result_df = detector.fit_predict(feat_df)
    n_anom    = result_df["is_anomaly"].sum()
    print(f"  Detected: {n_anom} anomalies / {len(result_df)} IPs "
          f"({n_anom/len(result_df)*100:.1f}%)")
    if verbose:
        print("\n  Top anomalies:")
        top = result_df[result_df["is_anomaly"]==1].nsmallest(5, "anomaly_score")
        for _, r in top.iterrows():
            print(f"  {r['ip']:<20} score={r['anomaly_score']:+.4f}  "
                  f"failed={int(r['failed_attempts'])}  "
                  f"burst={int(r['burst_score'])}  "
                  f"risk={r['risk_level']}")

    # Logistic Regression
    print("\n[4/5] Training Intrusion Classifier...")
    classifier = IntrusionClassifier()
    lr_metrics = classifier.train(feat_df, verbose=verbose)
    full_result = classifier.classify_all(feat_df)

    # GeoIP
    if geoip:
        print("\n[5/5] GeoIP enrichment (requires internet)...")
        result_df = enrich_with_geoip(result_df, max_ips=30)
        full_result = enrich_with_geoip(full_result, max_ips=30)
    else:
        print("\n[5/5] GeoIP skipped (use --geoip flag to enable)")

    # Plots
    print("\n  Generating plots...")
    p1 = detector.plot_anomaly_scores(result_df)
    p2 = classifier.plot_roc_and_features(lr_metrics)
    print(f"  ✓ {p1}")
    print(f"  ✓ {p2}")

    if geoip and "country" in result_df.columns:
        p3 = plot_attack_origins(result_df)
        if p3:
            print(f"  ✓ {p3}")

    # Summary
    n_attacks = full_result["is_attack"].sum()
    print(f"\n{'='*52}")
    print(f"  SUMMARY")
    print(f"  Total events     : {len(df):,}")
    print(f"  Unique IPs       : {feat_df.shape[0]}")
    print(f"  Anomalies (IF)   : {n_anom}  ({n_anom/len(feat_df)*100:.1f}%)")
    print(f"  Attacks (LR)     : {n_attacks}  ({n_attacks/len(full_result)*100:.1f}%)")
    print(f"  LR ROC-AUC       : {lr_metrics['test_auc']:.4f}")
    print(f"{'='*52}\n")

    return {
        "log_df":        df,
        "feat_df":       feat_df,
        "anomaly_result":result_df,
        "attack_result": full_result,
        "detector":      detector,
        "classifier":    classifier,
        "lr_metrics":    lr_metrics,
    }


if __name__ == "__main__":
    import sys
    log_path = sys.argv[1] if len(sys.argv) > 1 else \
               os.path.join(os.path.dirname(__file__), "..", "auth.log")
    geoip    = "--geoip" in sys.argv

    results = run_ml_pipeline(log_path, geoip=geoip)

    # Demo: classify a specific IP
    clf = results["classifier"]
    print("  DEMO: Classify specific IP behavior")
    print("  " + "─" * 40)
    test_cases = [
        {"name": "Brute forcer",   "failed_attempts": 150, "burst_score": 45,
         "unique_usernames": 2,  "failure_rate": 0.99, "attempts_per_hour": 300,
         "total_attempts": 152, "success_attempts": 1, "invalid_user_count": 0,
         "unique_ports": 1, "avg_interval_sec": 2.0, "min_interval_sec": 0.5,
         "night_ratio": 0.9},
        {"name": "Normal user",    "failed_attempts": 1,   "burst_score": 0,
         "unique_usernames": 1,  "failure_rate": 0.2,  "attempts_per_hour": 0.5,
         "total_attempts": 5,   "success_attempts": 4, "invalid_user_count": 0,
         "unique_ports": 3, "avg_interval_sec": 3600, "min_interval_sec": 1800,
         "night_ratio": 0.0},
        {"name": "Dict attacker",  "failed_attempts": 25,  "burst_score": 3,
         "unique_usernames": 11, "failure_rate": 1.0,  "attempts_per_hour": 12,
         "total_attempts": 25,  "success_attempts": 0, "invalid_user_count": 8,
         "unique_ports": 1, "avg_interval_sec": 15.0, "min_interval_sec": 5.0,
         "night_ratio": 0.2},
    ]
    for tc in test_cases:
        name = tc.pop("name")
        pred = clf.predict_ip(tc)
        print(f"  {name:<20} → {pred['verdict']:<8}  "
              f"prob={pred['probability']:.3f}  {pred['risk_level']}")
