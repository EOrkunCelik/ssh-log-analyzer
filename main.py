"""
main.py — SSH Log Analyzer CLI
================================
Usage:
  python main.py [log_file]             # basic analysis
  python main.py [log_file] --ml        # + ML anomaly/intrusion detection
  python main.py [log_file] --ml --geoip # + GeoIP enrichment
  python main.py --api                   # start FastAPI server
  python main.py --generate              # generate sample log
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from main_original import (read_log_file, extract_failed_logins,
                            extract_successful_logins, extract_ip,
                            extract_user, extract_hour, count_by_field,
                            print_summary, export_csv)


def cmd_basic(log_path, args):
    """Original basic analysis."""
    lines   = read_log_file(log_path)
    failed  = extract_failed_logins(lines)
    success = extract_successful_logins(lines)
    ip_counts   = count_by_field(failed, extract_ip)
    user_counts = count_by_field(failed, extract_user)
    hour_counts = count_by_field(failed, extract_hour)
    print_summary(failed, success, ip_counts, user_counts,
                  hour_counts, args.threshold, args.top)
    if args.export_csv:
        export_csv(ip_counts, args.export_csv)


def cmd_ml(log_path, geoip=False):
    """ML pipeline."""
    from src.models import run_ml_pipeline
    run_ml_pipeline(log_path, geoip=geoip)


def cmd_api():
    """Start FastAPI server."""
    try:
        import uvicorn
        print("\n  Starting SSH Log Analyzer API...")
        print("  Swagger UI: http://localhost:8000/docs")
        print("  ReDoc     : http://localhost:8000/redoc\n")
        uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("\n  uvicorn not installed.")
        print("  Install: pip install fastapi uvicorn")
        print("\n  API is fully implemented in api/app.py")
        print("  Endpoints:")
        print("    GET  /              — health check")
        print("    POST /analyze       — analyze uploaded log")
        print("    POST /predict/ip    — classify single IP")
        print("    GET  /anomalies     — list detected anomalies")
        print("    GET  /top-attackers — top attacking IPs")
        print("    GET  /stats         — summary statistics")
        print("    GET  /events        — raw log events")
        print("    POST /geoip/{ip}    — GeoIP lookup")


def main():
    p = argparse.ArgumentParser(
        description="SSH Log Analyzer — ML-powered Intrusion Detection"
    )
    p.add_argument("log_file",   nargs="?",
                   default=os.path.join(os.path.dirname(__file__), "auth.log"),
                   help="Path to SSH auth.log file")
    p.add_argument("--ml",       action="store_true",
                   help="Run ML anomaly + intrusion detection")
    p.add_argument("--geoip",    action="store_true",
                   help="Enrich with GeoIP data (requires internet)")
    p.add_argument("--api",      action="store_true",
                   help="Start FastAPI REST server")
    p.add_argument("--generate", action="store_true",
                   help="Generate a sample auth.log for testing")
    p.add_argument("--threshold",type=int,   default=3)
    p.add_argument("--top",      type=int,   default=5)
    p.add_argument("--export-csv", metavar="FILE")

    args = p.parse_args()

    if args.generate:
        from generate_logs import write
        write()
        return

    if args.api:
        cmd_api()
        return

    if args.ml:
        cmd_ml(args.log_file, geoip=args.geoip)
    else:
        cmd_basic(args.log_file, args)


if __name__ == "__main__":
    main()
