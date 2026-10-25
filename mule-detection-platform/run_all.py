"""
=====================================================================
  MASTER RUNNER — Ek command mein sab kuch
  Run:  python run_all.py
=====================================================================
  1. ✅ Data validation
  2. ✅ Dependencies check
  3. ✅ Feature engineering
  4. ✅ Model training (3 models)
  5. ✅ Score fusion + Complaints
  6. ✅ Model comparison (13 algorithms + graphs)
  7. ✅ Start dashboard server
=====================================================================
"""
import os
import sys
import time
import subprocess
import importlib


def print_banner(text, char="="):
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_step(step, total, text):
    print(f"\n{'─' * 70}")
    print(f"  📌 STEP {step}/{total}: {text}")
    print(f"{'─' * 70}")


def run_command(cmd, description):
    """Command run karo, error aaye to batao"""
    print(f"\n  ⏳ Running: {description}")
    print(f"  💻 Command: {cmd}\n")

    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  ✅ {description} — DONE ({elapsed:.1f}s)")
    else:
        print(f"\n  ❌ {description} — FAILED!")
        print(f"     Error code: {result.returncode}")
        response = input("\n     Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("     Stopping here. Fix the error and run again.")
            sys.exit(1)

    return result.returncode


def check_data_files():
    """Pehle check karo ki data files hain ya nahi"""
    print_step(1, 7, "DATA VALIDATION")

    raw_dir = "data/raw"
    required_files = ["SAML-D.csv", "HI-Small_Trans.csv", "Base.csv"]

    if not os.path.exists(raw_dir):
        print(f"\n  ❌ Folder 'data/raw/' exists nahi karta!")
        print(f"     Banao: mkdir data/raw")
        sys.exit(1)

    all_ok = True
    for f in required_files:
        path = os.path.join(raw_dir, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > 0:
                print(f"  ✅ {f:30s} ({size:,} bytes)")
            else:
                print(f"  ❌ {f:30s} (EMPTY FILE!)")
                all_ok = False
        else:
            print(f"  ❌ {f:30s} (NOT FOUND!)")
            all_ok = False

    if not all_ok:
        print(f"\n  ⚠️  Kuch files missing/empty hain!")
        print(f"     'data/raw/' folder mein saari 3 CSV files daalo")
        print(f"     File names exactly: SAML-D.csv, HI-Small_Trans.csv, Base.csv")
        response = input("\n     Still continue? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    print(f"\n  ✅ Output folders ready (data/processed/, models/)")


def main():
    total_start = time.time()

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   🛡️  MULE DETECTION PLATFORM — FULL RUNNER                  ║
    ║                                                               ║
    ║   Ek command mein sab kuch chalega:                          ║
    ║   Data → Features → Models → Comparison → Dashboard          ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # ──────────────────────────────────────────────
    # STEP 1: Data Validation
    # ──────────────────────────────────────────────
    check_data_files()

    # ──────────────────────────────────────────────
    # STEP 2: Install Dependencies
    # ──────────────────────────────────────────────
    print_step(2, 7, "CHECK DEPENDENCIES")

    required_packages = [
        "pandas", "numpy", "sklearn", "xgboost",
        "shap", "networkx", "joblib", "matplotlib",
        "fastapi", "uvicorn"
    ]

    missing = []
    for pkg in required_packages:
        try:
            importlib.import_module(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            missing.append(pkg)
            print(f"  ❌ {pkg} — MISSING")

    if missing:
        pip_names = {
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
        }
        install_names = [pip_names.get(p, p) for p in missing]
        print(f"\n  📦 Missing packages install ho rahe hain: {install_names}")
        run_command(
            "pip install " + " ".join(install_names),
            "Installing missing packages"
        )
    else:
        print(f"\n  ✅ All dependencies installed!")

    # ──────────────────────────────────────────────
    # STEP 3: Feature Engineering
    # ──────────────────────────────────────────────
    print_step(3, 7, "FEATURE ENGINEERING")

    run_command(
        "python -m src.save_features",
        "Building behavioural + graph + identity features"
    )

    # ──────────────────────────────────────────────
    # STEP 4: Model Training (3 Models)
    # ──────────────────────────────────────────────
    print_step(4, 7, "MODEL TRAINING (3 Models)")

    print("\n  📌 Training Model 1: Behavioural (XGBoost)")
    run_command(
        "python -m src.train_behavioural",
        "Model 1: Behavioural XGBoost"
    )

    print("\n  📌 Training Model 2: Graph Anomaly (Isolation Forest)")
    run_command(
        "python -m src.train_graph",
        "Model 2: Graph Anomaly"
    )

    print("\n  📌 Training Model 3: Identity Fraud (XGBoost)")
    run_command(
        "python -m src.train_identity",
        "Model 3: Identity XGBoost"
    )

    # ──────────────────────────────────────────────
    # STEP 5: Score Fusion + Complaints
    # ──────────────────────────────────────────────
    print_step(5, 7, "FUSION + COMPLAINTS")

    run_command(
        "python -m src.fusion",
        "Fusing scores (weighted ensemble)"
    )

    run_command(
        "python -m src.complaints",
        "Generating + correlating complaints"
    )

    # ──────────────────────────────────────────────
    # STEP 6: Model Comparison (Research Paper)
    # ──────────────────────────────────────────────
    print_step(6, 7, "MODEL COMPARISON (13 Algorithms)")

    print("\n  📌 Ye step 5-30 minute le sakta hai (data size pe depend)")
    print("  📌 13 algorithms train honge + graphs banenge\n")

    response = input("  Model comparison chalana hai? (y/n): ")
    if response.lower() == 'y':
        run_command(
            "python compare_models.py",
            "Comparing 13 ML models"
        )
    else:
        print("  ⏭️  Skipped. Baad mein: python compare_models.py")

    # ──────────────────────────────────────────────
    # STEP 7: Start Dashboard
    # ──────────────────────────────────────────────
    print_step(7, 7, "START DASHBOARD SERVER")

    total_elapsed = time.time() - total_start

    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ✅  SAB KUCH COMPLETE!                                      ║
    ║                                                               ║
    ║   ⏱️  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} minutes)                            ║
    ║                                                               ║
    ║   📁 Outputs:                                                 ║
    ║      data/processed/  → features + scores + complaints        ║
    ║      models/          → trained models (.pkl)                 ║
    ║      data/processed/results/ → comparison tables + graphs     ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    print("  🌐 Dashboard start karna hai? (y/n): ", end="")
    response = input()

    if response.lower() == 'y':
        print("\n  🚀 Starting server...")
        print("  📌 Browser mein kholo: http://localhost:8000")
        print("  📌 Server band karne ke liye: Ctrl + C")
        print()
        os.system("uvicorn api.main:app --reload --port 8000")
    else:
        print("\n  📌 Dashboard baad mein start karne ke liye:")
        print("     uvicorn api.main:app --reload --port 8000")
        print()


if __name__ == "__main__":
    main()