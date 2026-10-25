import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

SAMLD_PATH = os.path.join(RAW_DIR, "SAML-D.csv")
HI_PATH = os.path.join(RAW_DIR, "HI-Small_Trans.csv")
BASE_PATH = os.path.join(RAW_DIR, "Base.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
RISK_HIGH = 70
RISK_MEDIUM = 40
W_BEHAVIOURAL = 0.50
W_GRAPH = 0.30
W_IDENTITY = 0.20
TOP_K_ALERTS = 50

if __name__ == "__main__":
    print(f"Base dir: {BASE_DIR}")
    print(f"Raw dir : {RAW_DIR}")
    print(f"Proc dir: {PROCESSED_DIR}")
    print(f"Model   : {MODEL_DIR}")