"""
Project Name : Healthcare Cost & Claims Analysis
Author       : Twinkle Grover
Objective    : Analyze CMS Medicare claims data and predict reimbursement cost
Dataset      : CMS DE-SynPUF
Model        : XGBoost with log-transformed target
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from pathlib import Path
plt.style.use('ggplot')

print("==================================================")
print("CMS CLAIMS DATA INGESTION | XGBoost Model")
print("==================================================")

FILE_PATH = Path(
    r"C:\Users\Hp\Videos\Python,ML & DL\Adhoc Project\CMS Insurance\Master_claims_beni.csv"
)

start_time = datetime.now()

try:
    df = pd.read_csv(FILE_PATH, nrows=100000, low_memory=False)
    print("Data Loaded Successfully")
    print(df.head())
except Exception as e:
    print(f"Error Loading Dataset: {e}")
    raise

print("Dataset Shape:", df.shape)
print(df.columns.tolist())
df.info()
print(df.describe(include="object"))
df.describe()

claims_columns = ['DESYNPUF_ID','CLM_ID', 'CLM_FROM_DT', 'CLM_THRU_DT', 'ICD9_DGNS_CD_1',
                  'PRF_PHYSN_NPI_1',
                  'HCPCS_CD_1', 'LINE_NCH_PMT_AMT_1', 'LINE_BENE_PTB_DDCTBL_AMT_1',
                  'LINE_COINSRNC_AMT_1', 'LINE_PRCSG_IND_CD_1', 'LINE_ICD9_DGNS_CD_1']

bene_columns = ['DESYNPUF_ID', 'BENE_BIRTH_DT', 'BENE_DEATH_DT',
                 'BENE_SEX_IDENT_CD', 'BENE_RACE_CD', 'BENE_ESRD_IND',
                 'SP_STATE_CODE', 'BENE_COUNTY_CD', 'BENE_HI_CVRAGE_TOT_MONS',
                 'BENE_SMI_CVRAGE_TOT_MONS', 'BENE_HMO_CVRAGE_TOT_MONS',
                 'PLAN_CVRG_MOS_NUM', 'SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN',
                 'SP_CNCR', 'SP_COPD', 'SP_DEPRESSN', 'SP_DIABETES', 'SP_ISCHMCHT',
                 'SP_OSTEOPRS', 'SP_RA_OA', 'SP_STRKETIA', 'MEDREIMB_IP', 'BENRES_IP',
                 'PPPYMT_IP', 'MEDREIMB_OP', 'BENRES_OP', 'PPPYMT_OP', 'MEDREIMB_CAR',
                 'BENRES_CAR', 'PPPYMT_CAR']

DATE_COLUMNS = ["BENE_BIRTH_DT", "BENE_DEATH_DT", "CLM_FROM_DT", "CLM_THRU_DT"]

CHRONIC_CONDITION_COLS = [
    "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD",
    "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS",
    "SP_RA_OA", "SP_STRKETIA",
]


def get_beneficiary_summary(df):
    cols = [col for col in bene_columns if col in df.columns]
    return df[cols].drop_duplicates("DESYNPUF_ID")


def get_claim_lines(df):
    claim_cols = [c for c in claims_columns if c in df.columns]
    return df[claim_cols]


missing_values = (
    df
    .isnull()
    .sum()
    .sort_values(ascending=False)
)
print(missing_values.head(20))

import missingno as msno
msno.bar(df, fontsize=20, color='green')
plt.show()

existing_date_cols = [c for c in DATE_COLUMNS if c in df.columns]
if existing_date_cols:
    df[existing_date_cols] = df[existing_date_cols].apply(
        pd.to_datetime, format='%Y%m%d', errors='coerce'
    )

if "BENE_BIRTH_DT" in df.columns and "bene_age" not in df.columns:
    df["bene_age"] = datetime.now().year - df["BENE_BIRTH_DT"].dt.year
    print(df["bene_age"])

for col in df.select_dtypes(include="object"):
    df[col] = df[col].astype(str).str.strip().str.lower()

for col in CHRONIC_CONDITION_COLS:
    if col in df.columns:
        df[col] = df[col].replace({1: 0, 2: 1})


# ==========================================
# DATA CLEANING
# ==========================================
print("\n========== DATA CLEANING ==========")

before = df.shape[0]
df = df.drop_duplicates()
print(f"Removed {before - df.shape[0]} duplicate rows")

DROP_COLS = [
    "DESYNPUF_ID", "CLM_ID", "PRF_PHYSN_NPI_1", "HCPCS_CD_1",
    "SP_STATE_CODE", "BENE_COUNTY_CD", "LINE_ICD9_DGNS_CD_1",
    "ICD9_DGNS_CD_1",
]
cols_to_drop = [c for c in DROP_COLS if c in df.columns]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped columns: {cols_to_drop}")

PAYMENT_COLS = [
    "LINE_NCH_PMT_AMT_1", "LINE_BENE_PTB_DDCTBL_AMT_1",
    "LINE_COINSRNC_AMT_1", "MEDREIMB_IP", "BENRES_IP", "PPPYMT_IP",
    "MEDREIMB_OP", "BENRES_OP", "PPPYMT_OP", "MEDREIMB_CAR",
    "BENRES_CAR", "PPPYMT_CAR",
]
for col in PAYMENT_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

CAT_COLS = ["BENE_SEX_IDENT_CD", "BENE_RACE_CD", "BENE_ESRD_IND",
            "LINE_PRCSG_IND_CD_1", "PLAN_CVRG_MOS_NUM"]
for col in CAT_COLS:
    if col in df.columns:
        df[col] = df[col].astype("category")

for col in PAYMENT_COLS:
    if col in df.columns:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

if "bene_age" in df.columns:
    df = df[(df["bene_age"] >= 0) & (df["bene_age"] <= 120)]
    print(f"Rows after age filter: {df.shape[0]}")

print("Data cleaning complete\n")


# ==========================================
# PREPROCESSING
# ==========================================
print("\n========== MISSING VALUE HANDLING ==========")
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
print(missing_pct[missing_pct > 0].head(10))

high_missing = missing_pct[missing_pct > 50].index.tolist()
if high_missing:
    df.drop(columns=high_missing, inplace=True)
    print(f"Dropped high-missing columns: {high_missing}")

num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include=["category", "object"]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")

OHE_COLS = ["BENE_SEX_IDENT_CD", "BENE_RACE_CD", "BENE_ESRD_IND"]
existing_ohe = [c for c in OHE_COLS if c in df.columns]
if existing_ohe:
    df = pd.get_dummies(df, columns=existing_ohe, drop_first=True, dummy_na=False)
    print(f"OneHotEncoded: {existing_ohe}")

SCALE_COLS = [
    "bene_age", "BENE_HI_CVRAGE_TOT_MONS", "BENE_SMI_CVRAGE_TOT_MONS",
    "BENE_HMO_CVRAGE_TOT_MONS",
] + [c for c in PAYMENT_COLS if c in df.columns]
SCALE_COLS = [c for c in SCALE_COLS if c in df.columns]

if SCALE_COLS:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[[f"{c}_scaled" for c in SCALE_COLS]] = scaler.fit_transform(df[SCALE_COLS])
    print(f"Scaled columns: {SCALE_COLS}")

print("Preprocessing complete\n")


# ==========================================
# FEATURE ENGINEERING
# ==========================================
print("\n========== FEATURE ENGINEERING ==========")

existing_chron = [c for c in CHRONIC_CONDITION_COLS if c in df.columns]
if existing_chron:
    df["TOTAL_CHRONIC"] = df[existing_chron].sum(axis=1)
    print(f"Created TOTAL_CHRONIC from {len(existing_chron)} conditions")

COST_GROUPS = {
    "TOTAL_COST_IP": [c for c in ["MEDREIMB_IP", "BENRES_IP", "PPPYMT_IP"] if c in df.columns],
    "TOTAL_COST_OP": [c for c in ["MEDREIMB_OP", "BENRES_OP", "PPPYMT_OP"] if c in df.columns],
    "TOTAL_COST_CAR": [c for c in ["MEDREIMB_CAR", "BENRES_CAR", "PPPYMT_CAR"] if c in df.columns],
}
for name, cols in COST_GROUPS.items():
    if cols:
        df[name] = df[cols].sum(axis=1)
        print(f"Created {name} from {cols}")

if "PLAN_CVRG_MOS_NUM" in df.columns:
    df["FULL_YEAR_COVERAGE"] = (df["PLAN_CVRG_MOS_NUM"] >= 12).astype(int)

print("Feature engineering complete\n")


# ==========================================
# EDA
# ==========================================
print("\n========== EXPLORATORY DATA ANALYSIS ==========")

if "MEDREIMB_IP" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df["MEDREIMB_IP"].hist(bins=50, ax=axes[0])
    axes[0].set_title("Inpatient Reimbursement Distribution")
    df["MEDREIMB_IP"].pipe(np.log1p).hist(bins=50, ax=axes[1])
    axes[1].set_title("Log(Inpatient Reimbursement + 1)")
    plt.tight_layout()
    plt.show()

if existing_chron:
    prev = df[existing_chron].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    prev.plot(kind="bar")
    plt.title("Chronic Condition Prevalence")
    plt.ylabel("Proportion of Beneficiaries")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if "bene_age" in df.columns:
    plt.figure(figsize=(8, 4))
    df["bene_age"].hist(bins=30, edgecolor="black")
    plt.title("Beneficiary Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

if "bene_age" in df.columns and "TOTAL_COST_IP" in df.columns:
    df["age_group"] = pd.cut(df["bene_age"], bins=[0, 50, 60, 70, 80, 100],
                             labels=["<50", "50-59", "60-69", "70-79", "80+"])
    cost_by_age = df.groupby("age_group", observed=True)["TOTAL_COST_IP"].mean()
    cost_by_age.plot(kind="bar", figsize=(8, 4))
    plt.title("Mean Inpatient Cost by Age Group")
    plt.ylabel("Mean Cost ($)")
    plt.tight_layout()
    plt.show()
    df.drop(columns=["age_group"], inplace=True)

print("EDA complete\n")


# ==========================================
# MODELING  —  XGBoost with log-transformed target
# ==========================================
print("\n========== MODELING (XGBoost) ==========")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

TARGET = "MEDREIMB_IP"
if TARGET not in df.columns:
    TARGET = next((c for c in ["TOTAL_COST_IP", "MEDREIMB_OP", "MEDREIMB_CAR"]
                   if c in df.columns), None)

if TARGET is None:
    print("No suitable target column found. Exiting.")
    raise SystemExit(0)

# Separate zero vs non-zero for possible two-stage
zero_mask = df[TARGET] == 0
print(f"Zero-cost beneficiaries: {zero_mask.sum():,} ({zero_mask.mean():.1%})")

LEAKY_COLS = ["MEDREIMB_OP", "MEDREIMB_CAR", "BENRES_IP", "BENRES_OP", "BENRES_CAR",
              "PPPYMT_IP", "PPPYMT_OP", "PPPYMT_CAR", "TOTAL_COST_IP", "TOTAL_COST_OP", "TOTAL_COST_CAR"]
feature_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c != TARGET and c not in LEAKY_COLS]

if len(feature_cols) <= 1:
    print("Not enough features. Exiting.")
    raise SystemExit(0)

X = df[feature_cols].dropna()
y = df.loc[X.index, TARGET]

# Log-transform target
y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# --- Baseline: Linear Regression on log-scale ---
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr_log = lr.predict(X_test)
y_pred_lr = np.expm1(y_pred_lr_log)

print(f"\nLinear Regression (log target) — "
      f"MAE: {mean_absolute_error(np.expm1(y_test), y_pred_lr):.2f}, "
      f"RMSE: {np.sqrt(mean_squared_error(np.expm1(y_test), y_pred_lr)):.2f}, "
      f"R²: {r2_score(np.expm1(y_test), y_pred_lr):.4f}")

# --- XGBoost ---
try:
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }

    evals = [(dtrain, "train"), (dtest, "eval")]
    xgb_model = xgb.train(
        params, dtrain, num_boost_round=500,
        evals=evals, early_stopping_rounds=20,
        verbose_eval=False
    )

    y_pred_xgb_log = xgb_model.predict(dtest)
    y_pred_xgb = np.expm1(y_pred_xgb_log)

    print(f"\nXGBoost (log target) — "
          f"MAE: {mean_absolute_error(np.expm1(y_test), y_pred_xgb):.2f}, "
          f"RMSE: {np.sqrt(mean_squared_error(np.expm1(y_test), y_pred_xgb)):.2f}, "
          f"R²: {r2_score(np.expm1(y_test), y_pred_xgb):.4f}")

    # Feature importance
    importance = pd.Series(
        xgb_model.get_score(importance_type="weight"),
        index=feature_cols
    ).sort_values(ascending=False)
    print("\nTop 10 Feature Importances (XGBoost):")
    print(importance.head(10))

    importance.head(10).plot(kind="barh", figsize=(8, 5))
    plt.title("Top 10 Feature Importances (XGBoost)")
    plt.tight_layout()
    plt.show()

    # Actual vs Predicted scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(np.expm1(y_test), y_pred_xgb, alpha=0.3, s=10)
    ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], "r--", lw=1)
    ax.set_xlabel("Actual Inpatient Cost")
    ax.set_ylabel("Predicted Inpatient Cost")
    ax.set_title("XGBoost: Actual vs Predicted")
    plt.tight_layout()
    plt.show()

except ImportError:
    print("\n[WARN] xgboost not installed. Install with: pip install xgboost")
    print("Falling back to Linear Regression only.")

print(f"\nTotal runtime: {datetime.now() - start_time}")
