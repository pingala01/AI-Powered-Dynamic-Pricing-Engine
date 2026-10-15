"""CSV loading and validation."""
import pandas as pd

REQUIRED_COLUMNS = [
    "product_id", "product_name", "date", "current_price", "units_sold",
    "inventory_level", "competitor_price", "seasonal_factor",
]
NUMERIC_COLUMNS = ["current_price", "units_sold", "inventory_level", "competitor_price", "seasonal_factor"]
MIN_DAYS_PER_PRODUCT = 21


class DataValidationError(Exception):
    """Raised with a clear, user-facing message when the CSV is not usable."""


def validate_and_clean(df: pd.DataFrame):
    """Returns (clean_dataframe, list_of_warnings). Raises DataValidationError on fatal problems."""
    warnings = []
    if df is None or df.empty:
        raise DataValidationError("The file is empty. Please upload a CSV with data rows.")

    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Required columns are: {', '.join(REQUIRED_COLUMNS)}."
        )
    df = df[REQUIRED_COLUMNS].copy()

    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["product_name"] = df["product_name"].astype(str).str.strip()

    parsed = pd.to_datetime(df["date"], errors="coerce")
    if int(parsed.isna().sum()) == len(df):
        raise DataValidationError("No valid dates found in the 'date' column (use e.g. 2026-06-01).")
    df["date"] = parsed.dt.normalize()

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["date"] + NUMERIC_COLUMNS)
    df = df[df["product_id"].ne("") & df["product_id"].ne("nan")]
    dropped = before - len(df)
    if len(df) == 0:
        raise DataValidationError("All rows contain invalid or missing values. Check dates and numeric columns.")
    if dropped:
        warnings.append(f"Removed {dropped} row(s) with missing or non-numeric values.")

    for col in ["current_price", "competitor_price"]:
        if (df[col] <= 0).any():
            raise DataValidationError(f"Column '{col}' must contain only values greater than 0.")
    for col in ["units_sold", "inventory_level"]:
        if (df[col] < 0).any():
            raise DataValidationError(f"Column '{col}' must not contain negative values.")
    if (df["seasonal_factor"] <= 0).any():
        raise DataValidationError("Column 'seasonal_factor' must contain only values greater than 0.")

    dup = int(df.duplicated(subset=["product_id", "date"]).sum())
    if dup:
        df = df.drop_duplicates(subset=["product_id", "date"], keep="last")
        warnings.append(f"Removed {dup} duplicate product/date row(s) (kept the last one).")

    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    counts = df.groupby("product_id")["date"].nunique()
    short = counts[counts < MIN_DAYS_PER_PRODUCT]
    if len(short) == len(counts):
        raise DataValidationError(
            f"Every product has fewer than {MIN_DAYS_PER_PRODUCT} days of history. "
            "Please provide more daily history."
        )
    if len(short):
        warnings.append(
            f"{len(short)} product(s) have fewer than {MIN_DAYS_PER_PRODUCT} days of history and were "
            f"excluded: {', '.join(short.index.astype(str))}."
        )
        df = df[~df["product_id"].isin(short.index)].reset_index(drop=True)
    return df, warnings


def load_csv(source):
    """source: file path or file-like object (e.g. Streamlit upload)."""
    try:
        raw = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        raise DataValidationError("The file is empty.") from None
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Could not read the file as a CSV: {exc}") from None
    return validate_and_clean(raw)
