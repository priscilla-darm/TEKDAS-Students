"""Audit dan persiapan Coffee Shop Sales.
Tidak mengubah Excel sumber dan tidak menghapus transaksi otomatis.
Jalankan: python audit_prepare_coffee.py
Opsional: python audit_prepare_coffee.py --input "Coffee Shop Sales.xlsx" --output data_prepared
"""
from pathlib import Path
import argparse
import json
import pandas as pd

COLUMNS = [
    "transaction_id", "transaction_date", "transaction_time", "transaction_qty",
    "store_id", "store_location", "product_id", "unit_price",
    "product_category", "product_type", "product_detail",
]
TEXT = ["store_location", "product_category", "product_type", "product_detail"]
NUMERIC = ["transaction_id", "transaction_qty", "store_id", "product_id", "unit_price"]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "Coffee Shop Sales.xlsx")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data_prepared")
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"File sumber tidak ditemukan: {source}")
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(source, sheet_name="Transactions")
    report = {
        "source_file": source.name, "sheet": "Transactions",
        "original_rows": int(len(df)), "original_columns": list(df.columns),
        "missing_before": {str(k): int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows_before": int(df.duplicated().sum()), "trimmed_values": {},
        "issues": [],
    }
    issues = report["issues"]
    missing_cols = sorted(set(COLUMNS) - set(df.columns))
    if missing_cols:
        issues.append(f"Kolom wajib tidak tersedia: {missing_cols}")
    if df.empty:
        issues.append("Dataset tidak memiliki baris.")
    if issues:
        report["status"] = "PERLU PEMERIKSAAN"
        (output / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        raise ValueError("; ".join(issues))

    for col in TEXT:
        before = df[col].astype("string")
        after = before.str.strip()
        report["trimmed_values"][col] = int(before.ne(after).fillna(False).sum())
        df[col] = after
        if df[col].isna().any() or df[col].eq("").any():
            issues.append(f"{col}: teks kosong atau hilang")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            issues.append(f"{col}: angka tidak valid atau hilang")
        vals = df[col].dropna()
        if not np_is_finite(vals):
            issues.append(f"{col}: angka tak hingga")
        if (vals <= 0).any():
            issues.append(f"{col}: angka nol atau negatif")
        if col != "unit_price" and (vals % 1 != 0).any():
            issues.append(f"{col}: seharusnya bilangan bulat")
    dates = pd.to_datetime(df["transaction_date"], errors="coerce")
    if dates.isna().any():
        issues.append("transaction_date: tanggal tidak valid atau hilang")
    elif dates.ne(dates.dt.normalize()).any():
        issues.append("transaction_date mengandung jam.")
    df["transaction_date"] = dates
    times = pd.to_datetime(df["transaction_time"].astype("string"), format="%H:%M:%S", errors="coerce")
    if times.isna().any():
        issues.append("transaction_time: waktu tidak valid, harus HH:MM:SS")
    report["duplicate_transaction_ids"] = int(df["transaction_id"].duplicated().sum())
    report["duplicate_rows_after_trim"] = int(df.duplicated().sum())
    if report["duplicate_transaction_ids"]:
        issues.append("transaction_id berulang: periksa grain sebelum lanjut")
    if report["duplicate_rows_after_trim"]:
        issues.append("Ada baris duplikat setelah trim, perlu diperiksa")
    for key, cols in {"store_id": ["store_location"], "product_id": ["product_category", "product_type", "product_detail"]}.items():
        if (df.groupby(key)[cols].nunique() > 1).any().any():
            issues.append(f"Pemetaan {key} ke atribut tidak konsisten")
    report["products_with_multiple_prices"] = int(df.groupby("product_id")["unit_price"].nunique().gt(1).sum())
    report["price_note"] = "Harga berbeda dipertahankan dan tidak otomatis dianggap kesalahan."
    report["status"] = "PERLU PEMERIKSAAN" if issues else "LULUS AUDIT DASAR"
    if issues:
        (output / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        raise ValueError("Audit belum lulus; periksa audit_report.json. Transaksi tidak dihapus.")

    for col in NUMERIC:
        if col != "unit_price":
            df[col] = df[col].astype("int64")
    df["transaction_time"] = times.dt.strftime("%H:%M:%S")
    df["transaction_datetime"] = df["transaction_date"] + pd.to_timedelta(df["transaction_time"])
    df["sales"] = (df["transaction_qty"] * df["unit_price"]).round(2)
    df["month"] = df["transaction_date"].dt.strftime("%Y-%m")
    df["hour"] = df["transaction_datetime"].dt.hour
    df["day_of_week"] = df["transaction_date"].dt.dayofweek
    df["day_name_id"] = df["day_of_week"].map({0:"Senin",1:"Selasa",2:"Rabu",3:"Kamis",4:"Jumat",5:"Sabtu",6:"Minggu"})
    assert len(df) == int(report["original_rows"]), "Jumlah baris berubah tanpa izin"
    report.update({
        "prepared_rows": int(len(df)),
        "date_min": str(df["transaction_date"].min().date()),
        "date_max": str(df["transaction_date"].max().date()),
        "observed_days": int(df["transaction_date"].dt.normalize().nunique()),
        "total_sales": round(float(df["sales"].sum()), 2),
        "total_transactions": int(df["transaction_id"].nunique()),
        "total_units": int(df["transaction_qty"].sum()),
        "average_recorded_transaction_value": round(float(df["sales"].sum()/df["transaction_id"].nunique()), 6),
        "currency": "Tidak dinyatakan pada file sumber",
        "grain": "Satu baris per transaction_id dan satu produk; belum dipastikan sebagai seluruh isi struk.",
        "sales_reconciliation_passed": bool(round(float(df["sales"].sum()), 2) == round(float((df["transaction_qty"] * df["unit_price"]).sum()), 2)),
    })
    monthly = df.groupby("month", as_index=False).agg(
        total_sales=("sales","sum"), total_transactions=("transaction_id","nunique"), total_units=("transaction_qty","sum"))
    months = pd.period_range(df["transaction_date"].min().to_period("M"), df["transaction_date"].max().to_period("M"), freq="M").astype(str)
    monthly = monthly.set_index("month").reindex(months).rename_axis("month").reset_index()
    previous = monthly["total_sales"].shift(1)
    monthly["growth_pct"] = ((monthly["total_sales"] - previous) / previous * 100).where(previous.notna() & previous.ne(0))
    monthly["average_transaction_value"] = monthly["total_sales"] / monthly["total_transactions"]
    df.to_csv(output / "coffee_sales_prepared.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    monthly.to_csv(output / "monthly_summary.csv", index=False, encoding="utf-8-sig")
    (output / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("LULUS AUDIT DASAR")
    print(f"Baris tetap: {len(df):,}; total penjualan: {report['total_sales']:,.2f}; unit: {report['total_units']:,}")
    print("Output:", output)

def np_is_finite(values):
    import numpy as np
    return bool(np.isfinite(values.astype(float)).all())

if __name__ == "__main__":
    main()
