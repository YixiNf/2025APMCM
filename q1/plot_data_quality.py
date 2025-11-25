import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = 'preprocessed_data'
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

def load_all():
    tr = pd.read_csv(os.path.join(DATA_DIR, 'train_data.csv'), parse_dates=['Date'])
    va = pd.read_csv(os.path.join(DATA_DIR, 'val_data.csv'), parse_dates=['Date'])
    te = pd.read_csv(os.path.join(DATA_DIR, 'test_data.csv'), parse_dates=['Date'])
    df = pd.concat([tr, va, te], ignore_index=True)
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def num_cols(df):
    cols = []
    for c in df.columns:
        if c == 'Date':
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def plot_missing_bar(df):
    miss_pct = df.isna().mean().drop(labels=['Date'])
    miss_pct = miss_pct[miss_pct > 0].sort_values(ascending=False)
    top = miss_pct.head(20)
    plt.figure(figsize=(12, 6))
    plt.bar(top.index, top.values)
    plt.xticks(rotation=80)
    plt.ylabel('Missing Ratio')
    plt.title('Top Missing Columns (All Sets)')
    out = os.path.join(FIG_DIR, 'data_missing_bar.png')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print('saved', out)

def plot_missing_heatmap(df, cols, title, fname):
    d = df[['Date'] + cols].copy()
    m = d[cols].isna().astype(int).values
    plt.figure(figsize=(12, max(4, len(cols) * 0.4)))
    plt.imshow(m.T, aspect='auto', cmap='Reds')
    plt.yticks(range(len(cols)), cols)
    plt.xticks(range(len(d['Date'])), d['Date'].dt.strftime('%Y-%m'), rotation=90)
    plt.title(title)
    plt.xlabel('Month')
    plt.colorbar(label='Missing')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, fname)
    plt.savefig(out)
    plt.close()
    print('saved', out)

def detect_outliers(df):
    cols = num_cols(df)
    res = {}
    for c in cols:
        s = df[c].astype(float)
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            res[c] = pd.Series(False, index=df.index)
            continue
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        res[c] = (s < lo) | (s > hi)
    return res

def plot_outlier_bar(df, outliers):
    counts = {c: v.sum() for c, v in outliers.items()}
    s = pd.Series(counts).sort_values(ascending=False)
    top = s.head(20)
    plt.figure(figsize=(12, 6))
    plt.bar(top.index, top.values)
    plt.xticks(rotation=80)
    plt.ylabel('Outlier Count')
    plt.title('Top Outlier Columns (IQR)')
    out = os.path.join(FIG_DIR, 'data_outlier_bar.png')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print('saved', out)

def clean_data(df):
    d = df.copy()
    d = d.set_index('Date')
    cols = num_cols(d)
    for c in cols:
        s = d[c].astype(float)
        s = s.interpolate(method='time')
        s = s.ffill().bfill()
        p1 = s.quantile(0.01)
        p99 = s.quantile(0.99)
        s = s.clip(lower=p1, upper=p99)
        d[c] = s
    d = d.reset_index()
    return d

def plot_compare(df_raw, df_clean, cols, title, fname):
    plt.figure(figsize=(12, 5))
    for c in cols:
        plt.plot(df_raw['Date'], df_raw[c], label=f'{c}-raw', linestyle='--', alpha=0.6)
        plt.plot(df_clean['Date'], df_clean[c], label=f'{c}-clean')
    plt.legend()
    plt.title(title)
    plt.xlabel('Month')
    out = os.path.join(FIG_DIR, fname)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print('saved', out)

def main():
    df = load_all()
    plot_missing_bar(df)
    key_cols = [
        'US_Export_China_Q','AR_Export_China_Q','BR_Export_China_Q',
        'US_Export_China_V','AR_Export_China_V','BR_Export_China_V',
        'US_Import_Q','AR_Import_Q','BR_Import_Q',
        'US_Import_V','AR_Import_V','BR_Import_V',
        'US_Bilateral_Q','AR_Bilateral_Q','BR_Bilateral_Q',
        'US_Bilateral_V','AR_Bilateral_V','BR_Bilateral_V',
        'US_FOB_Price','AR_FOB_Price','BR_FOB_Price'
    ]
    key_cols = [c for c in key_cols if c in df.columns]
    if key_cols:
        plot_missing_heatmap(df, key_cols, 'Missingness Heatmap (Key Columns)', 'data_missing_heatmap.png')
    outliers = detect_outliers(df)
    plot_outlier_bar(df, outliers)
    df_clean = clean_data(df)
    cmp_cols_q = [c for c in ['US_Bilateral_Q','AR_Bilateral_Q','BR_Bilateral_Q'] if c in df.columns][:3]
    if cmp_cols_q:
        plot_compare(df, df_clean, cmp_cols_q, 'Cleaning Comparison - Quantity', 'data_clean_compare_quantity.png')
    cmp_cols_v = [c for c in ['US_Bilateral_V','AR_Bilateral_V','BR_Bilateral_V'] if c in df.columns][:3]
    if cmp_cols_v:
        plot_compare(df, df_clean, cmp_cols_v, 'Cleaning Comparison - Value', 'data_clean_compare_value.png')
    fob_cols = [c for c in ['US_FOB_Price','AR_FOB_Price','BR_FOB_Price'] if c in df.columns][:3]
    if fob_cols:
        plot_compare(df, df_clean, fob_cols, 'Cleaning Comparison - FOB Price', 'data_clean_compare_fob.png')

if __name__ == '__main__':
    main()

