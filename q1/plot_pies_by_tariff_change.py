import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = 'figures'
DATA_DIR = 'preprocessed_data'
TARIFF_CSV = os.path.join('data', 'tariff.csv')
os.makedirs(FIG_DIR, exist_ok=True)

frames = []
for name in ['train_data.csv', 'val_data.csv', 'test_data.csv']:
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        frames.append(pd.read_csv(path, parse_dates=['Date']))
if not frames:
    raise FileNotFoundError('No preprocessed_data CSVs found')
df = pd.concat(frames, ignore_index=True).sort_values('Date')

if not os.path.exists(TARIFF_CSV):
    raise FileNotFoundError('data/tariff.csv not found')
tar = pd.read_csv(TARIFF_CSV, header=None, names=['Month','Rate'])
if not tar['Month'].astype(str).str.match(r'^\d{4}-\d{2}$').all():
    tar = pd.read_csv(TARIFF_CSV)
    if 'Month' not in tar.columns or 'Rate' not in tar.columns:
        tar.columns = ['Month','Rate']
tar['Month'] = tar['Month'].astype(str).str.strip()
tar['Date'] = pd.to_datetime(tar['Month'] + '-01', format='%Y-%m-%d')
tar = tar.sort_values('Date')
tar['RateDiff'] = tar['Rate'].diff()

labels = ['US','AR','BR']

def pie_from_row(row, title, fname):
    shares = [row.get('US_Export_Share', np.nan),
              row.get('AR_Export_Share', np.nan),
              row.get('BR_Export_Share', np.nan)]
    if any(pd.isna(x) for x in shares):
        return False
    s = float(np.sum(shares))
    if s <= 0:
        return False
    shares = np.array(shares, dtype=np.float32) / s
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.pie(shares, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname))
    plt.close(fig)
    return True

targets = [3, 28, 87]
saved = []
for rate in targets:
    rows = tar[(tar['Rate'] == rate) & (tar['RateDiff'].fillna(0) != 0)]
    for _, r in rows.iterrows():
        change_month = r['Date'].strftime('%Y-%m')
        prev_date = r['Date'] - pd.offsets.MonthBegin(1)
        next_date = r['Date'] + pd.offsets.MonthBegin(1)
        prev_row = df[df['Date'] == prev_date]
        next_row = df[df['Date'] == next_date]
        if not prev_row.empty:
            ok1 = pie_from_row(prev_row.iloc[0], f'Quantity Share (Prev rate {rate}% {change_month})', f'pie_prev_rate_{rate}_{change_month}.png')
        else:
            ok1 = False
        if not next_row.empty:
            ok2 = pie_from_row(next_row.iloc[0], f'Quantity Share (Next rate {rate}% {change_month})', f'pie_next_rate_{rate}_{change_month}.png')
        else:
            ok2 = False
        if ok1 or ok2:
            saved.append((rate, change_month))

 

# extra: generate pie for a specific month if available
try:
    target_month = pd.Timestamp('2018-01-01')
    row = df[df['Date'] == target_month]
    if not row.empty:
        ok = pie_from_row(row.iloc[0], 'Quantity Share (2018-01)', 'pie_2018_01.png')
        if ok:
            pass
    else:
        pass
except Exception as e:
    pass
