import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / 'data'
out_dir = base_dir / 'visualize'
out_dir.mkdir(parents=True, exist_ok=True)

def read_all_trade_files(path: Path):
    files = [p for p in path.glob('*.xlsx') if p.name != 'tariff_data.xlsx']
    frames = []
    for p in files:
        df = pd.read_excel(p)
        cols = [c for c in ['refPeriodId','reporterDesc','flowDesc','partnerDesc','cmdCode','cmdDesc','qtyUnitAbbr','qty','primaryValue'] if c in df.columns]
        frames.append(df[cols].copy())
    return pd.concat(frames, ignore_index=True)

df = read_all_trade_files(data_dir)
df['date'] = pd.to_datetime(df['refPeriodId'].astype(str), format='%Y%m%d')

imp = df[(df['reporterDesc']=='USA') & (df['partnerDesc']=='China') & (df['flowDesc']=='Import')].copy()
exp = df[(df['reporterDesc']=='USA') & (df['partnerDesc']=='China') & (df['flowDesc']=='Export')].copy()

imp_m = imp.groupby('date', as_index=False)['primaryValue'].sum()
exp_m = exp.groupby('date', as_index=False)['primaryValue'].sum()

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(imp_m['date'], imp_m['primaryValue']/1e9, color='black', label='US Imports from China')
ax.plot(exp_m['date'], exp_m['primaryValue']/1e9, color='steelblue', label='US Exports to China')
ax.set_ylabel('Value (Billion USD)')
ax.set_title('US-China Chip Trade: Imports and Exports')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(out_dir / 'trade_overview.png', dpi=300, bbox_inches='tight')
plt.close()

imp_uv = imp.copy()
qty_safe = np.where(imp_uv['qty'].fillna(0)>0, imp_uv['qty'].fillna(0), np.nan)
imp_uv['unit_value'] = imp_uv['primaryValue']/qty_safe
imp_uv = imp_uv.dropna(subset=['unit_value'])
fig, ax = plt.subplots(figsize=(10,6))
ax.hist(imp_uv['unit_value'], bins=50, color='gray', alpha=0.8)
ax.set_xlabel('Unit Value (USD per unit)')
ax.set_title('Distribution of Unit Values (US Imports from China)')
plt.tight_layout()
plt.savefig(out_dir / 'unit_value_hist.png', dpi=300, bbox_inches='tight')
plt.close()

tar = pd.read_excel(data_dir / 'tariff_data.xlsx')
tar['hts8'] = tar['hts8'].astype(str).str.zfill(8)
min_date = pd.to_datetime('2020-01-01')
max_date = pd.to_datetime('2025-08-31')
all_dates = pd.date_range(min_date, max_date, freq='MS')
rows = []
for _, r in tar.iterrows():
    begin = pd.to_datetime(r['begin_effect_date']) if pd.notna(r['begin_effect_date']) else min_date
    end = pd.to_datetime(r['end_effective_date']) if pd.notna(r['end_effective_date']) else max_date
    val = str(r.get('非WTO成员国', '0')).strip()
    if '免税' in val or val=='Free':
        rate = 0.0
    elif '%' in val:
        try:
            rate = float(val.replace('%','').strip())/100.0
        except:
            rate = 0.0
    else:
        rate = 0.05
    for d in all_dates:
        if begin <= d <= end:
            rows.append({'date': d, 'rate': rate})
tar_ts = pd.DataFrame(rows)
tar_month = tar_ts.groupby('date', as_index=False)['rate'].mean()
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(tar_month['date'], tar_month['rate']*100, color='steelblue', linewidth=2, label='Average Tariff Rate')
ax.set_ylabel('Tariff Rate (%)')
ax.set_title('Tariff Rate Over Time')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(out_dir / 'tariff_rate_over_time.png', dpi=300, bbox_inches='tight')
plt.close()
