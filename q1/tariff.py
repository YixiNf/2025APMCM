import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

FIG_DIR = 'figures'
DATA_CSV = os.path.join('data', 'tariff.csv')
os.makedirs(FIG_DIR, exist_ok=True)

def read_tariff(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    try:
        tar = pd.read_csv(csv_path, header=None, names=['Month','Rate'])
        if not tar['Month'].astype(str).str.match(r'^\d{4}-\d{2}$').all():
            raise ValueError('format')
    except Exception:
        tar = pd.read_csv(csv_path)
        if 'Month' not in tar.columns or 'Rate' not in tar.columns:
            tar.columns = ['Month','Rate']
    tar['Month'] = tar['Month'].astype(str).str.strip()
    tar['Date'] = pd.to_datetime(tar['Month'] + '-01', format='%Y-%m-%d')
    tar = tar.sort_values('Date')
    # forward-fill monthly steps across the full covered range
    full = pd.DataFrame({'Date': pd.date_range(tar['Date'].min(), tar['Date'].max(), freq='MS')})
    tar = full.merge(tar[['Date','Rate']], on='Date', how='left').ffill()
    return tar

def plot_three():
    tar = read_tariff(DATA_CSV)
    us = tar[['Date','Rate']].copy()
    ar = pd.DataFrame({'Date': us['Date'], 'Rate': np.full(len(us), 3)})
    br = pd.DataFrame({'Date': us['Date'], 'Rate': np.full(len(us), 3)})
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(us['Date'], us['Rate'], label='US', color='#1f77b4', linewidth=2.2)
    ax.plot(ar['Date'], ar['Rate'], label='AR', color='#2ca02c', linewidth=2.0, linestyle='--')
    ax.plot(br['Date'], br['Rate'], label='BR', color='#ff7f0e', linewidth=2.0, linestyle='--')
    ax.set_title('China Soybean Import Tariff by Origin', fontsize=14)
    ax.set_ylabel('Tariff Rate')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))
    ymax = max(us['Rate'].max(), ar['Rate'].max(), br['Rate'].max())
    ax.set_ylim(0, min(100, ymax * 1.15))
    fig.autofmt_xdate()
    ax.legend(loc='best', frameon=True)
    ax.grid(True, which='major', linestyle='-', alpha=0.35)
    out = os.path.join(FIG_DIR, 'tariff_three_countries.png')
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print('saved', out)

def plot_us_annotated():
    tar = read_tariff(DATA_CSV)
    tar['Diff'] = tar['Rate'].diff().fillna(0)
    changes = tar[tar['Diff'] != 0]
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(tar['Date'], tar['Rate'], label='US', color='#1f77b4', linewidth=2.2, marker='o')
    ax.set_title('US Tariff Rate with Change Annotations', fontsize=14)
    ax.set_ylabel('Tariff Rate')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))
    ymax = tar['Rate'].max()
    ax.set_ylim(0, min(100, ymax * 1.15))
    for _, r in changes.iterrows():
        ax.axvline(r['Date'], color='#1f77b4', alpha=0.15, linestyle=':')
        ax.scatter(r['Date'], r['Rate'], color='#1f77b4', s=40)
        txt = f"{int(r['Rate'])}%\n{r['Date'].strftime('%Y-%m')}"
        ax.annotate(txt, xy=(r['Date'], r['Rate']), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9, color='#1f77b4')
    fig.autofmt_xdate()
    ax.legend(loc='best', frameon=True)
    ax.grid(True, which='major', linestyle='-', alpha=0.35)
    out = os.path.join(FIG_DIR, 'tariff_us_annotated.png')
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print('saved', out)

if __name__ == '__main__':
    plot_three()
    plot_us_annotated()
