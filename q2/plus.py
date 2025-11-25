import os
import sys
import atexit
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import FancyArrowPatch
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = True
plt.rcParams['mathtext.fontset'] = 'dejavusans'

sns.set_theme(style='whitegrid', context='talk')

os.makedirs('figure', exist_ok=True)
os.makedirs('chart', exist_ok=True)

class Tee:
    def __init__(self, filename):
        self.f = open(filename, 'w', encoding='utf-8')
        self._stdout = sys.stdout
    def write(self, s):
        self._stdout.write(s)
        self.f.write(s)
    def flush(self):
        try:
            self._stdout.flush()
        except Exception:
            pass
        try:
            if not self.f.closed:
                self.f.flush()
        except Exception:
            pass

tee = Tee('chart/plus_output.txt')
sys.stdout = tee
atexit.register(lambda: tee.f.close())

def load_trade():
    df = pd.read_excel('downloaded data/美国进口数据.xlsx')
    if 'freqCode' in df.columns:
        if (df['freqCode'] == 'M').any():
            df = df[df['freqCode'] == 'M'].copy()
    if 'qty' in df.columns and 'quantity' not in df.columns:
        df = df.rename(columns={'qty': 'quantity'})
    if 'primaryValue' in df.columns and 'TradeValue(US$)' not in df.columns:
        df = df.rename(columns={'primaryValue': 'TradeValue(US$)'})
    if {'refYear', 'refMonth'}.issubset(df.columns):
        df['date'] = pd.to_datetime(
            df['refYear'].astype(int).astype(str) + '-' +
            df['refMonth'].astype(int).astype(str).str.zfill(2) + '-01'
        )
    elif 'period' in df.columns:
        df['date'] = pd.to_datetime(df['period'].astype(str).str.slice(0, 6), format='%Y%m')
    else:
        raise ValueError('无法识别时间列')
    return df

def load_tariff():
    df = pd.read_excel('downloaded data/美日汽车关税数据.xlsx')
    df['date'] = pd.to_datetime(df['Month'].astype(str) + '-01')
    return df

def get_policy_change_date(tariff_df):
    baseline = float(tariff_df['US Tariff on Japan Autos'].min())
    change_rows = tariff_df[tariff_df['US Tariff on Japan Autos'] != baseline]
    d = pd.to_datetime('2025-04-01') if len(change_rows) == 0 else change_rows['date'].min()
    return d

def fig_did_framework(trade_df, tariff_df):
    d = get_policy_change_date(tariff_df)
    sub = trade_df[(trade_df['reporterDesc'] == 'USA') & (trade_df['flowDesc'] == 'Import') & (trade_df['partnerDesc'].isin(['Japan', 'Germany', 'Rep. of Korea']))].copy()
    gp = sub.groupby(['date', 'partnerDesc'], as_index=False)['quantity'].sum()
    jp = gp[gp['partnerDesc'] == 'Japan'][['date', 'quantity']].rename(columns={'quantity': 'Q_JP'})
    de = gp[gp['partnerDesc'] == 'Germany'][['date', 'quantity']].rename(columns={'quantity': 'Q_DE'})
    kr = gp[gp['partnerDesc'] == 'Rep. of Korea'][['date', 'quantity']].rename(columns={'quantity': 'Q_KR'})
    ctrl = pd.merge(pd.merge(jp[['date']], de, on='date', how='left'), kr, on='date', how='left').fillna(0)
    ctrl['Q_CTRL'] = ctrl['Q_DE'] + ctrl['Q_KR']
    df = pd.merge(jp, ctrl[['date', 'Q_CTRL']], on='date', how='left').sort_values('date')
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['Q_JP'], label='Japan (Treatment)', color='#4C78A8', linewidth=2.4)
    plt.plot(df['date'], df['Q_CTRL'], label='Germany+Korea (Control)', color='#E45756', linewidth=2.2)
    plt.axvline(d, color='gray', linestyle='--', alpha=0.6)
    plt.title('DID Identification Framework: Trade Scale', fontsize=13)
    plt.xlabel('Date')
    plt.ylabel('Vehicles')
    plt.legend(loc='upper left', fontsize=10)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    path = 'figure/did_framework.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print('保存图表:', path)

def prepare_japan_features(trade_df, tariff_df):
    baseline = float(tariff_df['US Tariff on Japan Autos'].min())
    d = get_policy_change_date(tariff_df)
    us_jp = trade_df[(trade_df['reporterDesc'] == 'USA') & (trade_df['partnerDesc'] == 'Japan') & (trade_df['flowDesc'] == 'Import')].copy()
    us_jp = us_jp.groupby('date', as_index=False)[['quantity','TradeValue(US$)']].sum().sort_values('date')
    tmp = tariff_df[['date', 'US Tariff on Japan Autos']].copy()
    us_jp = us_jp.merge(tmp, on='date', how='left')
    us_jp['tariff_rate'] = us_jp['US Tariff on Japan Autos'].fillna(baseline)
    us_jp = us_jp.drop(columns=['US Tariff on Japan Autos'])
    us_jp['unit_price'] = us_jp['TradeValue(US$)'] / us_jp['quantity']
    us_jp['year'] = us_jp['date'].dt.year
    us_jp['month'] = us_jp['date'].dt.month
    us_jp['quarter'] = us_jp['date'].dt.quarter
    us_jp['time_index'] = range(len(us_jp))
    us_jp['is_Q1'] = (us_jp['quarter'] == 1).astype(int)
    us_jp['is_Q2'] = (us_jp['quarter'] == 2).astype(int)
    us_jp['is_Q3'] = (us_jp['quarter'] == 3).astype(int)
    us_jp['is_Q4'] = (us_jp['quarter'] == 4).astype(int)
    us_jp['covid_period'] = ((us_jp['date'] >= '2020-03-01') & (us_jp['date'] <= '2020-06-30')).astype(int)
    us_jp['post_policy'] = (us_jp['date'] >= d).astype(int)
    us_jp['ln_quantity'] = np.log(us_jp['quantity'] + 1)
    us_jp['ln_trade_value'] = np.log(us_jp['TradeValue(US$)'] + 1)
    us_jp['ln_unit_price'] = np.log(us_jp['unit_price'] + 1)
    us_jp['ln_tariff'] = np.log(1 + us_jp['tariff_rate'])
    us_jp['sin_m1'] = np.sin(2 * np.pi * us_jp['month'] / 12)
    us_jp['cos_m1'] = np.cos(2 * np.pi * us_jp['month'] / 12)
    us_jp['sin_m2'] = np.sin(4 * np.pi * us_jp['month'] / 12)
    us_jp['cos_m2'] = np.cos(4 * np.pi * us_jp['month'] / 12)
    for lag in [1, 2, 3]:
        us_jp[f'ln_quantity_lag{lag}'] = us_jp['ln_quantity'].shift(lag)
        us_jp[f'quantity_lag{lag}'] = us_jp['quantity'].shift(lag)
    us_jp['quantity_ma3'] = us_jp['quantity'].rolling(window=3, min_periods=1).mean()
    us_jp['quantity_ma6'] = us_jp['quantity'].rolling(window=6, min_periods=1).mean()
    us_jp['quantity_growth'] = us_jp['quantity'].pct_change()
    us_jp['ln_tariff_post'] = us_jp['ln_tariff'] * us_jp['post_policy']
    us_jp = us_jp.dropna().reset_index(drop=True)
    feature_list = ['year', 'month', 'quarter', 'time_index', 'is_Q1', 'is_Q2', 'is_Q3', 'is_Q4', 'tariff_rate', 'ln_tariff', 'ln_quantity_lag1', 'ln_quantity_lag2', 'ln_quantity_lag3', 'quantity_lag1', 'quantity_lag2', 'quantity_lag3', 'quantity_ma3', 'quantity_ma6', 'quantity_growth', 'covid_period', 'post_policy', 'sin_m1', 'cos_m1', 'sin_m2', 'cos_m2']
    X = us_jp[feature_list].copy()
    y = us_jp['quantity'].copy()
    X['tariff_x_time'] = X['tariff_rate'] * X['time_index']
    X['tariff_x_post'] = X['tariff_rate'] * X['post_policy']
    return X, y

def fig_rf_importance(trade_df, tariff_df):
    X, y = prepare_japan_features(trade_df, tariff_df)
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=42)
    rf.fit(X, y)
    imp = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
    top = imp.head(12)
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette('Set2', n_colors=len(top))
    plt.bar(range(len(top)), top['importance'], color=colors)
    plt.xticks(range(len(top)), top['feature'], rotation=35, ha='right')
    plt.ylabel('MDI')
    plt.title('Random Forest Feature Importance (MDI)', fontsize=13)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = 'figure/rf_importance_plus.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print('保存图表:', path)

def fig_stacking_structure():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    b1 = FancyBboxPatch((0.1, 0.65), 0.25, 0.15, boxstyle='round,pad=0.02', fc='#4C78A8', ec='none', alpha=0.85)
    b2 = FancyBboxPatch((0.1, 0.40), 0.25, 0.15, boxstyle='round,pad=0.02', fc='#72B7B2', ec='none', alpha=0.85)
    b3 = FancyBboxPatch((0.1, 0.15), 0.25, 0.15, boxstyle='round,pad=0.02', fc='#F58518', ec='none', alpha=0.85)
    ax.add_patch(b1)
    ax.add_patch(b2)
    ax.add_patch(b3)
    ax.text(0.225, 0.725, 'Gravity', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.225, 0.475, 'Dynamic Panel', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.225, 0.225, 'Random Forest', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    meta = FancyBboxPatch((0.45, 0.40), 0.28, 0.20, boxstyle='round,pad=0.02', fc='#E45756', ec='none', alpha=0.85)
    ax.add_patch(meta)
    ax.text(0.59, 0.50, 'Linear Regression\n(Meta Model)', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    out = FancyBboxPatch((0.80, 0.43), 0.15, 0.14, boxstyle='round,pad=0.02', fc='#54A24B', ec='none', alpha=0.85)
    ax.add_patch(out)
    ax.text(0.875, 0.50, 'Ensemble\nPrediction', color='white', ha='center', va='center', fontsize=11, fontweight='bold')
    for y in [0.725, 0.475, 0.225]:
        arr = FancyArrowPatch((0.35, y), (0.45, 0.50), arrowstyle='->', mutation_scale=15, lw=2, color='#555555')
        ax.add_patch(arr)
    arr2 = FancyArrowPatch((0.73, 0.50), (0.80, 0.50), arrowstyle='->', mutation_scale=15, lw=2, color='#555555')
    ax.add_patch(arr2)
    ax.text(0.5, 0.08, 'Stacking Ensemble Structure', ha='center', va='center', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = 'figure/stacking_structure.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print('保存图表:', path)

def main():
    print('=' * 80)
    print('Plus 可视化生成')
    print('=' * 80)
    trade_df = load_trade()
    tariff_df = load_tariff()
    fig_did_framework(trade_df, tariff_df)
    fig_rf_importance(trade_df, tariff_df)
    fig_stacking_structure()
    print('=' * 80)
    print('完成')
    print('=' * 80)

if __name__ == '__main__':
    main()
