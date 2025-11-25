import os
import sys
import atexit
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, PercentFormatter

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = True
plt.rcParams['mathtext.fontset'] = 'dejavusans'

sns.set_theme(style='whitegrid', context='talk')

os.makedirs('visualize', exist_ok=True)

def fmt_thousands(x, pos):
    return f"{x:,.0f}"

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

def figure_timeseries_quantity_value(df):
    sub = df[(df.get('reporterDesc') == 'USA') & (df.get('partnerDesc') == 'Japan') & (df.get('flowDesc') == 'Import')].copy()
    sub = sub.sort_values('date')
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(sub['date'], sub['quantity'], color='#4C78A8', linewidth=2.2, label='Quantity')
    ax1.fill_between(sub['date'], sub['quantity'], color='#4C78A8', alpha=0.18)
    ax1.set_ylabel('Vehicles')
    ax2 = ax1.twinx()
    ax2.plot(sub['date'], sub['TradeValue(US$)'], color='#F58518', linewidth=2.0, linestyle='--', label='Value (US$)')
    ax2.set_ylabel('Value (US$)')
    ax1.set_title('US Imports from Japan: Quantity vs Value', fontsize=13)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.grid(True, alpha=0.25)
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=10, loc='upper left')
    plt.tight_layout()
    path = 'visualize/timeseries_quantity_value.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print('保存图表:', path)

def figure_seasonality_heatmap(df):
    sub = df[(df.get('reporterDesc') == 'USA') & (df.get('partnerDesc') == 'Japan') & (df.get('flowDesc') == 'Import')].copy()
    sub['year'] = sub['date'].dt.year
    sub['month'] = sub['date'].dt.month
    pivot = sub.pivot_table(index='year', columns='month', values='quantity', aggfunc='sum')
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(pivot, cmap='YlGnBu', ax=ax, linewidths=0.5)
    ax.set_title('Seasonality Heatmap: Quantity', fontsize=14)
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    plt.tight_layout()
    path = 'visualize/seasonality_heatmap.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print('保存图表:', path)

def figure_yoy_growth(df, tariff):
    sub = df[(df.get('reporterDesc') == 'USA') & (df.get('partnerDesc') == 'Japan') & (df.get('flowDesc') == 'Import')].copy()
    sub = sub.sort_values('date')
    sub['yoy'] = sub['quantity'].pct_change(12)
    baseline = float(tariff['US Tariff on Japan Autos'].min())
    changes = tariff[tariff['US Tariff on Japan Autos'] != baseline]
    policy_date = changes['date'].min() if len(changes) > 0 else None
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sub['date'], sub['yoy'], color='#2F4B7C', linewidth=2.2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    if policy_date is not None:
        ax.axvline(policy_date, color='gray', linestyle='--', alpha=0.5)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title('YoY Growth of Quantity', fontsize=13)
    ax.set_xlabel('Date')
    ax.set_ylabel('YoY Growth')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = 'visualize/yoy_growth.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print('保存图表:', path)

def figure_partner_stackshare(df):
    sub = df[(df.get('reporterDesc') == 'USA') & (df.get('flowDesc') == 'Import')].copy()
    sub = sub.sort_values('date')
    pivot = sub.pivot_table(index='date', columns='partnerDesc', values='quantity', aggfunc='sum').fillna(0)
    totals = pivot.sum()
    top_partners = list(totals.sort_values(ascending=False).head(6).index)
    reduced = pivot[top_partners]
    others = pivot.drop(columns=top_partners).sum(axis=1)
    reduced['Others'] = others
    shares = reduced.div(reduced.sum(axis=1), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette('Set2', n_colors=len(shares.columns))
    ax.stackplot(shares.index, *[shares[c].values for c in shares.columns], labels=shares.columns, colors=colors)
    ax.set_title('US Import Share by Partner (Stacked Area)', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Share')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = 'visualize/stacked_share.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print('保存图表:', path)

def figure_quantity_distribution(df):
    sub = df[(df.get('reporterDesc') == 'USA') & (df.get('partnerDesc') == 'Japan') & (df.get('flowDesc') == 'Import')].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(sub['quantity'], bins=40, kde=True, ax=ax, color='#4C78A8')
    ax.set_title('Distribution of Monthly Quantity (Japan)', fontsize=13)
    ax.set_xlabel('Vehicles')
    ax.set_ylabel('Frequency')
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_thousands))
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = 'visualize/quantity_distribution.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print('保存图表:', path)

def figure_rolling_vs_tariff(df, tariff):
    sub = df[(df.get('reporterDesc') == 'USA') & (df.get('partnerDesc') == 'Japan') & (df.get('flowDesc') == 'Import')].copy()
    sub = sub.sort_values('date')
    sub['rolling_12m'] = sub['quantity'].rolling(12).sum()
    t = tariff.sort_values('date')
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(sub['date'], sub['rolling_12m'], color='#6F4C9B', linewidth=2.4, label='12M Rolling Quantity')
    ax1.set_ylabel('Vehicles (12M Rolling)')
    ax2 = ax1.twinx()
    ax2.plot(t['date'], t['US Tariff on Japan Autos'], color='#E45756', linewidth=2.0, linestyle='--', label='Tariff Rate')
    ax2.set_ylabel('Tariff Rate')
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_title('12M Rolling Quantity vs Tariff', fontsize=13)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.grid(True, alpha=0.25)
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=10, loc='upper left')
    plt.tight_layout()
    path = 'visualize/rolling_vs_tariff.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print('保存图表:', path)

def figure_autocorr(df):
    sub = df[(df.get('reporterDesc') == 'USA') & (df.get('partnerDesc') == 'Japan') & (df.get('flowDesc') == 'Import')].copy()
    sub = sub.sort_values('date')
    series = sub['quantity']
    lags = list(range(1, 13))
    ac = [series.autocorr(lag=l) for l in lags]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(lags, ac, color='#72B7B2')
    ax.set_xticks(lags)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation of Quantity (1–12 lags)', fontsize=13)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = 'visualize/autocorrelation.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print('保存图表:', path)

def figure_seasonal_boxen(df):
    sub = df[(df.get('reporterDesc') == 'USA') & (df.get('partnerDesc') == 'Japan') & (df.get('flowDesc') == 'Import')].copy()
    sub['month'] = sub['date'].dt.month
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxenplot(x='month', y='quantity', data=sub, ax=ax, palette='Set3')
    ax.set_title('Seasonal Distribution by Month (Boxen)', fontsize=13)
    ax.set_xlabel('Month')
    ax.set_ylabel('Vehicles')
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_thousands))
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = 'visualize/seasonal_boxen.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print('保存图表:', path)

def main():
    print('=' * 80)
    print('原始数据可视化')
    print('=' * 80)
    trade_df = load_trade()
    tariff_df = load_tariff()
    figure_timeseries_quantity_value(trade_df)
    figure_seasonality_heatmap(trade_df)
    figure_yoy_growth(trade_df, tariff_df)
    figure_partner_stackshare(trade_df)
    figure_quantity_distribution(trade_df)
    figure_rolling_vs_tariff(trade_df, tariff_df)
    figure_autocorr(trade_df)
    figure_seasonal_boxen(trade_df)
    print('=' * 80)
    print('完成')
    print('=' * 80)

if __name__ == '__main__':
    main()
