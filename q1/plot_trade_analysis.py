import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

DATA_DIR = 'preprocessed_data'
OUT_DIR = 'fancy_figures'
os.makedirs(OUT_DIR, exist_ok=True)
CHART_DIR = 'chart'
os.makedirs(CHART_DIR, exist_ok=True)

def load_all():
    tr = pd.read_csv(os.path.join(DATA_DIR, 'train_data.csv'), parse_dates=['Date'])
    va = pd.read_csv(os.path.join(DATA_DIR, 'val_data.csv'), parse_dates=['Date'])
    te = pd.read_csv(os.path.join(DATA_DIR, 'test_data.csv'), parse_dates=['Date'])
    df = pd.concat([tr, va, te], ignore_index=True)
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def load_scaler():
    p = os.path.join(DATA_DIR, 'minmax_scaler.pkl')
    return joblib.load(p) if os.path.exists(p) else None

def denorm_series(s, scaler, col):
    if scaler is None:
        return s
    names = list(getattr(scaler, 'feature_names_in_', []))
    if col not in names:
        return s
    idx = names.index(col)
    mn = float(scaler.data_min_[idx])
    mx = float(scaler.data_max_[idx])
    return s.astype(float) * (mx - mn) + mn if mx > mn else s

def denorm_selected(df, scaler, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = denorm_series(out[c], scaler, c)
    return out

def plot_scatter_with_reg(df, country, price_col, q_col, v_col):
    d = df.copy()
    x = d[price_col].values.astype(float)
    def reg_and_plot(y, title, fname):
        yv = y.values.astype(float)
        mask = np.isfinite(x) & np.isfinite(yv)
        xv = x[mask]
        yv = yv[mask]
        if len(xv) < 3:
            return
        k, b = np.polyfit(xv, yv, 1)
        yhat = k * xv + b
        ss_res = np.sum((yv - yhat) ** 2)
        ss_tot = np.sum((yv - yv.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        plt.figure(figsize=(6,5))
        plt.scatter(xv, yv, s=18, alpha=0.7)
        xs = np.linspace(xv.min(), xv.max(), 100)
        plt.plot(xs, k * xs + b, color='red')
        plt.title(f'{title} (slope={k:.3f}, R2={r2:.3f})')
        plt.xlabel('FOB Price')
        plt.ylabel(title.split('-')[-1])
        out = os.path.join(OUT_DIR, fname)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
    if q_col in d.columns:
        reg_and_plot(d[q_col], f'{country}-Quantity', f'scatter_reg_{country}_quantity.png')
    if v_col in d.columns:
        reg_and_plot(d[v_col], f'{country}-Value', f'scatter_reg_{country}_value.png')

def plot_ccf_price_to_qty(df, country, price_col, q_col, max_lag=6):
    if price_col not in df.columns or q_col not in df.columns:
        return
    x = df[price_col].values.astype(float)
    y = df[q_col].values.astype(float)
    lags = list(range(0, max_lag + 1))
    corrs = []
    for L in lags:
        if L == 0:
            xx, yy = x, y
        else:
            xx, yy = x[:-L], y[L:]
        if len(xx) < 3:
            corrs.append(np.nan)
            continue
        xm = xx - np.nanmean(xx)
        ym = yy - np.nanmean(yy)
        num = np.nansum(xm * ym)
        den = np.sqrt(np.nansum(xm ** 2) * np.nansum(ym ** 2))
        corrs.append(num / den if den > 0 else np.nan)
    plt.figure(figsize=(6,4))
    plt.plot(lags, corrs, marker='o')
    plt.xticks(lags)
    plt.xlabel('Lag (months)')
    plt.ylabel('Correlation')
    plt.title(f'CCF {country}: FOB -> Quantity')
    out = os.path.join(OUT_DIR, f'ccf_{country}_price_to_quantity.png')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def read_tariff(csv_path):
    tar = pd.read_csv(csv_path)
    if 'Month' in tar.columns and 'Rate' in tar.columns:
        tar['Date'] = pd.to_datetime(tar['Month'].astype(str).str.strip() + '-01')
    else:
        tar.columns = ['Month','Rate']
        tar['Date'] = pd.to_datetime(tar['Month'].astype(str).str.strip() + '-01')
    tar = tar.sort_values('Date')
    full = pd.DataFrame({'Date': pd.date_range(tar['Date'].min(), tar['Date'].max(), freq='MS')})
    tar = full.merge(tar[['Date','Rate']], on='Date', how='left').ffill()
    return tar

def plot_event_study(df, tar):
    tar = tar.copy()
    tar['Diff'] = tar['Rate'].diff().fillna(0)
    evs = tar[tar['Diff'] != 0]['Date'].tolist()
    if not evs:
        return
    rows = []
    for dt in evs:
        pre = df[(df['Date'] >= dt - pd.DateOffset(months=6)) & (df['Date'] < dt)]
        post = df[(df['Date'] > dt) & (df['Date'] <= dt + pd.DateOffset(months=6))]
        for col, tag in [('US_Bilateral_Q','Quantity'), ('US_Bilateral_V','Value')]:
            if col not in df.columns:
                continue
            pre_mean = float(pre[col].mean()) if len(pre) else np.nan
            post_mean = float(post[col].mean()) if len(post) else np.nan
            rows.append({'Event': dt.strftime('%Y-%m'), 'Metric': tag, 'Pre': pre_mean, 'Post': post_mean, 'Diff': post_mean - pre_mean if np.isfinite(pre_mean) and np.isfinite(post_mean) else np.nan})
    r = pd.DataFrame(rows)
    for tag in r['Metric'].unique():
        d = r[r['Metric'] == tag]
        ind = np.arange(len(d))
        w = 0.35
        plt.figure(figsize=(10,4))
        plt.bar(ind - w/2, d['Pre'].values, width=w, label='Pre')
        plt.bar(ind + w/2, d['Post'].values, width=w, label='Post')
        plt.xticks(ind, d['Event'].tolist(), rotation=45)
        plt.title(f'Tariff Event Study - US {tag}')
        plt.legend()
        out = os.path.join(OUT_DIR, f'event_study_us_{tag.lower()}.png')
        plt.tight_layout()
        plt.savefig(out)
        plt.close()

def plot_share_stacked(df):
    cols = ['US_Export_China_Q','AR_Export_China_Q','BR_Export_China_Q']
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return
    d = df[['Date'] + cols].copy()
    tot = d[cols].sum(axis=1)
    shares = [d[c] / tot.replace(0, np.nan) for c in cols]
    plt.figure(figsize=(10,5))
    plt.stackplot(d['Date'], shares, labels=[c.split('_')[0] for c in cols])
    plt.legend(loc='upper right')
    plt.title('Export-to-China Share (Quantity)')
    out = os.path.join(OUT_DIR, 'share_stacked_export_to_china_quantity.png')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_pred_share(csv_path, title, out_name):
    if not os.path.exists(csv_path):
        return
    d = pd.read_csv(csv_path, parse_dates=['Date'])
    piv = d.pivot_table(index='Date', columns='Country', values='Pred', aggfunc='sum').sort_values('Date')
    piv = piv.clip(lower=0)
    tot = piv.sum(axis=1)
    shares = [piv.get('US'), piv.get('AR'), piv.get('BR')]
    shares = [s.divide(tot.replace(0, np.nan)).fillna(0) if s is not None else None for s in shares]
    labels = ['US','AR','BR']
    x = piv.index
    plt.figure(figsize=(10,5))
    vals = [s.values if s is not None else np.zeros(len(x)) for s in shares]
    plt.stackplot(x, vals, labels=labels)
    plt.legend(loc='upper right')
    plt.title(title)
    out = os.path.join(OUT_DIR, out_name)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def main():
    df = load_all()
    scaler = load_scaler()
    need_cols = [
        'US_Bilateral_Q','AR_Bilateral_Q','BR_Bilateral_Q',
        'US_Bilateral_V','AR_Bilateral_V','BR_Bilateral_V',
        'US_Export_China_Q','AR_Export_China_Q','BR_Export_China_Q',
        'US_Export_China_V','AR_Export_China_V','BR_Export_China_V',
        'US_Import_Q','AR_Import_Q','BR_Import_Q',
        'US_Import_V','AR_Import_V','BR_Import_V',
        'US_FOB_Price','AR_FOB_Price','BR_FOB_Price'
    ]
    df = denorm_selected(df, scaler, [c for c in need_cols if c in df.columns])
    for cc, p, q, v in [
        ('US','US_FOB_Price','US_Import_Q','US_Import_V'),
        ('AR','AR_FOB_Price','AR_Import_Q','AR_Import_V'),
        ('BR','BR_FOB_Price','BR_Import_Q','BR_Import_V')
    ]:
        if p in df.columns:
            plot_scatter_with_reg(df, cc, p, q, v)
    for cc, p, q in [
        ('US','US_FOB_Price','US_Import_Q'),
        ('AR','AR_FOB_Price','AR_Import_Q'),
        ('BR','BR_FOB_Price','BR_Import_Q')
    ]:
        if p in df.columns and q in df.columns:
            plot_ccf_price_to_qty(df, cc, p, q, max_lag=6)
    tar = read_tariff(os.path.join('data', 'tariff.csv')) if os.path.exists(os.path.join('data','tariff.csv')) else None
    if tar is not None:
        plot_event_study(df, tar)
    plot_share_stacked(df)
    preds = [
        (os.path.join('figures','predictions_lstm_quantity.csv'), 'Predicted Share - LSTM Quantity', 'pred_lstm_share_quantity.png'),
        (os.path.join('figures','predictions_lstm_value.csv'), 'Predicted Share - LSTM Value', 'pred_lstm_share_value.png'),
        (os.path.join('figures','predictions_transformer_quantity.csv'), 'Predicted Share - Transformer Quantity', 'pred_transformer_share_quantity.png'),
        (os.path.join('figures','predictions_transformer_value.csv'), 'Predicted Share - Transformer Value', 'pred_transformer_share_value.png')
    ]
    for p, t, f in preds:
        if os.path.exists(p):
            plot_pred_share(p, t, f)
    
    def export_model_predictions(model_name, q_csv, v_csv, out_file):
        if not (os.path.exists(q_csv) and os.path.exists(v_csv)):
            return
        q = pd.read_csv(q_csv, parse_dates=['Date'])
        v = pd.read_csv(v_csv, parse_dates=['Date'])
        q_keep = q[['Date','Country','Actual','Pred']].rename(columns={'Actual':'Actual_Quantity','Pred':'Pred_Quantity'})
        v_keep = v[['Date','Country','Actual','Pred']].rename(columns={'Actual':'Actual_Value','Pred':'Pred_Value'})
        merged = pd.merge(q_keep, v_keep, on=['Date','Country'], how='outer').sort_values(['Date','Country'])
        merged['Model'] = model_name
        out_path = os.path.join(CHART_DIR, out_file)
        merged.to_csv(out_path, index=False)
    export_model_predictions('LSTM', os.path.join('figures','predictions_lstm_quantity.csv'), os.path.join('figures','predictions_lstm_value.csv'), 'pred_lstm.csv')
    export_model_predictions('Transformer', os.path.join('figures','predictions_transformer_quantity.csv'), os.path.join('figures','predictions_transformer_value.csv'), 'pred_transformer.csv')
    

if __name__ == '__main__':
    main()
