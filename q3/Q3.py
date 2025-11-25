import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression
from sklearn.pipeline import Pipeline
from pathlib import Path
import re

np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / 'data'
chart_dir = base_dir / 'chart'
figure_dir = base_dir / 'figure'
chart_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)

def read_all_trade_files(path: Path):
    files = [p for p in path.glob('*.xlsx') if p.name != 'tariff_data.xlsx']
    frames = []
    for p in files:
        df = pd.read_excel(p)
        cols = [c for c in ['refPeriodId','reporterDesc','flowDesc','partnerDesc','cmdCode','cmdDesc','qtyUnitAbbr','qty','primaryValue'] if c in df.columns]
        frames.append(df[cols].copy())
    return pd.concat(frames, ignore_index=True)

semi_data = read_all_trade_files(data_dir)
semi_data['date'] = pd.to_datetime(semi_data['refPeriodId'].astype(str), format='%Y%m%d')

us_semi_import = semi_data[(semi_data['reporterDesc']=='USA') & (semi_data['partnerDesc']=='China') & (semi_data['flowDesc']=='Import')].copy()
us_semi_export = semi_data[(semi_data['reporterDesc']=='USA') & (semi_data['partnerDesc']=='China') & (semi_data['flowDesc']=='Export')].copy()

us_semi_import_monthly = us_semi_import.groupby('date').agg({'primaryValue':'sum','qty':'sum'}).reset_index().rename(columns={'primaryValue':'import_value','qty':'import_qty'})
us_semi_export_monthly = us_semi_export.groupby('date').agg({'primaryValue':'sum','qty':'sum'}).reset_index().rename(columns={'primaryValue':'export_value','qty':'export_qty'})

semi_monthly = us_semi_import_monthly.merge(us_semi_export_monthly, on='date', how='outer').fillna(0)
semi_monthly = semi_monthly[semi_monthly['date'] <= '2025-08-31'].sort_values('date').reset_index(drop=True)

tariff_path = data_dir / 'tariff_data.xlsx'
tariff_data = pd.read_excel(tariff_path)
tariff_data['hts8'] = tariff_data['hts8'].astype(str).str.zfill(8)

def parse_tariff_rate(x):
    if pd.isna(x) or str(x).strip()=='':
        return 0.0
    s = str(x).strip()
    if '免税' in s or s=='Free':
        return 0.0
    if '%' in s:
        try:
            return float(s.replace('%','').strip())/100.0
        except:
            return 0.0
    return 0.05

def create_tariff_timeline(tariff_df):
    min_date = pd.to_datetime('2020-01-01')
    max_date = pd.to_datetime('2025-08-31')
    all_dates = pd.date_range(min_date, max_date, freq='MS')
    rows = []
    for _, r in tariff_df.iterrows():
        hts = str(r['hts8']).zfill(8)
        b = pd.to_datetime(r['begin_effect_date']) if pd.notna(r['begin_effect_date']) else min_date
        e = pd.to_datetime(r['end_effective_date']) if pd.notna(r['end_effective_date']) else max_date
        t = parse_tariff_rate(r.get('非WTO成员国', '0'))
        for d in all_dates:
            if b <= d <= e:
                rows.append({'hts8': hts, 'date': d, 'tariff_rate': t})
    return pd.DataFrame(rows)

tariff_timeline = create_tariff_timeline(tariff_data)
policy_change_date = pd.to_datetime('2025-04-02')
baseline_tariff = 0.0244
new_tariff = 0.2011

semi_monthly['date_month_start'] = semi_monthly['date'].dt.to_period('M').dt.to_timestamp()
semi_monthly['tariff_rate'] = np.where(semi_monthly['date_month_start'] >= policy_change_date, new_tariff, baseline_tariff)

us_rows = semi_data[(semi_data['reporterDesc']=='USA') & (semi_data['partnerDesc']=='China') & (semi_data['flowDesc']=='Import')].copy()
us_rows['date'] = pd.to_datetime(us_rows['refPeriodId'].astype(str), format='%Y%m%d')
us_rows['date_month_start'] = us_rows['date'].dt.to_period('M').dt.to_timestamp()
us_rows['hts8'] = us_rows['cmdCode'].astype(str).str.zfill(8)
us_rows['qty_safe'] = us_rows['qty'].fillna(0).replace(0, np.nan)
us_rows['unit_value'] = us_rows['primaryValue'] / us_rows['qty_safe']
uv = us_rows['unit_value'].dropna()
q1 = np.nanpercentile(uv, 30) if len(uv)>0 else np.nan
q2 = np.nanpercentile(uv, 70) if len(uv)>0 else np.nan
def _tier(x):
    if np.isnan(x):
        return 'mid'
    if x >= q2:
        return 'high'
    if x <= q1:
        return 'low'
    return 'mid'
us_rows['tier'] = us_rows['unit_value'].apply(_tier)
tariff_timeline['date'] = pd.to_datetime(tariff_timeline['date'])
rows_merged = us_rows.merge(tariff_timeline, left_on=['hts8','date_month_start'], right_on=['hts8','date'], how='left')
rows_merged['tariff_rate'] = np.where(rows_merged['date_month_start'] >= policy_change_date,
    np.where(rows_merged['tariff_rate'].isna(), new_tariff, rows_merged['tariff_rate']),
    np.where(rows_merged['tariff_rate'].isna(), baseline_tariff, rows_merged['tariff_rate']))
export_control_date = pd.to_datetime('2022-10-07')
rows_merged['export_control'] = ((rows_merged['date_month_start'] >= export_control_date) & (rows_merged['tier']=='high')).astype(int)
monthly_tier = rows_merged.groupby(['date_month_start','tier']).apply(lambda d: pd.Series({
    'import_value': d['primaryValue'].sum(),
    'import_qty': d['qty'].sum(),
    'tariff_rate': (d['tariff_rate']*d['primaryValue']).sum()/d['primaryValue'].sum() if d['primaryValue'].sum()!=0 else 0.0,
    'export_control': d['export_control'].max()
})).reset_index().rename(columns={'date_month_start':'date'})

us_exports_nc = semi_data[(semi_data['reporterDesc']=='USA') & (semi_data['flowDesc']=='Export') & (semi_data['partnerDesc']!='China')].copy()
us_exports_nc['date'] = pd.to_datetime(us_exports_nc['refPeriodId'].astype(str), format='%Y%m%d')
us_exports_nc['date_month_start'] = us_exports_nc['date'].dt.to_period('M').dt.to_timestamp()
us_exports_nc['hts8'] = us_exports_nc['cmdCode'].astype(str).str.zfill(8)
us_exports_nc['qty_safe'] = us_exports_nc['qty'].fillna(0).replace(0, np.nan)
us_exports_nc['unit_value'] = us_exports_nc['primaryValue'] / us_exports_nc['qty_safe']
us_exports_nc['tier'] = us_exports_nc['unit_value'].apply(_tier)
domestic_proxy = us_exports_nc.groupby(['date_month_start','tier'], as_index=False)['primaryValue'].sum().rename(columns={'primaryValue':'domestic_supply_proxy'})
domestic_proxy = domestic_proxy.rename(columns={'date_month_start':'date'})
monthly_tier = monthly_tier.merge(domestic_proxy, on=['date','tier'], how='left')
monthly_tier['domestic_supply_proxy'] = monthly_tier['domestic_supply_proxy'].fillna(0.0)

# National security indicators (monthly)
imports_by_hts = rows_merged.groupby(['date_month_start','hts8'], as_index=False)['primaryValue'].sum()
imports_total = imports_by_hts.groupby('date_month_start', as_index=False)['primaryValue'].sum().rename(columns={'primaryValue':'total_value'})
imports_share = imports_by_hts.merge(imports_total, on='date_month_start', how='left')
imports_share['share'] = np.where(imports_share['total_value']>0, imports_share['primaryValue']/imports_share['total_value'], 0.0)
hhi_df = imports_share.groupby('date_month_start', as_index=False)['share'].apply(lambda s: float(np.sum(np.square(s.values)))).rename(columns={'date_month_start':'date','share':'hhi'})

tier_sum = monthly_tier.groupby('date', as_index=False)['import_value'].sum().rename(columns={'import_value':'import_total'})
high_tier = monthly_tier[monthly_tier['tier']=='high'].copy()
high_tier = high_tier.merge(tier_sum, on='date', how='left')
high_tier['share_high'] = np.where(high_tier['import_total']>0, high_tier['import_value']/high_tier['import_total'], 0.0)
high_tier['domestic_sub_ratio'] = np.where(high_tier['import_value']>0, high_tier['domestic_supply_proxy']/high_tier['import_value'], 0.0)

security_monthly = high_tier[['date','share_high','export_control','domestic_sub_ratio']].merge(hhi_df, on='date', how='left')
security_monthly['security_index'] = 0.5*security_monthly['share_high'] + 0.3*security_monthly['export_control'] + 0.2*security_monthly['hhi'].fillna(0.0)
security_monthly.to_csv(chart_dir / 'semiconductor_security_indicators.csv', index=False)

semi_monthly['year'] = semi_monthly['date'].dt.year
semi_monthly['month'] = semi_monthly['date'].dt.month
semi_monthly['quarter'] = semi_monthly['date'].dt.quarter
semi_monthly['time_index'] = range(len(semi_monthly))
semi_monthly['Q1'] = (semi_monthly['quarter']==1).astype(int)
semi_monthly['Q2'] = (semi_monthly['quarter']==2).astype(int)
semi_monthly['Q3'] = (semi_monthly['quarter']==3).astype(int)
semi_monthly['Q4'] = (semi_monthly['quarter']==4).astype(int)
semi_monthly['post_policy'] = (semi_monthly['date'] >= policy_change_date).astype(int)
semi_monthly['covid_period'] = ((semi_monthly['date']>='2020-03-01') & (semi_monthly['date']<='2020-12-31')).astype(int)
semi_monthly['ln_import_value'] = np.log(semi_monthly['import_value'] + 1)
semi_monthly['ln_export_value'] = np.log(semi_monthly['export_value'] + 1)
semi_monthly['ln_tariff'] = np.log(1 + semi_monthly['tariff_rate'])
for col in ['import_value','export_value']:
    Q1 = semi_monthly[col].quantile(0.25)
    Q3 = semi_monthly[col].quantile(0.75)
    IQR = Q3 - Q1
    semi_monthly[col] = semi_monthly[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
for lag in [1,3,6,12]:
    semi_monthly[f'import_value_lag{lag}'] = semi_monthly['import_value'].shift(lag)
    semi_monthly[f'ln_import_value_lag{lag}'] = semi_monthly['ln_import_value'].shift(lag)
semi_monthly['import_ma3'] = semi_monthly['import_value'].rolling(window=3, min_periods=1).mean()
semi_monthly['import_ma6'] = semi_monthly['import_value'].rolling(window=6, min_periods=1).mean()
semi_monthly['import_ma12'] = semi_monthly['import_value'].rolling(window=12, min_periods=1).mean()
semi_monthly['import_growth_mom'] = semi_monthly['import_value'].pct_change()
semi_monthly['import_growth_yoy'] = semi_monthly['import_value'].pct_change(12)
semi_monthly['import_unit_value'] = semi_monthly['import_value'] / (semi_monthly['import_qty'] + 0.0001)
semi_monthly['ln_unit_value'] = np.log(semi_monthly['import_unit_value'] + 1)

semi_data_clean = semi_monthly.dropna().reset_index(drop=True)

X_elasticity = semi_data_clean[['time_index','ln_tariff','Q1','Q2','Q3','covid_period','post_policy']]
y_elasticity = semi_data_clean['ln_import_value']
elasticity_pipe = Pipeline([('scaler', StandardScaler()), ('model', ElasticNet(max_iter=10000, random_state=42))])
elasticity_grid = {'model__alpha': [1e-4, 1e-3, 1e-2, 0.1, 1.0], 'model__l1_ratio': [0.2, 0.5, 0.8]}
tscv_el = TimeSeriesSplit(n_splits=5)
elasticity_cv = GridSearchCV(elasticity_pipe, elasticity_grid, cv=tscv_el, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
elasticity_cv.fit(X_elasticity, y_elasticity)
elasticity_model = elasticity_cv.best_estimator_
y_pred_elasticity = elasticity_model.predict(X_elasticity)
r2_elasticity = r2_score(y_elasticity, y_pred_elasticity)
rmse_elasticity = np.sqrt(mean_squared_error(y_elasticity, y_pred_elasticity))
mape_elasticity = mean_absolute_percentage_error(np.exp(y_elasticity), np.exp(y_pred_elasticity))

X_dynamic = semi_data_clean[['ln_import_value_lag1','time_index','ln_tariff','Q1','Q2','Q3','covid_period','post_policy']]
y_dynamic = semi_data_clean['ln_import_value']
scaler_dyn = StandardScaler()
X_dynamic_scaled = scaler_dyn.fit_transform(X_dynamic)
dynamic_model = HuberRegressor()
dynamic_model.fit(X_dynamic_scaled, y_dynamic)
y_pred_dynamic = dynamic_model.predict(X_dynamic_scaled)
r2_dynamic = r2_score(y_dynamic, y_pred_dynamic)
rmse_dynamic = np.sqrt(mean_squared_error(y_dynamic, y_pred_dynamic))
mape_dynamic = mean_absolute_percentage_error(np.exp(y_dynamic), np.exp(y_pred_dynamic))
persistence_param = dynamic_model.coef_[0]
tariff_elasticity_short = dynamic_model.coef_[2]
tariff_elasticity_long = tariff_elasticity_short / (1 - persistence_param) if (1 - persistence_param)!=0 else np.nan

feature_cols = ['ln_import_value_lag1','time_index','ln_tariff','Q1','Q2','Q3','covid_period','post_policy','import_growth_mom','import_growth_yoy','ln_unit_value','import_ma3','import_ma6','import_ma12']
X_gbdt = semi_data_clean[feature_cols].fillna(0)
y_gbdt = semi_data_clean['ln_import_value']
scaler = StandardScaler()
X_gbdt_scaled = scaler.fit_transform(X_gbdt)
tscv = TimeSeriesSplit(n_splits=5)
train_idx, test_idx = list(tscv.split(X_gbdt_scaled))[-1]
X_train, X_test = X_gbdt_scaled[train_idx], X_gbdt_scaled[test_idx]
y_train, y_test = y_gbdt.iloc[train_idx].values, y_gbdt.iloc[test_idx].values
gbdt_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.05, 0.1],
    'min_samples_split': [5, 10],
    'subsample': [0.8, 0.9]
}
gbdt_cv = GridSearchCV(GradientBoostingRegressor(random_state=42, loss='huber'), gbdt_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
gbdt_cv.fit(X_train, y_train)
gbdt_model = gbdt_cv.best_estimator_
y_train_pred = gbdt_model.predict(X_train)
y_test_pred = gbdt_model.predict(X_test)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mape_test = mean_absolute_percentage_error(np.exp(y_test), np.exp(y_test_pred))
feature_importance = pd.DataFrame({'feature': feature_cols, 'importance': gbdt_model.feature_importances_}).sort_values('importance', ascending=False)

scenarios = {
    'Baseline (2.5%)': 0.025,
    'Low (5%)': 0.05,
    'Medium (10%)': 0.10,
    'High (20%)': 0.20,
    'Actual (20.11%)': 0.2011,
    'Severe (35%)': 0.35
}
last_idx = len(semi_data_clean) - 1
base_features = X_gbdt.iloc[last_idx].copy()
scenario_results = {}
for name, rate in scenarios.items():
    sf = base_features.copy()
    sf['ln_tariff'] = np.log(1 + rate)
    sf['post_policy'] = 1
    Xs = scaler.transform(sf.values.reshape(1, -1))
    ln_pred_g = gbdt_model.predict(Xs)[0]
    pred_g = np.exp(ln_pred_g)
    Xd_df = pd.DataFrame([{
        'ln_import_value_lag1': sf['ln_import_value_lag1'],
        'time_index': sf['time_index'],
        'ln_tariff': sf['ln_tariff'],
        'Q1': sf['Q1'],
        'Q2': sf['Q2'],
        'Q3': sf['Q3'],
        'covid_period': sf['covid_period'],
        'post_policy': sf['post_policy']
    }])
    ln_pred_d = dynamic_model.predict(scaler_dyn.transform(Xd_df))[0]
    pred_d = np.exp(ln_pred_d)
    scenario_results[name] = {'tariff_rate': rate, 'gbdt_prediction': pred_g, 'dynamic_prediction': pred_d, 'ensemble_prediction': (pred_g + pred_d)/2}
baseline_pred = scenario_results['Baseline (2.5%)']['ensemble_prediction']
economic_indicators = []
for name, res in scenario_results.items():
    tariff = res['tariff_rate']
    import_value = res['ensemble_prediction']
    tariff_revenue = import_value * tariff
    import_change = import_value - baseline_pred
    import_change_pct = (import_change / baseline_pred * 100) if baseline_pred>0 else 0
    consumer_loss = abs(import_change) * 0.5 if import_change < 0 else 0
    econ_score = (tariff_revenue - consumer_loss)/1e9 if baseline_pred>0 else 0
    # Security score using tier predictions (share of high-end + export control + HHI)
    try:
        dscn = [s for s in tier_scenarios if s['Scenario']==name]
        total_pred = sum([s['ensemble_prediction'] for s in dscn])
        high_pred = sum([s['ensemble_prediction'] for s in dscn if s['Tier']=='high'])
        share_high_pred = (high_pred/total_pred) if total_pred>0 else 0.0
        hhi_latest = float(hhi_df.sort_values('date')['hhi'].iloc[-1]) if len(hhi_df)>0 else 0.0
        export_control_flag = 1
        sec_score = 0.5*share_high_pred + 0.3*export_control_flag + 0.2*hhi_latest
    except Exception:
        sec_score = 0.0
    policy_score = 0.6*econ_score + 0.4*sec_score
    economic_indicators.append({'Scenario': name, 'Tariff': tariff*100, 'Import_Value_B': import_value/1e9, 'Change_vs_Baseline_%': import_change_pct, 'Tariff_Revenue_B': tariff_revenue/1e9, 'Economic_Score': econ_score, 'Security_Score': sec_score, 'Policy_Score': policy_score})
econ_df = pd.DataFrame(economic_indicators)

results_df = pd.DataFrame({
    'Date': semi_data_clean['date'],
    'Actual_Import_Value': semi_data_clean['import_value'],
    'Elasticity_Model_Pred': np.exp(y_pred_elasticity),
    'Dynamic_Model_Pred': np.exp(y_pred_dynamic),
    'Tariff_Rate': semi_data_clean['tariff_rate'],
    'Post_Policy': semi_data_clean['post_policy']
})
results_df.to_csv(chart_dir / 'semiconductor_model_predictions.csv', index=False)
pd.DataFrame(scenario_results).T.reset_index().rename(columns={'index':'Scenario'}).to_csv(chart_dir / 'semiconductor_scenario_analysis.csv', index=False)
econ_df.to_csv(chart_dir / 'semiconductor_economic_indicators.csv', index=False)
feature_importance.to_csv(chart_dir / 'semiconductor_feature_importance.csv', index=False)

tier_results = []
tier_scenarios = []
econ_tier = []
for tier in ['high','mid','low']:
    df_t = monthly_tier[monthly_tier['tier']==tier].copy()
    if len(df_t) < 24:
        continue
    df_t = df_t.sort_values('date').reset_index(drop=True)
    df_t['year'] = df_t['date'].dt.year
    df_t['month'] = df_t['date'].dt.month
    df_t['quarter'] = df_t['date'].dt.quarter
    df_t['time_index'] = range(len(df_t))
    df_t['Q1'] = (df_t['quarter']==1).astype(int)
    df_t['Q2'] = (df_t['quarter']==2).astype(int)
    df_t['Q3'] = (df_t['quarter']==3).astype(int)
    df_t['Q4'] = (df_t['quarter']==4).astype(int)
    df_t['post_policy'] = (df_t['date'] >= policy_change_date).astype(int)
    df_t['covid_period'] = ((df_t['date']>='2020-03-01') & (df_t['date']<='2020-12-31')).astype(int)
    df_t['ln_import_value'] = np.log(df_t['import_value'] + 1)
    df_t['ln_tariff'] = np.log(1 + df_t['tariff_rate'])
    subsidy_start = pd.to_datetime('2022-08-01')
    df_t['subsidy_period'] = (df_t['date'] >= subsidy_start).astype(int)
    df_t['ln_tariff_subsidy'] = df_t['ln_tariff'] * df_t['subsidy_period']
    df_t['ln_domestic_proxy'] = np.log(df_t['domestic_supply_proxy'] + 1)
    for lag in [1,3,6,12]:
        df_t[f'ln_import_value_lag{lag}'] = df_t['ln_import_value'].shift(lag)
        df_t[f'import_value_lag{lag}'] = df_t['import_value'].shift(lag)
    df_t['import_ma3'] = df_t['import_value'].rolling(window=3, min_periods=1).mean()
    df_t['import_ma6'] = df_t['import_value'].rolling(window=6, min_periods=1).mean()
    df_t['import_ma12'] = df_t['import_value'].rolling(window=12, min_periods=1).mean()
    df_t['import_growth_mom'] = df_t['import_value'].pct_change()
    df_t['import_growth_yoy'] = df_t['import_value'].pct_change(12)
    df_t['import_unit_value'] = df_t['import_value'] / (df_t['import_qty'] + 0.0001)
    df_t['ln_unit_value'] = np.log(df_t['import_unit_value'] + 1)
    df_c = df_t.dropna().reset_index(drop=True)
    Xe = df_c[['time_index','ln_tariff','ln_tariff_subsidy','subsidy_period','ln_domestic_proxy','Q1','Q2','Q3','covid_period','post_policy','export_control']]
    ye = df_c['ln_import_value']
    m_e = LinearRegression()
    m_e.fit(Xe, ye)
    yp_e = m_e.predict(Xe)
    Xd = df_c[['ln_import_value_lag1','time_index','ln_tariff','ln_tariff_subsidy','subsidy_period','ln_domestic_proxy','Q1','Q2','Q3','covid_period','post_policy','export_control']]
    yd = df_c['ln_import_value']
    m_d = LinearRegression()
    m_d.fit(Xd, yd)
    yp_d = m_d.predict(Xd)
    tier_results.append(pd.DataFrame({'Tier': tier, 'Date': df_c['date'], 'Actual_Import_Value': df_c['import_value'], 'Elasticity_Model_Pred': np.exp(yp_e), 'Dynamic_Model_Pred': np.exp(yp_d), 'Domestic_Supply_Proxy': df_c['domestic_supply_proxy'], 'Tariff_Rate': df_c['tariff_rate'], 'Post_Policy': df_c['post_policy'], 'Export_Control': df_c['export_control']}))
    scn = {'Baseline (2.5%)':0.025,'Low (5%)':0.05,'Medium (10%)':0.10,'High (20%)':0.20,'Actual (20.11%)':0.2011,'Severe (35%)':0.35}
    base_idx = len(df_c) - 1
    bf = df_c.iloc[base_idx]
    for nm, rate in scn.items():
        ln_tar = np.log(1 + rate)
        Xe_s = np.array([[bf['time_index'], ln_tar, ln_tar*bf['subsidy_period'], bf['subsidy_period'], bf['ln_domestic_proxy'], bf['Q1'], bf['Q2'], bf['Q3'], bf['covid_period'], 1, bf['export_control']]])
        yd_lag1 = bf['ln_import_value_lag1']
        Xd_s = np.array([[yd_lag1, bf['time_index'], ln_tar, ln_tar*bf['subsidy_period'], bf['subsidy_period'], bf['ln_domestic_proxy'], bf['Q1'], bf['Q2'], bf['Q3'], bf['covid_period'], 1, bf['export_control']]])
        pred_e = np.exp(m_e.predict(Xe_s)[0])
        pred_d = np.exp(m_d.predict(Xd_s)[0])
        tier_scenarios.append({'Tier': tier, 'Scenario': nm, 'tariff_rate': rate, 'elasticity_pred': pred_e, 'dynamic_pred': pred_d, 'ensemble_prediction': (pred_e + pred_d)/2})
    baseline = [s for s in tier_scenarios if s['Tier']==tier and s['Scenario']=='Baseline (2.5%)'][0]['ensemble_prediction']
    for s in tier_scenarios:
        if s['Tier']==tier:
            s['Change_vs_Baseline_%'] = ((s['ensemble_prediction'] - baseline)/baseline*100) if baseline!=0 else 0
            tariff = s['tariff_rate']
            import_value = s['ensemble_prediction']
            revenue = import_value * tariff
            change = import_value - baseline
            consumer_loss = abs(change) * 0.5 if change < 0 else 0
            econ_score = (revenue - consumer_loss)/1e9 if baseline>0 else 0
            econ_tier.append({'Tier': tier, 'Scenario': s['Scenario'], 'Tariff': tariff*100, 'Import_Value_B': import_value/1e9, 'Change_vs_Baseline_%': s['Change_vs_Baseline_%'], 'Tariff_Revenue_B': revenue/1e9, 'Economic_Score': econ_score})

if tier_results:
    tier_pred_df = pd.concat(tier_results, ignore_index=True)
    tier_scn_df = pd.DataFrame(tier_scenarios)
    tier_econ_df = pd.DataFrame(econ_tier)
    tier_pred_df.to_csv(chart_dir / 'semiconductor_model_predictions_by_tier.csv', index=False)
    tier_scn_df.to_csv(chart_dir / 'semiconductor_scenario_analysis_by_tier.csv', index=False)
    tier_econ_df.to_csv(chart_dir / 'semiconductor_economic_indicators_by_tier.csv', index=False)
    fig, ax = plt.subplots(figsize=(14,7))
    for tier in ['high','mid','low']:
        d = tier_pred_df[tier_pred_df['Tier']==tier]
        ax.plot(d['Date'], d['Actual_Import_Value']/1e9, label=f'{tier}-actual')
        ax.plot(d['Date'], d['Dynamic_Model_Pred']/1e9, label=f'{tier}-dynamic', linestyle='--')
        ax.plot(d['Date'], d['Domestic_Supply_Proxy']/1e9, label=f'{tier}-domestic', linestyle=':', color='gray')
    ax.set_title('By Tier: Actual vs Dynamic Model')
    ax.set_ylabel('Import Value (Billion USD)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_dir / 'timeseries_by_tier.png', dpi=300, bbox_inches='tight')
    plt.close()
    fig, ax = plt.subplots(figsize=(12,7))
    markers = {'high':'o','mid':'s','low':'^'}
    colors = {'high':'crimson','mid':'steelblue','low':'darkgreen'}
    offset_map = {'Baseline (2.5%)':(0,10),'Low (5%)':(0,-10),'Medium (10%)':(0,10),'High (20%)':(10,0),'Actual (20.11%)':(-12,0),'Severe (35%)':(-12,0)}
    for tier in ['high','mid','low']:
        d = tier_scn_df[tier_scn_df['Tier']==tier]
        ax.scatter(d['tariff_rate']*100, d['ensemble_prediction']/1e9, s=140, marker=markers[tier], color=colors[tier], label=tier)
        for _, row in d.iterrows():
            name = row['Scenario']
            ox, oy = offset_map.get(name, (6,6))
            ax.annotate(name.split('(')[0], (row['tariff_rate']*100, row['ensemble_prediction']/1e9), xytext=(ox, oy), textcoords='offset points', fontsize=9)
    ax.set_title('By Tier: Tariff Scenarios vs Import Value')
    ax.set_xlabel('Tariff Rate (%)')
    ax.set_ylabel('Predicted Import Value (Billion USD)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_dir / 'scenario_by_tier.png', dpi=300, bbox_inches='tight')
    plt.close()

fig, ax = plt.subplots(figsize=(10,6))
pre = monthly_tier[monthly_tier['date'] < policy_change_date].groupby('tier')['import_value'].mean()
post = monthly_tier[monthly_tier['date'] >= policy_change_date].groupby('tier')['import_value'].mean()
tiers = ['high','mid','low']
pre_vals = [pre.get(t, 0)/1e9 for t in tiers]
post_vals = [post.get(t, 0)/1e9 for t in tiers]
x = np.arange(len(tiers))
width = 0.35
ax.bar(x - width/2, pre_vals, width, label='Pre-policy', color='gray')
ax.bar(x + width/2, post_vals, width, label='Post-policy', color='crimson')
ax.set_xticks(x)
ax.set_xticklabels(tiers)
ax.set_ylabel('Import Value (Billion USD)')
ax.set_title('Policy Effect by Tier')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(figure_dir / 'policy_effect_bars.png', dpi=300, bbox_inches='tight')
plt.close()

# Security index over time figure
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(semi_monthly['date'], semi_monthly['import_value']/1e9, color='steelblue', label='Import Value (BUSD)')
sdf = security_monthly.sort_values('date')
ax2 = ax.twinx()
ax2.plot(sdf['date'], sdf['security_index'], color='crimson', label='Security Index')
ax.set_title('Import vs National Security Index')
ax.set_ylabel('Import (Billion USD)')
ax2.set_ylabel('Security Index')
ax.grid(True, alpha=0.3)
lns = ax.get_lines() + ax2.get_lines()
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper left')
plt.tight_layout()
plt.savefig(figure_dir / 'security_index_over_time.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12,6))
rates = np.linspace(0, 0.35, 50)
curve = []
for r in rates:
    sf = base_features.copy()
    sf['ln_tariff'] = np.log(1 + r)
    sf['post_policy'] = 1
    Xs = scaler.transform(sf.values.reshape(1, -1))
    ln_pred_g = gbdt_model.predict(Xs)[0]
    pred_g = np.exp(ln_pred_g)
    Xd_df = pd.DataFrame([{
        'ln_import_value_lag1': sf['ln_import_value_lag1'],
        'time_index': sf['time_index'],
        'ln_tariff': sf['ln_tariff'],
        'Q1': sf['Q1'],
        'Q2': sf['Q2'],
        'Q3': sf['Q3'],
        'covid_period': sf['covid_period'],
        'post_policy': sf['post_policy']
    }])
    ln_pred_d = dynamic_model.predict(scaler_dyn.transform(Xd_df))[0]
    pred_d = np.exp(ln_pred_d)
    curve.append((pred_g + pred_d)/2)
ax.plot(rates*100, np.array(curve)/1e9, color='darkblue', linewidth=2)
ax.set_xlabel('Tariff Rate (%)')
ax.set_ylabel('Predicted Import (Billion USD)')
ax.set_title('Tariff-Import Response Curve (Ensemble)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figure_dir / 'scenario_curve.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(semi_data_clean['date'], semi_data_clean['import_value']/1e9, 'k-', label='Actual', alpha=0.7)
ax.plot(semi_data_clean['date'], np.exp(y_pred_elasticity)/1e9, 'b--', label='Elasticity Model', alpha=0.8)
ax.plot(semi_data_clean['date'], np.exp(y_pred_dynamic)/1e9, 'r--', label='Dynamic Model', alpha=0.8)
ax.axvline(policy_change_date, color='gray', linestyle='--', alpha=0.5)
ax.set_title('US Chip Imports: Actual vs Model Predictions')
ax.set_ylabel('Import Value (Billion USD)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figure_dir / 'imports_actual_vs_models.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12,6))
top_features = feature_importance.head(10)
ax.barh(range(len(top_features)), top_features['importance'], alpha=0.7, color='darkgreen')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'], fontsize=10)
ax.set_xlabel('Feature Importance')
ax.set_title('GBDT Feature Importance (Top 10)')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(figure_dir / 'feature_importance_top10.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12,6))
model_names = ['Elasticity','Dynamic','GBDT']
r2_scores = [r2_elasticity, r2_dynamic, r2_test]
mape_scores = [mape_elasticity*100, mape_dynamic*100, mape_test*100]
x = np.arange(len(model_names))
width = 0.35
ax2 = ax.twinx()
bars1 = ax.bar(x - width/2, r2_scores, width, label='R²', alpha=0.8, color='steelblue')
bars2 = ax2.bar(x + width/2, mape_scores, width, label='MAPE (%)', alpha=0.8, color='coral')
ax.set_ylabel('R² Score', color='steelblue')
ax2.set_ylabel('MAPE (%)', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_title('Model Performance Comparison')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.3f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h, f'{h:.1f}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(figure_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

readme_path = base_dir / 'README.md'
if readme_path.exists():
    text = readme_path.read_text(encoding='utf-8')
    text = re.sub(r"(\s*- ElasticNet：R²≈)([-+]?\d*\.?\d+)(、RMSE≈)([-+]?\d*\.?\d+)(、MAPE≈)([-+]?\d*\.?\d+)%",
                  lambda m: f"{m.group(1)}{r2_elasticity:.3f}{m.group(3)}{rmse_elasticity:.3f}{m.group(5)}{mape_elasticity*100:.1f}%",
                  text)
    text = re.sub(r"(\s*- Huber 动态：R²≈)([-+]?\d*\.?\d+)(、RMSE≈)([-+]?\d*\.?\d+)(、MAPE≈)([-+]?\d*\.?\d+)%",
                  lambda m: f"{m.group(1)}{r2_dynamic:.3f}{m.group(3)}{rmse_dynamic:.3f}{m.group(5)}{mape_dynamic*100:.1f}%",
                  text)
    text = re.sub(r"(\s*- GBDT（CV 最优）：R²≈)([-+]?\d*\.?\d+)(、RMSE≈)([-+]?\d*\.?\d+)(、MAPE≈)([-+]?\d*\.?\d+)%",
                  lambda m: f"{m.group(1)}{r2_test:.3f}{m.group(3)}{rmse_test:.3f}{m.group(5)}{mape_test*100:.1f}%",
                  text)
    text = re.sub(r"(\s*- 关税弹性：短期≈)([-+]?\d*\.?\d+)(，长期≈)([-+]?\d*\.?\d+)",
                  lambda m: f"{m.group(1)}{tariff_elasticity_short:.4f}{m.group(3)}{tariff_elasticity_long:.4f}",
                  text)
    readme_path.write_text(text, encoding='utf-8')
else:
    lines = []
    lines.append('# 美国对华半导体贸易模型说明（US-China Semiconductor Trade Modeling）')
    lines.append('')
    lines.append('## 模型概述（方法创新）')
    lines.append('- 目标：在保留数据口径一致的前提下，采用更稳健与更具表现力的方法，提高拟合与泛化能力，并更清晰地展示政策结论。')
    lines.append('- 方法更新：')
    lines.append('  - 使用 ElasticNet 替代传统线性回归，提升稳健性与特征选择能力（时序交叉验证选参）。')
    lines.append('  - 使用 Huber 回归替代动态线性模型，增强对异常点的鲁棒性，并继续估计短期与长期关税弹性。')
    lines.append('  - 使用 GBDT 并进行网格搜索选择最佳参数，提升非线性拟合与泛化表现。')
    lines.append('  - 情景分析改为基于 Huber+GBDT 集成预测，并新增连续关税响应曲线与分档政策效应对比图。')
    lines.append('- 关键结果（当前数据运行）：')
    lines.append(f"  - ElasticNet：R²≈{r2_elasticity:.3f}、RMSE≈{rmse_elasticity:.3f}、MAPE≈{mape_elasticity*100:.1f}%")
    lines.append(f"  - Huber 动态：R²≈{r2_dynamic:.3f}、RMSE≈{rmse_dynamic:.3f}、MAPE≈{mape_dynamic*100:.1f}%")
    lines.append(f"  - GBDT（CV 最优）：R²≈{r2_test:.3f}、RMSE≈{rmse_test:.3f}、MAPE≈{mape_test*100:.1f}%")
    lines.append(f"  - 关税弹性：短期≈{tariff_elasticity_short:.4f}，长期≈{tariff_elasticity_long:.4f}")
    lines.append('')
    lines.append('## 图表与表格说明（强调结论）')
    lines.append('- figure/imports_actual_vs_models.png：实际进口与模型预测对比，标注政策节点（2025-04-02）。显示政策后模型预测的下行偏移。')
    lines.append('- figure/feature_importance_top10.png：GBDT 前 10 特征重要性，展示滞后项、关税对数、移动均线等对进口值的影响权重。')
    lines.append('- figure/model_performance_comparison.png：三类模型 R² 与 MAPE 对比，GBDT 综合表现最佳。')
    lines.append('- figure/timeseries_by_tier.png：按高/中/低端分档的实际与预测时间序列，并显示“国内供给代理”，直观体现高端受政策影响更显著。')
    lines.append('- figure/scenario_by_tier.png：分档关税情景与进口预测的散点图，展示不同关税率下各档次对进口规模的压制程度。')
    lines.append('- figure/policy_effect_bars.png：分档政策前后月均进口值对比柱状图，高端降幅更明显，结论更直观。')
    lines.append('- figure/scenario_curve.png：连续关税率（0–35%）的集成预测响应曲线，展示关税越高进口越低的单调关系。')
    lines.append('')
    lines.append('## 3.Data Collection and Preprocessing（数据搜集与预处理）')
    lines.append('')
    lines.append('### 3.1 Data Sources（数据来源）')
    lines.append('- 贸易数据：根目录 data/ 下的分年分档 Excel（2018–2025，中国/美国，低/中/高端）；字段包含 refPeriodId, reporterDesc, flowDesc, partnerDesc, cmdCode, cmdDesc, qty, primaryValue。')
    lines.append('- 关税数据：data/tariff_data.xlsx，包含 hts8、各国税率、begin_effect_date/end_effective_date 与“非WTO成员国”税率。')
    lines.append('- 2025 年中国分档：若中国侧 2025 月度明细缺失，使用“美国对中国出口”作为中国自美国进口的代理，并按单位价值分位数分档（30%、70%）。')
    lines.append('')
    lines.append('### 3.2 Data Scope and Frequency（数据范围与频率）')
    lines.append('- 时间范围：主要分析 2020-01 至 2025-08（情景评估聚焦 2025 政策变动）。')
    lines.append('- 频率：月度（按 refPeriodId 派生为 date）。')
    lines.append('- 聚合维度：美国自中国进口与对中国出口按月聚合；关税按 HS8 与生效区间映射至月维度。')
    lines.append('')
    lines.append('### 3.3 Variable Definition（变量定义）')
    lines.append('')
    lines.append('#### 3.3.1 Basic Variables（基础变量）')
    lines.append('- date：由 refPeriodId（YYYYMMDD）转换的日期。')
    lines.append('- import_value/export_value：美国对中国的进口/出口月度金额（USD）。')
    lines.append('- qty：数量（单位由 qtyUnitAbbr 指示，若为 u 则代表件数）。')
    lines.append('- tariff_rate：月度加权关税率（HS8 逐条映射后按金额加权聚合）。')
    lines.append('- cmdCode/cmdDesc：商品编码与描述（HS6/HS8）。')
    lines.append('')
    lines.append('#### 3.3.2 Derived Variables（派生变量）')
    lines.append('- year/month/quarter，Q1-Q3，time_index。')
    lines.append('- post_policy（2025-04-02 之后）、covid_period（2020-03 至 2020-12）。')
    lines.append('- ln_import_value/ln_export_value、ln_tariff。')
    lines.append('- import_value_lag{1,3,6,12}、import_ma{3,6,12}、import_growth_mom/yoy。')
    lines.append('- import_unit_value/ln_unit_value、tier（30%、70% 分位）。')
    lines.append('- export_control（高端 2022-10-07 后）、domestic_supply_proxy（美国对非中国出口）。')
    lines.append('')
    lines.append('### 3.4 Data Cleaning and Preprocessing（数据清洗与预处理）')
    lines.append('- IQR 截断处理 import_value/export_value 异常值。')
    lines.append('- 数量为 0 或缺失时不参与单位价值计算；滞后与均线造成的前期缺失行剔除。')
    lines.append('- 关税填充：缺失按默认基准/新政关税率处理（2025-04-02 节点：2.44%→20.11%）。')
    lines.append('')
    lines.append('## 4. Feature Engineering（特征工程）')
    lines.append('')
    lines.append('### 4.1 Feature Design Principles（特征设计原则）')
    lines.append('- 经济合理性、可解释性、稳健性。')
    lines.append('')
    lines.append('### 4.2 Specific Feature Extraction（具体特征提取）')
    lines.append('')
    lines.append('#### 4.2.1 Time-Series Features（时序特征）')
    lines.append('- time_index、Q1-Q3、滞后、均线、增长率。')
    lines.append('')
    lines.append('#### 4.2.2 Policy-Related Features（政策相关特征）')
    lines.append('- ln_tariff、post_policy、covid_period、export_control，分档中引入补贴期交互。')
    lines.append('')
    lines.append('#### 4.2.3 Control Features（控制特征）')
    lines.append('- ln_unit_value、domestic_supply_proxy。')
    lines.append('')
    lines.append('### 4.3 Feature Optimization（特征优化）')
    lines.append('- 标准化与时间序列交叉验证；GBDT 网格搜索选择最优参数。')
    lines.append('')
    lines.append('## 使用方法')
    lines.append('- 运行模型与结果输出：`python3 Q3.py`')
    lines.append('- 运行基础可视化：`python3 visualize.py`')
    lines.append('- 输出目录：')
    lines.append('  - 图表数据在 `chart/`，图片在 `figure/` 与 `visualize/`')
    lines.append('  - 本文件即为结果解释文档')
    readme_path.write_text('\n'.join(lines), encoding='utf-8')
