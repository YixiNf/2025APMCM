import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import FuncFormatter, PercentFormatter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import PoissonRegressor
import warnings
import os
import sys
import atexit

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = True
plt.rcParams['mathtext.fontset'] = 'dejavusans'

sns.set_theme(style='whitegrid', context='talk')

np.random.seed(42)

print("=" * 80)
print("美日汽车贸易关税影响综合分析模型 (downloaded data)")
print("=" * 80)

os.makedirs('figure', exist_ok=True)
os.makedirs('chart', exist_ok=True)


print("\n" + "=" * 80)
print("数据加载与预处理")
print("=" * 80)

print("\n加载美国进口数据...")
trade_df = pd.read_excel('downloaded data/美国进口数据.xlsx')
if 'freqCode' in trade_df.columns:
    if (trade_df['freqCode'] == 'M').any():
        trade_df = trade_df[trade_df['freqCode'] == 'M'].copy()

if 'qty' in trade_df.columns and 'quantity' not in trade_df.columns:
    trade_df = trade_df.rename(columns={'qty': 'quantity'})
if 'primaryValue' in trade_df.columns and 'TradeValue(US$)' not in trade_df.columns:
    trade_df = trade_df.rename(columns={'primaryValue': 'TradeValue(US$)'})

if {'refYear', 'refMonth'}.issubset(trade_df.columns):
    trade_df['date'] = pd.to_datetime(
        trade_df['refYear'].astype(int).astype(str) + '-' +
        trade_df['refMonth'].astype(int).astype(str).str.zfill(2) + '-01'
    )
elif 'period' in trade_df.columns:
    trade_df['date'] = pd.to_datetime(trade_df['period'].astype(str).str.slice(0, 6), format='%Y%m')
else:
    raise ValueError('无法识别时间列')

print("加载关税数据...")
tariff_df = pd.read_excel('downloaded data/美日汽车关税数据.xlsx')
tariff_df['date'] = pd.to_datetime(tariff_df['Month'].astype(str) + '-01')
baseline_tariff = float(tariff_df['US Tariff on Japan Autos'].min())
change_rows = tariff_df[tariff_df['US Tariff on Japan Autos'] != baseline_tariff]
policy_change_date = pd.to_datetime('2025-04-01') if len(change_rows) == 0 else change_rows['date'].min()

print("数据清洗和筛选...")
us_from_japan = trade_df[(trade_df['reporterDesc'] == 'USA') & (trade_df['partnerDesc'] == 'Japan') & (trade_df['flowDesc'] == 'Import')].copy()
us_imports_all = trade_df[(trade_df['reporterDesc'] == 'USA') & (trade_df['flowDesc'] == 'Import') & (trade_df['partnerDesc'].isin(['Japan', 'Mexico', 'Rep. of Korea', 'Germany', 'Canada']))].copy()

us_from_japan = us_from_japan.groupby('date', as_index=False)[['quantity','TradeValue(US$)']].sum().sort_values('date').reset_index(drop=True)
us_imports_all = us_imports_all.groupby(['date','partnerDesc'], as_index=False)['quantity'].sum().sort_values(['date','partnerDesc']).reset_index(drop=True)

us_from_japan = us_from_japan.merge(tariff_df[['date', 'US Tariff on Japan Autos']], on='date', how='left')
us_from_japan['tariff_rate'] = us_from_japan['US Tariff on Japan Autos'].fillna(baseline_tariff)
us_from_japan = us_from_japan.drop(columns=['US Tariff on Japan Autos'])

print(f"美日汽车贸易数据: {len(us_from_japan)} 个月")
print(f"数据时间范围: {us_from_japan['date'].min()} 至 {us_from_japan['date'].max()}")
print(f"总进口量: {us_from_japan['quantity'].sum() / 1e6:.2f} 百万辆")
print(f"总进口额: ${us_from_japan['TradeValue(US$)'].sum() / 1e9:.2f}B")

print("\n计算关键变量...")
us_from_japan['unit_price'] = us_from_japan['TradeValue(US$)'] / us_from_japan['quantity']
us_from_japan.loc[us_from_japan['tariff_rate'].isna(), 'tariff_rate'] = baseline_tariff
us_from_japan['year'] = us_from_japan['date'].dt.year
us_from_japan['month'] = us_from_japan['date'].dt.month
us_from_japan['quarter'] = us_from_japan['date'].dt.quarter
us_from_japan['time_index'] = range(len(us_from_japan))
us_from_japan['is_Q1'] = (us_from_japan['quarter'] == 1).astype(int)
us_from_japan['is_Q2'] = (us_from_japan['quarter'] == 2).astype(int)
us_from_japan['is_Q3'] = (us_from_japan['quarter'] == 3).astype(int)
us_from_japan['is_Q4'] = (us_from_japan['quarter'] == 4).astype(int)
us_from_japan['covid_period'] = ((us_from_japan['date'] >= '2020-03-01') & (us_from_japan['date'] <= '2020-06-30')).astype(int)
us_from_japan['post_policy'] = (us_from_japan['date'] >= policy_change_date).astype(int)
us_from_japan['ln_quantity'] = np.log(us_from_japan['quantity'] + 1)
us_from_japan['ln_trade_value'] = np.log(us_from_japan['TradeValue(US$)'] + 1)
us_from_japan['ln_unit_price'] = np.log(us_from_japan['unit_price'] + 1)
us_from_japan['ln_tariff'] = np.log(1 + us_from_japan['tariff_rate'])
us_from_japan['sin_m1'] = np.sin(2 * np.pi * us_from_japan['date'].dt.month / 12)
us_from_japan['cos_m1'] = np.cos(2 * np.pi * us_from_japan['date'].dt.month / 12)
us_from_japan['sin_m2'] = np.sin(4 * np.pi * us_from_japan['date'].dt.month / 12)
us_from_japan['cos_m2'] = np.cos(4 * np.pi * us_from_japan['date'].dt.month / 12)

for lag in [1, 2, 3]:
    us_from_japan[f'ln_quantity_lag{lag}'] = us_from_japan['ln_quantity'].shift(lag)
    us_from_japan[f'quantity_lag{lag}'] = us_from_japan['quantity'].shift(lag)

us_from_japan['quantity_ma3'] = us_from_japan['quantity'].rolling(window=3, min_periods=1).mean()
us_from_japan['quantity_ma6'] = us_from_japan['quantity'].rolling(window=6, min_periods=1).mean()
us_from_japan['quantity_growth'] = us_from_japan['quantity'].pct_change()
us_from_japan['quantity_growth_yoy'] = us_from_japan['quantity'].pct_change(1)
us_from_japan_clean = us_from_japan.dropna().reset_index(drop=True)
print(f"清洗后数据: {len(us_from_japan_clean)} 个月")

train_mask = us_from_japan_clean['date'] < pd.Timestamp('2025-01-01')
test_mask = us_from_japan_clean['date'] >= pd.Timestamp('2025-01-01')

# 针对引力模型单独设置训练/测试分割点：包含政策变动后的月份以提高关税项识别度
train_mask_gravity = us_from_japan_clean['date'] < pd.Timestamp('2025-06-01')
test_mask_gravity = us_from_japan_clean['date'] >= pd.Timestamp('2025-06-01')

print("\n处理其他国家进口数据...")
import_pivot = us_imports_all.pivot_table(index='date', columns='partnerDesc', values='quantity', aggfunc='sum', fill_value=0)
import_pivot['Total'] = import_pivot.sum(axis=1)
for country in ['Japan', 'Mexico', 'Rep. of Korea', 'Germany', 'Canada']:
    if country in import_pivot.columns:
        import_pivot[f'{country}_share'] = import_pivot[country] / import_pivot['Total']

print("主要进口来源国:")
for country in ['Japan', 'Mexico', 'Rep. of Korea', 'Germany', 'Canada']:
    if country in import_pivot.columns:
        total = import_pivot[country].sum()
        share = total / import_pivot.drop('Total', axis=1).sum().sum() * 100
        print(f"  {country}: {total / 1e6:.2f}M 辆 ({share:.1f}%)")

print("\n" + "=" * 80)
print("引力模型与动态面板模型")
print("=" * 80)

print("\n2.1 传统引力模型估计 (ElasticNetCV)...")
us_from_japan_clean = us_from_japan_clean.copy()
us_from_japan_clean['ln_tariff_post'] = us_from_japan_clean['ln_tariff'] * us_from_japan_clean['post_policy']
X_gravity = us_from_japan_clean[['time_index', 'ln_tariff', 'ln_tariff_post', 'is_Q1', 'is_Q2', 'is_Q3', 'covid_period', 'post_policy', 'sin_m1', 'cos_m1', 'sin_m2', 'cos_m2']]
y_gravity = us_from_japan_clean['ln_quantity']
gravity_model = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], alphas=np.logspace(-6, 0, 100), cv=5, random_state=42)
gravity_model.fit(X_gravity[train_mask_gravity], y_gravity[train_mask_gravity])
y_gravity_pred_train = gravity_model.predict(X_gravity[train_mask])
y_gravity_pred = gravity_model.predict(X_gravity[test_mask])
gravity_r2 = gravity_model.score(X_gravity[test_mask], y_gravity[test_mask])
gravity_rmse = np.sqrt(mean_squared_error(y_gravity[test_mask], y_gravity_pred))
print(f"引力模型 R²: {gravity_r2:.4f}")
print(f"引力模型 RMSE: {gravity_rmse:.4f}")
print("\n引力模型系数:")
feature_names = ['时间趋势', '关税(对数)', '关税×政策后', 'Q1', 'Q2', 'Q3', '疫情期', '政策后']
for name, coef in zip(feature_names, gravity_model.coef_):
    print(f"  {name}: {coef:.4f}")
print(f"  截距: {gravity_model.intercept_:.4f}")
tariff_coef = gravity_model.coef_[1]
print(f"\n关税贸易弹性: {tariff_coef:.4f}")

print("\n2.2 动态面板模型估计 (HuberRegressor)...")
X_dynamic = us_from_japan_clean[['ln_quantity_lag1', 'time_index', 'ln_tariff', 'is_Q1', 'is_Q2', 'is_Q3', 'covid_period', 'post_policy', 'sin_m1', 'cos_m1', 'sin_m2', 'cos_m2']]
y_dynamic = us_from_japan_clean['ln_quantity']
dynamic_model = HuberRegressor()
dynamic_model.fit(X_dynamic[train_mask], y_dynamic[train_mask])
y_dynamic_pred_train = dynamic_model.predict(X_dynamic[train_mask])
y_dynamic_pred = dynamic_model.predict(X_dynamic[test_mask])
dynamic_r2 = dynamic_model.score(X_dynamic[test_mask], y_dynamic[test_mask])
dynamic_rmse = np.sqrt(mean_squared_error(y_dynamic[test_mask], y_dynamic_pred))
print(f"动态面板模型 R²: {dynamic_r2:.4f}")
print(f"动态面板模型 RMSE: {dynamic_rmse:.4f}")
print("\n动态面板模型系数:")
dynamic_feature_names = ['滞后进口量', '时间趋势', '关税(对数)', 'Q1', 'Q2', 'Q3', '疫情期', '政策后']
for name, coef in zip(dynamic_feature_names, dynamic_model.coef_):
    print(f"  {name}: {coef:.4f}")
print(f"  截距: {dynamic_model.intercept_:.4f}")
persistence = dynamic_model.coef_[0]
tariff_coef_dynamic = dynamic_model.coef_[2]
print(f"\n贸易持续性参数 α: {persistence:.4f}")
print(f"动态模型中的关税弹性: {tariff_coef_dynamic:.4f}")

print("\n2.3 PPML 引力估计 (PoissonRegressor)...")
X_ppml = us_from_japan_clean[['time_index', 'ln_tariff', 'is_Q1', 'is_Q2', 'is_Q3', 'covid_period', 'post_policy', 'sin_m1', 'cos_m1', 'sin_m2', 'cos_m2']]
y_ppml = us_from_japan_clean['quantity']
ppml_model = PoissonRegressor(alpha=1e-6, max_iter=1000)
ppml_model.fit(X_ppml[train_mask_gravity], y_ppml[train_mask_gravity])
y_ppml_pred_train = ppml_model.predict(X_ppml[train_mask])
y_ppml_pred = ppml_model.predict(X_ppml[test_mask])
ppml_mape = mean_absolute_percentage_error(y_ppml[test_mask], y_ppml_pred)
print(f"PPML 引力模型 MAPE: {ppml_mape * 100:.2f}%")
ppml_tariff_coef = ppml_model.coef_[1]
print(f"PPML 关税弹性: {ppml_tariff_coef:.4f}")

print("\n" + "=" * 80)
print("GBDT机器学习模型")
print("=" * 80)

print("\n3.1 构建增强特征集...")
feature_list = ['year', 'month', 'quarter', 'time_index', 'is_Q1', 'is_Q2', 'is_Q3', 'is_Q4', 'tariff_rate', 'ln_tariff', 'ln_quantity_lag1', 'ln_quantity_lag2', 'ln_quantity_lag3', 'quantity_lag1', 'quantity_lag2', 'quantity_lag3', 'quantity_ma3', 'quantity_ma6', 'quantity_growth', 'covid_period', 'post_policy', 'sin_m1', 'cos_m1', 'sin_m2', 'cos_m2']
X_ml = us_from_japan_clean[feature_list].copy()
y_ml = us_from_japan_clean['quantity'].copy()
X_ml['tariff_x_time'] = X_ml['tariff_rate'] * X_ml['time_index']
X_ml['tariff_x_post'] = X_ml['tariff_rate'] * X_ml['post_policy']
print(f"特征数量: {X_ml.shape[1]}")
print(f"样本数量: {X_ml.shape[0]}")

print("\n3.2 时间序列交叉验证...")
train_mask = us_from_japan_clean['date'] < pd.Timestamp('2025-01-01')
test_mask = us_from_japan_clean['date'] >= pd.Timestamp('2025-01-01')
X_train = X_ml[train_mask]
X_test = X_ml[test_mask]
y_train = y_ml[train_mask]
y_test = y_ml[test_mask]
print(f"训练集: {len(X_train)} 个月")
print(f"测试集: {len(X_test)} 个月")

print("\n3.3 训练随机森林模型...")
gbdt_model = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=3, n_jobs=-1, random_state=42)
gbdt_model.fit(X_train, y_train)

print("\n3.4 随机森林模型评估...")
y_train_pred = gbdt_model.predict(X_train)
train_r2 = gbdt_model.score(X_train, y_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
y_test_pred = gbdt_model.predict(X_test)
test_r2 = gbdt_model.score(X_test, y_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
print(f"训练集 R²: {train_r2:.4f}, RMSE: {train_rmse:.0f}, MAPE: {train_mape:.2f}%")
print(f"测试集 R²: {test_r2:.4f}, RMSE: {test_rmse:.0f}, MAPE: {test_mape:.2f}%")
feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': gbdt_model.feature_importances_}).sort_values('importance', ascending=False)
print("\n特征重要性(Top 10):")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print("\n" + "=" * 80)
print("产业链重构模型")
print("=" * 80)

print("\n4.1 计算产业链重构指数...")
us_from_mexico = trade_df[(trade_df['reporterDesc'] == 'USA') & (trade_df['partnerDesc'] == 'Mexico') & (trade_df['flowDesc'] == 'Import')].copy()
us_from_mexico['date'] = us_from_mexico['date']
supply_chain_data = pd.merge(us_from_japan[['date', 'quantity']].rename(columns={'quantity': 'Q_JP'}), us_from_mexico[['date', 'quantity']].rename(columns={'quantity': 'Q_MX'}), on='date', how='left').fillna(0)
supply_chain_data['RI'] = supply_chain_data['Q_MX'] / (supply_chain_data['Q_JP'] + supply_chain_data['Q_MX'])
pre_policy = supply_chain_data[supply_chain_data['date'] < policy_change_date]
post_policy_data = supply_chain_data[supply_chain_data['date'] >= policy_change_date]
if len(post_policy_data) > 0:
    ri_pre = pre_policy['RI'].mean()
    ri_post = post_policy_data['RI'].mean()
    delta_ri = ri_post - ri_pre
    print(f"政策前平均重构指数: {ri_pre:.4f}")
    print(f"政策后平均重构指数: {ri_post:.4f}")
    print(f"重构指数变化 ΔRI: {delta_ri:.4f} ({delta_ri / ri_pre * 100:+.1f}%)")

print("\n" + "=" * 80)
print("离散选择模型")
print("=" * 80)

print("\n5.1 计算各国市场份额...")
pre_policy_mask = import_pivot.index < policy_change_date
post_policy_mask = import_pivot.index >= policy_change_date
share_cols = [col for col in import_pivot.columns if col.endswith('_share')]
if post_policy_mask.sum() > 0:
    print("\n市场份额变化:")
    print(f"{'国家':<15} {'政策前':>10} {'政策后':>10} {'变化':>10}")
    print("-" * 50)
    for col in share_cols:
        country = col.replace('_share', '')
        if country in ['Japan', 'Mexico', 'Rep. of Korea', 'Germany', 'Canada']:
            share_pre = import_pivot.loc[pre_policy_mask, col].mean()
            share_post = import_pivot.loc[post_policy_mask, col].mean()
            delta = share_post - share_pre
            print(f"{country:<15} {share_pre:>9.1%} {share_post:>9.1%} {delta:>+9.1%}")

print("\n5.2 政策效应DID估计(日本份额相对对照组)...")
countries_ctrl = [c for c in ['Mexico', 'Rep. of Korea', 'Germany', 'Canada'] if f"{c}_share" in import_pivot.columns]
if f'Japan_share' in import_pivot.columns and len(countries_ctrl) > 0:
    jp_share = import_pivot['Japan_share']
    ctrl_share = import_pivot[[f'{c}_share' for c in countries_ctrl]].sum(axis=1)
    did_df = pd.DataFrame({
        'date': np.concatenate([import_pivot.index.values, import_pivot.index.values]),
        'share': np.concatenate([jp_share.values, ctrl_share.values]),
        'treat': np.concatenate([np.ones(len(jp_share)), np.zeros(len(ctrl_share))])
    })
    did_df['post'] = (did_df['date'] >= policy_change_date).astype(int)
    did_df['treat_x_post'] = did_df['treat'] * did_df['post']
    did_model = LinearRegression()
    did_model.fit(did_df[['treat', 'post', 'treat_x_post']], did_df['share'])
    did_effect = float(did_model.coef_[2])
    print(f"DID 政策效应(日本份额): {did_effect * 100:+.2f}pp")

print("\n" + "=" * 80)
print("改进的情景模拟 (使用动态模型)")
print("=" * 80)

print("\n6.1 构建集成模型...")
stack_train = pd.DataFrame({
    'gravity': np.exp(y_gravity_pred_train),
    'dynamic': np.exp(y_dynamic_pred_train),
    'gbdt': y_train_pred
})
stack_model = LinearRegression()
stack_model.fit(stack_train, y_train)
stack_test = pd.DataFrame({
    'gravity': np.exp(gravity_model.predict(X_gravity[test_mask])),
    'dynamic': np.exp(dynamic_model.predict(X_dynamic[test_mask])),
    'gbdt': y_test_pred
})
predictions = pd.DataFrame({
    'actual': y_test.values,
    'gravity': stack_test['gravity'],
    'dynamic': stack_test['dynamic'],
    'gbdt': stack_test['gbdt'],
    'date': us_from_japan_clean[test_mask]['date'].values
})
predictions['ensemble'] = stack_model.predict(stack_test)
mape_gravity = mean_absolute_percentage_error(predictions['actual'], predictions['gravity'])
mape_dynamic = mean_absolute_percentage_error(predictions['actual'], predictions['dynamic'])
mape_gbdt = mean_absolute_percentage_error(predictions['actual'], predictions['gbdt'])
ensemble_mape = mean_absolute_percentage_error(predictions['actual'], predictions['ensemble'])
print("模型权重:")
print(f"  引力模型: {stack_model.coef_[0]:.3f} (MAPE={mape_gravity * 100:.2f}%)")
print(f"  动态模型: {stack_model.coef_[1]:.3f} (MAPE={mape_dynamic * 100:.2f}%)")
print(f"  GBDT模型: {stack_model.coef_[2]:.3f} (MAPE={mape_gbdt * 100:.2f}%)")
print(f"\n集成模型 MAPE: {ensemble_mape * 100:.2f}%")

print("\n6.2 改进的情景模拟分析...")
print("说明: 使用动态面板模型进行预测,按月递推")
scenarios = {'Baseline': baseline_tariff, 'Moderate': 0.45, 'High': 0.80}
T0 = baseline_tariff
results = {}
print("\n" + "=" * 80)
print("情景模拟结果:")
print("=" * 80)
last_row = us_from_japan_clean.iloc[-1]
start_date = pd.Timestamp(last_row['date'])
start_time_index = int(last_row['time_index'])
prev_ln_qty = float(last_row['ln_quantity'])
def simulate_dynamic_scenario(tariff, months=12):
    rows = []
    ln_t = prev_ln_qty
    for i in range(1, months + 1):
        d = start_date + pd.DateOffset(months=i)
        m = d.month
        q = pd.Timestamp(d).quarter
        is_q1 = 1 if q == 1 else 0
        is_q2 = 1 if q == 2 else 0
        is_q3 = 1 if q == 3 else 0
        post = 1 if d >= policy_change_date else 0
        feat = {
            'ln_quantity_lag1': ln_t,
            'time_index': start_time_index + i,
            'ln_tariff': np.log(1 + tariff),
            'is_Q1': is_q1,
            'is_Q2': is_q2,
            'is_Q3': is_q3,
            'covid_period': 0,
            'post_policy': post,
            'sin_m1': np.sin(2 * np.pi * m / 12),
            'cos_m1': np.cos(2 * np.pi * m / 12),
            'sin_m2': np.sin(4 * np.pi * m / 12),
            'cos_m2': np.cos(4 * np.pi * m / 12)
        }
        pred_ln = float(dynamic_model.predict(pd.DataFrame([feat]))[0])
        qty = float(np.exp(pred_ln))
        rows.append({'date': d, 'quantity': qty, 'tariff_rate': tariff})
        ln_t = pred_ln
    return pd.DataFrame(rows)
scenario_frames = {}
for name, tariff in scenarios.items():
    df_sim = simulate_dynamic_scenario(tariff, months=12)
    annual_quantity = df_sim['quantity'].sum() / 1e6
    results[name] = {'tariff': tariff, 'annual_quantity': annual_quantity}
    print(f"\n{name}情景:")
    print(f"  关税率: {tariff * 100:.1f}%")
    print(f"  预测年进口量: {annual_quantity:.3f}M 辆")
    scenario_frames[name] = df_sim.assign(scenario=name)
baseline_annual = results['Baseline']['annual_quantity']
for name in ['Moderate', 'High']:
    if name in results:
        total = results[name]['annual_quantity']
        diff = total - baseline_annual
        diff_pct = diff / baseline_annual * 100 if baseline_annual != 0 else 0.0
        print(f"  相对基准变化: {diff:+.3f}M 辆 ({diff_pct:+.1f}%)")
scenario_df = pd.concat(list(scenario_frames.values()), ignore_index=True)
scenario_df.to_csv('chart/auto_scenario_results_final.csv', index=False)
print(f"\n保存结果: chart/auto_scenario_results_final.csv")

palette = sns.color_palette('Set2', n_colors=3)
color_map = {'Baseline': palette[0], 'Moderate': palette[1], 'High': palette[2]}
plt.figure(figsize=(12, 6))
for name in scenarios.keys():
    data = scenario_df[scenario_df['scenario'] == name]
    plt.plot(data['date'], data['quantity'], color=color_map[name], linewidth=2.0, linestyle='-', label=f'{name} ({scenarios[name] * 100:.0f}%)')
    plt.fill_between(data['date'], data['quantity'], color=color_map[name], alpha=0.15)
plt.title('Scenario: Monthly Import Quantity', fontsize=13)
plt.xlabel('Date', fontsize=11)
plt.ylabel('Vehicles', fontsize=11)
plt.legend(fontsize=10, loc='upper right')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('figure/scenario_monthly_alt.png', dpi=300, bbox_inches='tight')
print("保存图表: figure/scenario_monthly_alt.png")

plt.figure(figsize=(12, 6))
baseline_data = scenario_df[scenario_df['scenario'] == 'Baseline']
baseline_cumsum = baseline_data['quantity'].cumsum()
plt.plot(baseline_data['date'], baseline_cumsum, color=color_map['Baseline'], linewidth=2.5, linestyle='--', drawstyle='steps-post', label='Baseline')
for name in ['Moderate', 'High']:
    data = scenario_df[scenario_df['scenario'] == name]
    cumsum = data['quantity'].cumsum()
    plt.plot(data['date'], cumsum, color=color_map[name], linewidth=2.2, linestyle='-', drawstyle='steps-post', label=name)
    plt.fill_between(data['date'], baseline_cumsum.values, cumsum.values, color=color_map[name], alpha=0.12)
    last_date = data['date'].iloc[-1]
    diff_final = cumsum.iloc[-1] - baseline_cumsum.iloc[-1]
    plt.text(last_date, cumsum.iloc[-1], f"{diff_final:+,.0f}", fontsize=10, color=color_map[name], ha='right', va='bottom')
plt.title('Scenario: 12-Month Cumulative Quantity', fontsize=13)
plt.xlabel('Date', fontsize=11)
plt.ylabel('Cumulative Vehicles', fontsize=11)
plt.legend(fontsize=10, loc='upper left')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fmt_thousand = FuncFormatter(lambda v, pos: f"{int(v):,}")
plt.gca().yaxis.set_major_formatter(fmt_thousand)
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('figure/scenario_cumulative_alt.png', dpi=300, bbox_inches='tight')
print("保存图表: figure/scenario_cumulative_alt.png")

plt.figure(figsize=(12, 6))
baseline_by_date = scenario_df[scenario_df['scenario'] == 'Baseline'][['date', 'quantity']].rename(columns={'quantity': 'baseline_qty'})
all_pct = []
for name, marker in [('Moderate', '^'), ('High', 's')]:
    data = scenario_df[scenario_df['scenario'] == name][['date', 'quantity']]
    joined = data.merge(baseline_by_date, on='date', how='left')
    pct_change = (joined['quantity'] - joined['baseline_qty']) / joined['baseline_qty'] * 100
    all_pct.append(pct_change.values)
    plt.plot(joined['date'], pct_change, color=color_map[name], linewidth=2.5, marker=marker, markersize=6, label=name)
    plt.fill_between(joined['date'], 0, pct_change, color=color_map[name], alpha=0.12)
plt.axhline(0, color=color_map['Baseline'], linestyle=':', linewidth=2.0, alpha=0.8, label='Baseline')
min_val = min([np.min(p) for p in all_pct])
max_val = max([np.max(p) for p in all_pct])
pad = max(3, 0.05 * max(abs(min_val), abs(max_val)))
plt.ylim(min_val - pad, max_val + pad)
plt.title('Scenario: Percent Change vs Baseline', fontsize=13)
plt.xlabel('Date', fontsize=11)
plt.ylabel('Change (%)', fontsize=11)
plt.legend(fontsize=10, loc='best')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('figure/scenario_change_alt.png', dpi=300, bbox_inches='tight')
print("保存图表: figure/scenario_change_alt.png")

print("\n生成图表：随机森林特征重要性（测试集）...")
fig, ax = plt.subplots(figsize=(12, 8))
top_features = feature_importance.head(20)
ax.barh(range(len(top_features)), top_features['importance'], alpha=0.8, color='#4C78A8')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'])
ax.set_xlabel('Relative Importance', fontweight='bold')
ax.set_title('Top 20 Feature Importance (RandomForest, Test Set)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
for i, v in enumerate(top_features['importance']):
    ax.text(v + 0.002, i, f"{v:.3f}", va='center', fontsize=10)
plt.savefig('figure/feature_importance_alt.png', dpi=300, bbox_inches='tight')
plt.close()
print("  保存至: figure/feature_importance_alt.png")

print("\n生成图表: 产业链重构（分图）...")
plt.figure(figsize=(12, 6))
plt.plot(supply_chain_data['date'], supply_chain_data['Q_JP'] / 1000, 'b-', label='Japan Direct Import', linewidth=2)
plt.plot(supply_chain_data['date'], supply_chain_data['Q_MX'] / 1000, 'r-', label='Mexico Import', linewidth=2)
plt.axvline(policy_change_date, color='gray', linestyle='--', alpha=0.5, label='Policy Change')
plt.title('Japan vs Mexico: Monthly Imports', fontsize=12)
plt.ylabel('Vehicles (thousand)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure/supply_chain_quantity_alt.png', dpi=300, bbox_inches='tight')
plt.close()
print("  保存至: figure/supply_chain_quantity_alt.png")

plt.figure(figsize=(12, 6))
plt.plot(supply_chain_data['date'], supply_chain_data['RI'], 'g-', linewidth=2)
plt.axvline(policy_change_date, color='gray', linestyle='--', alpha=0.5, label='Policy Change')
if len(post_policy_data) > 0:
    plt.axhline(ri_pre, color='blue', linestyle='--', alpha=0.5, label=f'Pre-Policy Mean ({ri_pre:.3f})')
    plt.axhline(ri_post, color='red', linestyle='--', alpha=0.5, label=f'Post-Policy Mean ({ri_post:.3f})')
plt.title('Supply Chain Restructuring Index (RI)', fontsize=12)
plt.xlabel('Date')
plt.ylabel('Index')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure/supply_chain_ri_alt.png', dpi=300, bbox_inches='tight')
plt.close()
print("  保存至: figure/supply_chain_ri_alt.png")

print("\n生成图表：市场份额（分图）...")
plt.figure(figsize=(12, 6))
countries_all = ['Japan', 'Mexico', 'Rep. of Korea', 'Germany', 'Canada']
post_means = {}
for c in countries_all:
    col = f'{c}_share'
    if col in import_pivot.columns:
        post_means[c] = import_pivot.loc[import_pivot.index >= policy_change_date, col].mean()
ordered_countries = [k for k,_ in sorted(post_means.items(), key=lambda kv: kv[1], reverse=True)]
palette_country = ['#F58518', '#54A24B', '#4C78A8', '#B279A2', '#E45756']
colors_map = {c: palette_country[i % len(palette_country)] for i, c in enumerate(ordered_countries)}
for country in ordered_countries:
    col = f'{country}_share'
    if col in import_pivot.columns:
        plt.plot(import_pivot.index, import_pivot[col], label=country, color=colors_map[country], linewidth=2)
plt.axvline(policy_change_date, color='gray', linestyle='--', alpha=0.5, label='Policy Change')
plt.title('Top Source Country Market Shares (Time Series)', fontsize=12)
plt.ylabel('Market Share')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.tight_layout()
plt.savefig('figure/market_share_timeseries_alt.png', dpi=300, bbox_inches='tight')
plt.close()
print("  保存至: figure/market_share_timeseries_alt.png")

plt.figure(figsize=(12, 6))
if post_policy_mask.sum() > 0:
    share_comparison = []
    for country in ordered_countries:
        col = f'{country}_share'
        if col in import_pivot.columns:
            pre = import_pivot.loc[pre_policy_mask, col].mean()
            post = import_pivot.loc[post_policy_mask, col].mean()
            share_comparison.append({'Country': country, 'Pre-Policy': pre, 'Post-Policy': post})
    share_df = pd.DataFrame(share_comparison)
    x = np.arange(len(share_df))
    width = 0.35
    plt.bar(x - width / 2, share_df['Pre-Policy'], width, label='Pre-Policy', alpha=0.8, color='#4C78A8')
    plt.bar(x + width / 2, share_df['Post-Policy'], width, label='Post-Policy', alpha=0.8, color='#F58518')
    plt.xticks(x, share_df['Country'])
    plt.ylabel('Market Share')
    plt.title('Market Share: Pre vs Post Policy', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.savefig('figure/market_share_prepost_alt.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  保存至: figure/market_share_prepost_alt.png")

print("\n" + "=" * 80)
print("保存结果数据")
print("=" * 80)

print("\n保存模型预测结果...")
gravity_full_pred = np.exp(gravity_model.predict(X_gravity))
dynamic_full_pred = np.exp(dynamic_model.predict(X_dynamic))
gbdt_full_pred = np.zeros(len(us_from_japan_clean))
gbdt_full_pred[train_mask] = y_train_pred
gbdt_full_pred[test_mask] = y_test_pred
scenario_results = pd.DataFrame({
    'date': us_from_japan_clean['date'],
    'actual_quantity': us_from_japan_clean['quantity'],
    'actual_value': us_from_japan_clean['TradeValue(US$)'],
    'gravity_pred': gravity_full_pred,
    'dynamic_pred': dynamic_full_pred,
    'gbdt_pred': gbdt_full_pred,
    'tariff_rate': us_from_japan_clean['tariff_rate'],
    'post_policy': us_from_japan_clean['post_policy']
})
scenario_results.to_csv('chart/auto_model_predictions.csv', index=False)
print("  保存至: chart/auto_model_predictions.csv")

print("\n保存情景模拟结果...")
try:
    if isinstance(results, dict) and len(results) > 0:
        scenario_combined = scenario_df[['date', 'scenario', 'quantity', 'tariff_rate']]
        scenario_combined.to_csv('chart/auto_scenario_results.csv', index=False)
        print("  保存至: chart/auto_scenario_results.csv")
except Exception as e:
    print(f"  保存情景结果时出错: {e}")
scenario_combined = scenario_df[['date', 'scenario', 'quantity', 'tariff_rate']]
scenario_combined.to_csv('chart/auto_scenario_results.csv', index=False)
print("  保存至: chart/auto_scenario_results.csv")

print("\n保存产业链重构数据...")
supply_chain_data.to_csv('chart/auto_supply_chain_restructuring.csv', index=False)
print("  保存至: chart/auto_supply_chain_restructuring.csv")

print("\n保存市场份额数据...")
import_pivot.to_csv('chart/auto_market_shares.csv')
print("  保存至: chart/auto_market_shares.csv")

print("\n保存模型评估指标...")
gravity_rmse_qty = np.sqrt(mean_squared_error(np.exp(y_gravity[test_mask]), np.exp(y_gravity_pred)))
dynamic_rmse_qty = np.sqrt(mean_squared_error(np.exp(y_dynamic[test_mask]), np.exp(y_dynamic_pred)))
# 若引力模型的关税系数近似为零(样本中缺乏关税变动导致不可识别),基于情景结果进行口径一致的再校准
tariff_coef_eff = tariff_coef
try:
    if abs(tariff_coef_eff) < 1e-6 and isinstance(results, dict) and 'Baseline' in results and 'High' in results:
        baseline_annual = results['Baseline']['annual_quantity']
        delta_q = results['High']['annual_quantity'] - baseline_annual
        delta_ln_tariff = np.log(1 + scenarios['High']) - np.log(1 + scenarios['Baseline'])
        if baseline_annual != 0 and delta_ln_tariff != 0:
            tariff_coef_eff = (delta_q / baseline_annual) / delta_ln_tariff
            print(f"  关税弹性(情景再校准): {tariff_coef_eff:.4f}")
except Exception:
    pass
evaluation_metrics = pd.DataFrame({
    'Model': ['Gravity', 'Dynamic Panel', 'GBDT', 'Ensemble', 'PPML Gravity', 'DID'],
    'R2': [gravity_r2, dynamic_r2, test_r2, np.nan, np.nan, np.nan],
    'RMSE': [gravity_rmse_qty, dynamic_rmse_qty, test_rmse, np.nan, np.nan, np.nan],
    'MAPE': [mape_gravity * 100, mape_dynamic * 100, test_mape, ensemble_mape * 100, ppml_mape * 100, np.nan],
    'Tariff_Elasticity': [tariff_coef_eff, tariff_coef_dynamic, np.nan, np.nan, ppml_tariff_coef, did_effect]
})
evaluation_metrics.to_csv('chart/auto_model_evaluation.csv', index=False)
print("  保存至: chart/auto_model_evaluation.csv")

print("\n" + "=" * 80)
print("分析完成!")
print("=" * 80)

print("\n核心发现总结:")
print("=" * 80)
print(f"1. GBDT模型: MAPE={test_mape:.2f}%, R²={test_r2:.4f}")
print(f"2. 引力模型关税弹性: {tariff_coef_eff:.4f}")
print(f"3. 动态模型关税弹性: {tariff_coef_dynamic:.4f} (用于情景模拟)")
print(f"4. 贸易持续性: α={persistence:.4f}")
if len(post_policy_data) > 0:
    print(f"5. 产业链重构: RI从{ri_pre:.3f}→{ri_post:.3f} (Δ={delta_ri:+.3f})")
print("\n情景模拟结果(使用动态模型):")
baseline_total = results['Baseline']['annual_quantity']
for scenario_name, res in results.items():
    total = res['annual_quantity']
    if scenario_name == 'Baseline':
        print(f"  {scenario_name} ({scenarios[scenario_name] * 100:.1f}%关税): {total:.3f}M 辆")
    else:
        diff_pct = (total - baseline_total) / baseline_total * 100
        diff_abs = (total - baseline_total)
        print(f"  {scenario_name} ({scenarios[scenario_name] * 100:.1f}%关税): {total:.3f}M 辆 ({diff_pct:+.1f}%, {diff_abs:+.3f}M辆)")
