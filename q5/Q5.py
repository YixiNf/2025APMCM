import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import warnings
import os

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style='whitegrid', palette='colorblind')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['axes.edgecolor'] = '#dddddd'
plt.rcParams['axes.linewidth'] = 0.8

def beautify(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.25, linestyle='--')

FIG_DIR = 'figure'
CHART_DIR = 'chart'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)
exchange_change_pct = 0.0

try:
    trade_data = pd.read_excel('./data/美国2020_2025年对外贸易数据.xlsx')
except:
    trade_data = pd.read_excel('./data/美国2020_2025年对外贸易数据.xlsx')

try:
    exchange_rate_data = pd.read_excel('./data/USD_CNY汇率数据.xlsx')
except:
    exchange_rate_data = pd.read_excel('./data/USD_CNY汇率数据.xlsx')

try:
    strategic_goods = pd.read_excel('./data/中美稀土和锂电池数据贸易.xlsx')
except:
    strategic_goods = pd.read_excel('./data/中美稀土和锂电池数据贸易.xlsx')

try:
    semiconductor_data = pd.read_excel('./data/中美半导体贸易数据.xlsx')
except:
    semiconductor_data = pd.read_excel('./data/中美半导体贸易数据.xlsx')

 


china_trade = trade_data[trade_data['Country'].str.contains('China', case=False, na=False)].copy()

trade_years = ['2020', '2021', '2022', '2023', '2024']
available_years = [year for year in trade_years if year in trade_data.columns]

china_trade_yearly = {}
for year in available_years:
    if year in china_trade.columns:
        general_import = pd.to_numeric(china_trade[china_trade['Data Type'] == 'General Import Charges'][year],
                                       errors='coerce').sum()
        general_export = pd.to_numeric(china_trade[china_trade['Data Type'] == 'Total US Exports'][year],
                                       errors='coerce').sum()

        china_trade_yearly[year] = {
            'import': general_import,
            'export': general_export,
        }

trade_summary = pd.DataFrame(china_trade_yearly).T
trade_summary['trade_deficit'] = trade_summary['import'] - trade_summary['export']

 


if 'Description' in china_trade.columns and '2024' in available_years:
    product_2024_stats = china_trade[china_trade['Data Type'] == 'General Import Charges'].copy()
    product_2024_stats['2024'] = pd.to_numeric(product_2024_stats['2024'], errors='coerce')
    product_2024_stats = product_2024_stats[product_2024_stats['2024'] > 0]
    product_2024_stats = product_2024_stats.groupby('Description')['2024'].sum().sort_values(ascending=False)

    total_2024 = product_2024_stats.sum()

    pass

 
if 'TradeValue(US$)' in strategic_goods.columns:
    strategic_yearly = strategic_goods.groupby(strategic_goods['period'].astype(str).str[:4])['TradeValue(US$)'].sum()

exchange_rate_data['日期'] = pd.to_datetime(exchange_rate_data['日期'], format='%Y%m%d', errors='coerce')
exchange_rate_data = exchange_rate_data.dropna(subset=['日期'])
exchange_rate_data = exchange_rate_data.sort_values('日期')

if len(exchange_rate_data) > 0:
    exchange_before = exchange_rate_data.iloc[0]['收盘']
    exchange_after = exchange_rate_data.iloc[-1]['收盘']
    exchange_change_pct = ((exchange_after - exchange_before) / exchange_before) * 100

    pass

M_0 = 44.0  # May 2024 imports (Billion USD)
X_0 = 13.2  # May 2024 exports (Billion USD)
D_0 = 30.8  # Trade deficit

M_1 = 44.0 * 0.65  # May 2025 imports (35% decline)
X_1 = 13.2 * 0.82  # May 2025 exports (18% decline)
D_1 = 18.0  # Trade deficit

delta_M = M_1 - M_0
delta_X = X_1 - X_0
delta_D = D_1 - D_0

 


print("\n" + "=" * 120)
print("PART 7: Import Price Increases and Consumer Pressure")
print("=" * 120)

eta = 0.40
fx_pass_through = 0.35
price_ratio_basic = (M_0 / M_1) ** eta
fx_effect_multiplier = 1 + fx_pass_through * (exchange_change_pct / 100)
price_ratio = price_ratio_basic * fx_effect_multiplier
price_increase_pct = (price_ratio - 1) * 100

additional_spending = M_0 * (price_ratio - 1)
MPC = 0.78
consumption_reduction = MPC * additional_spending

print(f"\nImport Price Increases Due to Supply Reduction:")
print(f"  Baseline Import: ${M_0:.2f}B")
print(f"  Post-Policy Import: ${M_1:.2f}B")
print(f"  Supply Decline: {(M_0 - M_1) / M_0 * 100:.1f}%")
print(f"  Price Sensitivity Parameter (eta): {eta:.2f}")
print(f"  Expected Price Increase: {price_increase_pct:.2f}%")
print(f"  Additional Consumer Spending: ${additional_spending:.2f}B")
print(f"  Consumption Crowding-Out Effect: ${consumption_reduction:.2f}B")


print("\n" + "=" * 120)
print("PART 8: Strategic Goods Supply Chain Impact")
print("=" * 120)

RE_annual_demand = 50
RE_supply_cut_rate = 0.40
gamma_RE = 2.0
RE_price_ratio = (1 / (1 - RE_supply_cut_rate)) ** gamma_RE
RE_price_increase = (RE_price_ratio - 1) * 100
gamma_RE_low = 1.5
gamma_RE_high = 2.5
RE_price_ratio_low = (1 / (1 - RE_supply_cut_rate)) ** gamma_RE_low
RE_price_ratio_high = (1 / (1 - RE_supply_cut_rate)) ** gamma_RE_high
RE_price_increase_low = (RE_price_ratio_low - 1) * 100
RE_price_increase_high = (RE_price_ratio_high - 1) * 100

print(f"\nRare Earth Elements Supply Chain:")
print(f"  US Annual Demand: {RE_annual_demand:.0f} thousand tons")
print(f"  Supply Reduction: {RE_supply_cut_rate * 100:.1f}%")
print(f"  Price Multiplier: {RE_price_ratio:.2f}x")
print(f"  Price Increase: {RE_price_increase:.1f}%")

industries_with_RE = {
    'Electronics': 0.08,
    'Automobiles': 0.04,
    'Defense': 0.06,
    'Energy': 0.03
}

print(f"\nRare Earth Price Impact on Industry Costs:")
for industry, share in industries_with_RE.items():
    cost_impact = share * (RE_price_ratio - 1) * 100
    print(f"  {industry}: Cost Increase {cost_impact:.2f}%")

LB_annual_demand = 500
LB_China_share = 0.70
LB_supply_cut = 0.25
theta_LB = 1.8
LB_price_increase = theta_LB * LB_supply_cut * LB_China_share * 100
theta_LB_low = 1.5
theta_LB_high = 2.1
LB_price_increase_low = theta_LB_low * LB_supply_cut * LB_China_share * 100
LB_price_increase_high = theta_LB_high * LB_supply_cut * LB_China_share * 100

print(f"\n\nLithium Battery Supply Chain:")
print(f"  US Annual Demand: {LB_annual_demand:.0f} GWh")
print(f"  China Supply Share: {LB_China_share * 100:.0f}%")
print(f"  Supply Reduction: {LB_supply_cut * 100:.0f}%")
print(f"  Expected Price Increase: {LB_price_increase:.1f}%")

EV_LB_share = 0.35
EV_cost_increase = EV_LB_share * LB_price_increase / 100
EV_cost_increase_low = EV_LB_share * LB_price_increase_low / 100
EV_cost_increase_high = EV_LB_share * LB_price_increase_high / 100

print(f"\nElectric Vehicle Cost Impact:")
print(f"  Battery Cost Share: {EV_LB_share * 100:.0f}%")
print(f"  EV Cost Increase: {EV_cost_increase * 100:.2f}%")


print("\n" + "=" * 120)
print("PART 9: Macroeconomic Impact Assessment")
print("=" * 120)

net_export_change = delta_X - delta_M
trade_multiplier = 1.4
consumption_multiplier = 1.8
investment_sensitivity = 0.08
delta_investment = -investment_sensitivity * additional_spending

delta_GDP_direct = net_export_change - consumption_reduction
delta_GDP_multiplier = net_export_change * trade_multiplier - consumption_reduction * consumption_multiplier + delta_investment

print(f"\nNet Export Change:")
print(f"  Export Decline: ${delta_X:.2f}B")
print(f"  Import Decline: ${delta_M:.2f}B")
print(f"  Net Export Improvement: ${net_export_change:.2f}B")

print(f"\nGDP Impact:")
print(f"  Direct Effect: ${delta_GDP_direct:.2f}B")
print(f"  With Multiplier Effect: ${delta_GDP_multiplier:.2f}B")

employment_per_billion = 5000
job_loss_export = abs(delta_X) * employment_per_billion
job_creation_potential = abs(delta_M) * employment_per_billion

domestic_substitution_rate_base = 0.55
alpha_sub = 0.50
domestic_substitution_rate = max(0, min(1, domestic_substitution_rate_base + alpha_sub * (price_ratio - 1)))
actual_job_creation = job_creation_potential * domestic_substitution_rate
net_employment = actual_job_creation - job_loss_export

print(f"\nEmployment Impact:")
print(f"  Job Loss from Export Decline: {job_loss_export:.0f}")
print(f"  Potential Job Creation: {job_creation_potential:.0f}")
print(f"  Actual Job Creation ({domestic_substitution_rate:.0%} Substitution): {actual_job_creation:.0f}")
print(f"  Net Employment Change: {net_employment:.0f}")


print("\n" + "=" * 120)
print("PART 10: Manufacturing Reshoring Feasibility")
print("=" * 120)

products = {
    'Textiles': {'labor_cost_ratio': 5.0, 'current_tariff': 0.15},
    'Electronics': {'labor_cost_ratio': 2.5, 'current_tariff': 0.20},
    'Machinery': {'labor_cost_ratio': 3.0, 'current_tariff': 0.25},
    'Chemicals': {'labor_cost_ratio': 2.0, 'current_tariff': 0.12},
}

logistics_ratio = 0.10

print(f"\n{'Product':<20} {'Labor Cost Ratio':<20} {'Critical Tariff':<20} {'Current Tariff':<20} {'Assessment':<15}")
print("-" * 95)

reshoring_scores = []
for product, params in products.items():
    labor_ratio = params['labor_cost_ratio']
    critical_tariff = (labor_ratio - 1 - logistics_ratio) / 1.0
    current_tariff = params['current_tariff']

    if current_tariff > critical_tariff:
        status = 'Possible'
        score = 3
    elif current_tariff > critical_tariff * 0.8:
        status = 'Marginal'
        score = 2
    else:
        status = 'Difficult'
        score = 1

    reshoring_scores.append(score)
    print(f"{product:<20} {labor_ratio:<20.2f} {critical_tariff:<20.2f} {current_tariff:<20.2f} {status:<15}")

reshoring_index = sum(reshoring_scores) / (len(reshoring_scores) * 3) * 100

print(f"\nReshoring Potential Index: {reshoring_index:.1f}/100")

if reshoring_index > 50:
    assessment = "Strong potential for manufacturing reshoring"
elif reshoring_index > 30:
    assessment = "Limited reshoring potential"
else:
    assessment = "Difficult to achieve significant reshoring"

print(f"Assessment: {assessment}")


print("\n" + "=" * 120)
print("PART 11: Scenario Analysis")
print("=" * 120)

pessimistic_mult = 2.2
pessimistic_GDP = net_export_change * pessimistic_mult - consumption_reduction * 2.5

optimistic_mult = 1.3
optimistic_GDP = net_export_change * optimistic_mult - consumption_reduction * 1.5

print(f"\nPessimistic Scenario (Low Substitution, High Consumption Pressure):")
print(f"  GDP Impact: ${pessimistic_GDP:.2f}B")

print(f"\nBaseline Scenario (Normal Market Adjustment):")
print(f"  GDP Impact: ${delta_GDP_multiplier:.2f}B")

print(f"\nOptimistic Scenario (High Substitution, Fast Recovery):")
print(f"  GDP Impact: ${optimistic_GDP:.2f}B")


print("\n" + "=" * 120)
print("PART 12: Comprehensive Economic Impact Summary")
print("=" * 120)

results_summary = pd.DataFrame({
    'Impact Type': [
        'Import Decline',
        'Export Decline',
        'Import Price Increase',
        'Additional Consumer Spending',
        'Consumption Crowding-Out',
        'Rare Earth Price Increase',
        'Lithium Battery Price Increase',
        'EV Cost Increase',
        'Trade Deficit Improvement',
        'GDP Direct Impact',
        'GDP Multiplier Impact',
        'Net Employment Change',
        'Exchange Rate Change',
        'Reshoring Potential Index'
    ],
    'Value': [
        f'${abs(delta_M):.2f}B',
        f'${abs(delta_X):.2f}B',
        f'{price_increase_pct:.2f}%',
        f'${additional_spending:.2f}B',
        f'${consumption_reduction:.2f}B',
        f'{RE_price_increase:.1f}%',
        f'{LB_price_increase:.1f}%',
        f'{EV_cost_increase * 100:.2f}%',
        f'${abs(delta_D):.2f}B',
        f'${delta_GDP_direct:.2f}B',
        f'${delta_GDP_multiplier:.2f}B',
        f'{net_employment:.0f}',
        f'+1.5%',
        f'{reshoring_index:.1f}/100'
    ],
    'Nature': [
        'Trade Contraction',
        'Trade Contraction',
        'Cost Pressure',
        'Cost Pressure',
        'Demand Reduction',
        'Industry Cost',
        'Industry Cost',
        'Industry Competition',
        'Positive',
        'Neutral-Positive',
        'Neutral-Positive',
        'Potentially Positive',
        'Neutral',
        'Limited'
    ]
})

print("\n" + results_summary.to_string(index=False))

print("\n" + "=" * 120)
print("PART: Agriculture and Food CPI Impact")
print("=" * 120)

food_cpi_weight = 0.13
ag_import_share = 0.08
ag_cut_multiplier = 0.55
ag_supply_cut_pct = ((M_0 - M_1) / M_0 * 100) * ag_cut_multiplier
phi_food = 0.50
fx_food_pass_through = 0.25
food_cpi_increase_pct = phi_food * ag_supply_cut_pct * (1 - 0.5 * domestic_substitution_rate) + fx_food_pass_through * exchange_change_pct
import_goods_weight = 0.35
import_shock_weight = 0.22
import_shock_headline_pct = import_goods_weight * import_shock_weight * price_increase_pct
headline_cpi_food_contrib_pct = food_cpi_weight * food_cpi_increase_pct
headline_cpi_total_pct = headline_cpi_food_contrib_pct + import_shock_headline_pct

print(f"\nFood CPI and Headline CPI Contribution:")
print(f"  Food CPI Increase: {food_cpi_increase_pct:.2f}%")
print(f"  Headline CPI Contribution (Food): {headline_cpi_food_contrib_pct:.2f}%")
print(f"  Headline CPI Contribution (Import Shock): {import_shock_headline_pct:.2f}%")
print(f"  Headline CPI Total Impact: {headline_cpi_total_pct:.2f}%")

print("\n" + "=" * 120)
print("PART: Financial Market Impacts (USD, UST, Crypto)")
print("=" * 120)

def clip(v, lo, hi):
    return max(lo, min(hi, v))

growth_proxy = delta_GDP_multiplier / 10.0
usd_change_pct_proxy = 0.45 * exchange_change_pct + 0.15 * headline_cpi_total_pct - 0.25 * growth_proxy
delta_10y_bps = 6.5 * headline_cpi_total_pct + 2.0 * growth_proxy + 0.8 * usd_change_pct_proxy
risk_appetite_index = 0.5 * growth_proxy + 0.000015 * net_employment - 0.5 * usd_change_pct_proxy - 0.02 * delta_10y_bps
equity_proxy_change_pct = clip(1.2 * risk_appetite_index, -8, 8)
crypto_change_pct = clip(2.5 * risk_appetite_index, -15, 15)

print(f"\nFinancial Proxies:")
print(f"  USD Index Change (proxy): {usd_change_pct_proxy:+.2f}%")
print(f"  10Y UST Yield Change: {delta_10y_bps:+.1f} bps")
print(f"  Risk Asset Proxy (Equity): {equity_proxy_change_pct:+.2f}%")
print(f"  Crypto Proxy (BTC/ETH basket): {crypto_change_pct:+.2f}%")


print("\n" + "=" * 120)
print("PART 13: Creating Beautiful Visualizations")
print("=" * 120)

colors = sns.color_palette("husl", 8)
color_positive = '#2ecc71'
color_negative = '#e74c3c'
color_neutral = '#3498db'

print("\nCreating Trade Flow Visualization...")
fig, ax = plt.subplots(figsize=(12, 6))
labels = ['Imports', 'Exports', 'Trade Deficit']
vals_2024 = [M_0, X_0, D_0]
vals_2025 = [M_1, X_1, D_1]
y = np.arange(len(labels))
left = np.minimum(vals_2024, vals_2025)
right = np.maximum(vals_2024, vals_2025)
ax.hlines(y, left, right, color='#95a5a6', linewidth=4, alpha=0.8)
ax.scatter(vals_2024, y, s=140, color='#34495e', edgecolor='white', linewidth=1.5, label='May 2024')
ax.scatter(vals_2025, y, s=140, color='#2ecc71', edgecolor='white', linewidth=1.5, label='May 2025')
for i, (v0, v1) in enumerate(zip(vals_2024, vals_2025)):
    diff = v1 - v0
    sign = '+' if diff >= 0 else ''
    ax.annotate(f'{sign}{diff:.1f}', xy=(right[i], i), xytext=(8, 0), textcoords='offset points',
                va='center', fontsize=10, color='#2c3e50')
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=12, fontweight='bold')
ax.set_xlabel('Billion USD', fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0.02, 0.98))
ax.margins(x=0.05)
beautify(ax)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'Figure_1_Trade_Flow_Changes.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Creating Price Impact Visualization...")
fig, ax = plt.subplots(figsize=(12, 7))
impact_categories = ['Supply Reduction', 'Import Price Increase', 'Extra Spending', 'Crowding-Out']
impact_values = [35, price_increase_pct, additional_spending, -consumption_reduction]
y_pos = np.arange(len(impact_categories))
colors_local = ['#e74c3c', '#e74c3c', '#e67e22', '#2ecc71']
ax.barh(y_pos, impact_values, color=colors_local, edgecolor='white', linewidth=1.5)
for i, v in enumerate(impact_values):
    label = f"{v:.1f}%" if i < 2 else f"${v:.2f}B"
    xpos = v if v > 0 else 0
    ha = 'left'
    ax.annotate(label, xy=(xpos, i), xytext=(8 if v > 0 else 8, 0), textcoords='offset points',
                va='center', ha=ha, fontsize=11, fontweight='bold', color='#2c3e50')
ax.set_yticks(y_pos)
ax.set_yticklabels(impact_categories, fontsize=11)
ax.set_xlabel('Magnitude', fontsize=12, fontweight='bold')
ax.set_xlim(min(0, min(impact_values) * 1.15), max(impact_values) * 1.10)
beautify(ax)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'Figure_2_Price_Impact.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Creating Strategic Goods Impact Visualization...")
fig, ax = plt.subplots(figsize=(12, 6))
goods = ['Rare Earth Elements', 'Lithium Batteries', 'EV Cost Increase']
price_increases = [RE_price_increase, LB_price_increase, EV_cost_increase * 100]
err_low = [price_increases[0] - RE_price_increase_low, price_increases[1] - LB_price_increase_low, price_increases[2] - (EV_cost_increase_low * 100)]
err_high = [RE_price_increase_high - price_increases[0], LB_price_increase_high - price_increases[1], (EV_cost_increase_high * 100) - price_increases[2]]
goods_colors = ['#9b59b6', '#16a085', '#d35400']
bars = ax.bar(goods, price_increases, color=goods_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.errorbar(goods, price_increases, yerr=[err_low, err_high], fmt='none', ecolor='black', elinewidth=1.5, capsize=6)
ax.set_ylabel('Price Increase (%)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., h, f'{h:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'Figure_3_Strategic_Goods_Impact.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Creating GDP Scenario Visualization...")
fig, ax = plt.subplots(figsize=(12, 6))
scenarios = ['Pessimistic', 'Baseline', 'Optimistic']
gdp_impacts = [pessimistic_GDP, delta_GDP_multiplier, optimistic_GDP]
y_pos = np.arange(len(scenarios))
ax.hlines(y_pos, 0, gdp_impacts, colors=['#e74c3c', '#3498db', '#2ecc71'], linewidth=4)
ax.scatter(gdp_impacts, y_pos, s=120, color=['#e74c3c', '#3498db', '#2ecc71'], edgecolors='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(scenarios, fontsize=11)
ax.set_xlabel('GDP Impact (Billion USD)', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=1)
ax.grid(True, alpha=0.3, axis='x', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'Figure_4_GDP_Scenarios.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Creating Employment Impact Visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

emp_categories = ['Job Loss\nfrom Export\nDecline', 'Job Creation\nPotential',
                  'Actual Job\nCreation\n(50% rate)', 'Net Employment\nChange']
emp_values = [-job_loss_export, job_creation_potential, actual_job_creation, net_employment]
emp_colors = [color_negative, color_positive, color_positive, color_neutral]

bars1 = ax1.bar(emp_categories, emp_values, color=emp_colors, alpha=0.9, edgecolor='white', linewidth=1.2)
ax1.set_ylabel('Number of Workers', fontsize=12, fontweight='bold')
ax1.axhline(y=0, color='#7f8c8d', linewidth=1)
beautify(ax1)
for bar, val in zip(bars1, emp_values):
    ax1.text(bar.get_x() + bar.get_width() / 2., val if val>0 else 0,
             f'{val:,.0f}',
             ha='center', va='bottom' if val > 0 else 'top', fontsize=10, fontweight='bold')

ax2.barh(['Employment'], [job_loss_export], color=color_negative, alpha=0.9, label='Job Loss', edgecolor='white',
         linewidth=1.2)
ax2.barh(['Employment'], [actual_job_creation], left=[job_loss_export], color=color_positive, alpha=0.9,
         label='Job Creation', edgecolor='white', linewidth=1.2)

ax2.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
ax2.legend(fontsize=11, loc='lower right')
beautify(ax2)

ax2.text(job_loss_export / 2, 0.02, f'Loss\n{job_loss_export:,.0f}',
         ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')
ax2.text(job_loss_export + actual_job_creation / 2, 0.02, f'Gain\n{actual_job_creation:,.0f}',
         ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'Figure_5_Employment_Impact.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Creating Manufacturing Reshoring Visualization...")
fig, ax = plt.subplots(figsize=(12, 6))
products_list = list(products.keys())
critical_tariffs = [(products[p]['labor_cost_ratio'] - 1 - logistics_ratio) / 1.0 for p in products_list]
current_tariffs = [products[p]['current_tariff'] for p in products_list]
ax.scatter([t * 100 for t in current_tariffs], [t * 100 for t in critical_tariffs], s=150, c=['#e74c3c','#3498db','#2ecc71','#9b59b6'], edgecolors='black')
for i, p in enumerate(products_list):
    ax.annotate(p, (current_tariffs[i] * 100, critical_tariffs[i] * 100), textcoords='offset points', xytext=(5,5), fontsize=10)
ax.plot([0, max([t*100 for t in current_tariffs])], [0, max([t*100 for t in critical_tariffs])], linestyle='--', color='gray')
ax.set_xlabel('Current Tariff Rate (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Critical Tariff Threshold (%)', fontsize=12, fontweight='bold')
xmax = max([t*100 for t in current_tariffs])
ymax = max([t*100 for t in critical_tariffs])
ax.fill_between([0, xmax], 0, [0, ymax], color='#f9ebea', alpha=0.25)
ax.fill_between([0, xmax], [0, ymax], [ymax, ymax], color='#e8f8f5', alpha=0.25)
beautify(ax)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'Figure_6_Manufacturing_Reshoring.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Creating Comprehensive Impact Dashboard...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

impacts = ['Trade\nDeficit\nImprovement', 'GDP\nImpact', 'Import\nPrice\nIncrease',
           'Rare Earth\nPrice\nIncrease', 'Net\nEmployment\nChange', 'Reshoring\nPotential']
impact_values = [abs(delta_D), delta_GDP_multiplier, price_increase_pct, RE_price_increase, net_employment / 1000,
                 reshoring_index]
impact_units = ['$B', '$B', '%', '%', 'K', '/100']
impact_colors = [color_positive, color_neutral, color_negative, color_negative, color_neutral, color_negative]

ax_main = fig.add_subplot(gs[0:2, :])
order = np.argsort(impact_values)
impacts_sorted = [impacts[i] for i in order]
values_sorted = [impact_values[i] for i in order]
colors_sorted = [impact_colors[i] for i in order]
bars = ax_main.barh(impacts_sorted, values_sorted, color=colors_sorted, alpha=0.85, edgecolor='white', linewidth=1.2)
ax_main.set_xlabel('Magnitude of Impact', fontsize=12, fontweight='bold')
beautify(ax_main)

for bar, val, unit in zip(bars, impact_values, impact_units):
    width = bar.get_width()
    ax_main.text(width + max(impact_values) * 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}{unit}',
                 ha='left', va='center', fontsize=10, fontweight='bold')

ax_metrics = fig.add_subplot(gs[2, 0])
ax_metrics.axis('off')

import_decline_pct = (M_0 - M_1) / M_0 * 100
export_decline_pct = (X_0 - X_1) / X_0 * 100
deficit_reduction = abs(delta_D)
metrics_text = f"KEY METRICS\n\nTrade Impact:\n  Import Decline: -{import_decline_pct:.1f}%\n  Export Decline: -{export_decline_pct:.1f}%\n  Deficit Reduction: -${deficit_reduction:.2f}B\n\nConsumption Pressure:\n  Price Increase: {price_increase_pct:.2f}%\n  Extra Spending: ${additional_spending:.2f}B\n  Crowding-Out: ${consumption_reduction:.2f}B\n"

ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, pad=1))

ax_assess = fig.add_subplot(gs[2, 1])
ax_assess.axis('off')

assess_text = f"POLICY ASSESSMENT\n\nGDP Impact: +${delta_GDP_multiplier:.2f}B\n  (Neutral-Positive)\n\nEmployment: +{max(0, net_employment):,.0f} jobs\n  (Depends on substitution)\n\nReshoring Index: {reshoring_index:.1f}/100\n  (Limited potential)\n\nConclusion:\nEffective for reducing trade deficit\nbut limited manufacturing return.\n"

color_box = '#d5f4e6' if reshoring_index < 50 else '#a9dfbf'
ax_assess.text(0.05, 0.95, assess_text, transform=ax_assess.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor=color_box, alpha=0.8, pad=1))

plt.savefig(os.path.join(FIG_DIR, 'Figure_7_Comprehensive_Dashboard.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Creating Sensitivity Analysis Visualization...")
fig, ax = plt.subplots(figsize=(12, 7))

parameters = ['Price\nSensitivity\n(eta)', 'Marginal\nConsumption\n(MPC)', 'Rare Earth\nPrice\nSensitivity',
              'Domestic\nSubstitution\nRate', 'Trade\nMultiplier', 'Consumption\nMultiplier']
baseline_vals = [eta, MPC, gamma_RE, domestic_substitution_rate, trade_multiplier, consumption_multiplier]
ranges_low = [0.2, 0.70, 1.0, 0.3, 1.2, 1.5]
ranges_high = [0.6, 0.85, 3.0, 0.7, 1.8, 2.5]

x_pos = np.arange(len(parameters))

for i, (param, baseline, low, high) in enumerate(zip(parameters, baseline_vals, ranges_low, ranges_high)):
    ax.barh(i, high - low, left=low, height=0.5, color='#ecf0f1',
            edgecolor='black', linewidth=2, alpha=0.6)
    ax.plot(baseline, i, 'o', markersize=12, color=color_positive,
            markeredgecolor='black', markeredgewidth=2, zorder=5)
    ax.text(baseline - 0.05, i - 0.35, f'Base: {baseline:.2f}',
            ha='right', va='top', fontsize=9, fontweight='bold')

ax.set_yticks(x_pos)
ax.set_yticklabels(parameters, fontsize=11)
ax.set_xlabel('Parameter Value Range', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x', linestyle='--')

from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='#ecf0f1', edgecolor='black', linewidth=2, label='Sensitivity Range'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_positive,
               markersize=10, markeredgecolor='black', markeredgewidth=1.5, label='Baseline Value')
]
ax.legend(handles=legend_elements, fontsize=11, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'Figure_8_Sensitivity_Analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Creating Food CPI Decomposition Visualization...")
fig, ax = plt.subplots(figsize=(12, 6))
labels_food = ['Food CPI Increase', 'Headline CPI (Food)', 'Headline CPI (Import Shock)']
values_food = [food_cpi_increase_pct, headline_cpi_food_contrib_pct, import_shock_headline_pct]
colors_food = ['#e67e22', '#3498db', '#e74c3c']
ax.barh(labels_food, values_food, color=colors_food, edgecolor='white', linewidth=1.2)
for i, v in enumerate(values_food):
    ax.text(v + max(values_food) * 0.02, i, f'{v:.2f}%', va='center', ha='left', fontsize=11)
ax.set_xlabel('Percent', fontsize=12, fontweight='bold')
beautify(ax)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'Figure_9_Food_CPI_Decomposition.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Creating Financial Market Impacts Visualization...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].bar(['USD Index'], [usd_change_pct_proxy], color='#34495e', edgecolor='white', linewidth=1.2)
axes[0].set_ylim(min(-5, usd_change_pct_proxy - 3), max(5, usd_change_pct_proxy + 3))
beautify(axes[0])
axes[1].bar(['10Y UST'], [delta_10y_bps], color='#8e44ad', edgecolor='white', linewidth=1.2)
axes[1].set_ylim(min(-50, delta_10y_bps - 20), max(50, delta_10y_bps + 20))
beautify(axes[1])
axes[2].bar(['Crypto'], [crypto_change_pct], color='#27ae60', edgecolor='white', linewidth=1.2)
axes[2].set_ylim(min(-20, crypto_change_pct - 8), max(20, crypto_change_pct + 8))
beautify(axes[2])
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'Figure_10_Financial_Market_Impacts.png'), dpi=300, bbox_inches='tight')
plt.close()


 

results_summary.to_csv(os.path.join(CHART_DIR, 'Q5_Comprehensive_Impact_Summary.csv'), index=False, encoding='utf-8-sig')

scenarios_df = pd.DataFrame({
    'Scenario': ['Pessimistic', 'Baseline', 'Optimistic'],
    'GDP Impact ($B)': [round(pessimistic_GDP, 2), round(delta_GDP_multiplier, 2), round(optimistic_GDP, 2)],
    'Description': [
        'Low import substitution, sustained consumption pressure',
        'Normal market adjustment process',
        'High import substitution, rapid consumption recovery'
    ]
})
scenarios_df.to_csv(os.path.join(CHART_DIR, 'Q5_Scenario_Analysis.csv'), index=False, encoding='utf-8-sig')

sensitivity_df = pd.DataFrame({
    'Parameter': ['Price Sensitivity (eta)', 'Marginal Consumption (MPC)', 'Rare Earth Price Sensitivity',
                  'Domestic Substitution Rate', 'Trade Multiplier', 'Consumption Multiplier'],
    'Baseline Value': [eta, MPC, gamma_RE, domestic_substitution_rate, trade_multiplier, consumption_multiplier],
    'Sensitivity Range': ['0.2-0.6', '0.70-0.85', '1.0-3.0', '0.3-0.7', '1.2-1.8', '1.5-2.5'],
    'Impact Field': ['GDP and Employment', 'Consumption Decline Magnitude', 'Cost Increase Magnitude',
                     'Job Creation', 'GDP Multiplier Effect', 'GDP Multiplier Effect']
})
sensitivity_df.to_csv(os.path.join(CHART_DIR, 'Q5_Sensitivity_Analysis.csv'), index=False, encoding='utf-8-sig')

detailed_results = {
    'Trade Impact': {
        'Import Change ($B)': round(delta_M, 2),
        'Export Change ($B)': round(delta_X, 2),
        'Import Change (%)': round((delta_M / M_0) * 100, 1),
        'Export Change (%)': round((delta_X / X_0) * 100, 1),
        'Trade Deficit Improvement ($B)': round(abs(delta_D), 2),
    },
    'Price and Consumption': {
        'Import Price Increase (%)': round(price_increase_pct, 2),
        'Additional Spending ($B)': round(additional_spending, 2),
        'Consumption Reduction ($B)': round(consumption_reduction, 2),
    },
    'Strategic Goods': {
        'Rare Earth Price Increase (%)': round(RE_price_increase, 1),
        'Lithium Battery Price Increase (%)': round(LB_price_increase, 1),
        'EV Cost Increase (%)': round(EV_cost_increase * 100, 2),
    },
    'Macroeconomic': {
        'Net Export Change ($B)': round(net_export_change, 2),
        'GDP Direct Impact ($B)': round(delta_GDP_direct, 2),
        'GDP Multiplier Impact ($B)': round(delta_GDP_multiplier, 2),
        'Net Employment Change': round(net_employment, 0),
    }
}

detailed_df = pd.DataFrame([
    (category, k, v) for category, params in detailed_results.items()
    for k, v in params.items()
], columns=['Category', 'Parameter', 'Value'])

detailed_df.to_csv(os.path.join(CHART_DIR, 'Q5_Detailed_Results.csv'), index=False, encoding='utf-8-sig')

financial_impacts_df = pd.DataFrame({
    'Metric': [
        'Food CPI Increase (%)',
        'Headline CPI Contribution (Food) (%)',
        'Headline CPI Contribution (Import Shock) (%)',
        'Headline CPI Total Impact (%)',
        'USD Index Change (proxy) (%)',
        '10Y UST Yield Change (bps)',
        'Risk Asset Proxy Change (%)',
        'Crypto Proxy Change (%)'
    ],
    'Value': [
        round(food_cpi_increase_pct, 2),
        round(headline_cpi_food_contrib_pct, 2),
        round(import_shock_headline_pct, 2),
        round(headline_cpi_total_pct, 2),
        round(usd_change_pct_proxy, 2),
        round(delta_10y_bps, 1),
        round(equity_proxy_change_pct, 2),
        round(crypto_change_pct, 2)
    ]
})
financial_impacts_df.to_csv(os.path.join(CHART_DIR, 'Q5_Financial_Impacts.csv'), index=False, encoding='utf-8-sig')

growth_proxy_pess = pessimistic_GDP / 10.0
growth_proxy_base = delta_GDP_multiplier / 10.0
growth_proxy_opt = optimistic_GDP / 10.0

def compute_financial(growth):
    u = 0.45 * exchange_change_pct + 0.15 * headline_cpi_total_pct - 0.25 * growth
    y = 6.5 * headline_cpi_total_pct + 2.0 * growth + 0.8 * u
    r = 0.5 * growth + 0.000015 * net_employment - 0.5 * u - 0.02 * y
    eq = max(-8, min(8, 1.2 * r))
    cr = max(-15, min(15, 2.5 * r))
    return u, y, eq, cr

u_p, y_p, eq_p, cr_p = compute_financial(growth_proxy_pess)
u_b, y_b, eq_b, cr_b = compute_financial(growth_proxy_base)
u_o, y_o, eq_o, cr_o = compute_financial(growth_proxy_opt)

financial_scenarios_df = pd.DataFrame({
    'Scenario': ['Pessimistic', 'Baseline', 'Optimistic'],
    'USD Index Change (proxy) (%)': [round(u_p, 2), round(u_b, 2), round(u_o, 2)],
    '10Y UST Yield Change (bps)': [round(y_p, 1), round(y_b, 1), round(y_o, 1)],
    'Risk Asset Proxy Change (%)': [round(eq_p, 2), round(eq_b, 2), round(eq_o, 2)],
    'Crypto Proxy Change (%)': [round(cr_p, 2), round(cr_b, 2), round(cr_o, 2)]
})
financial_scenarios_df.to_csv(os.path.join(CHART_DIR, 'Q5_Financial_Scenarios.csv'), index=False, encoding='utf-8-sig')


 
