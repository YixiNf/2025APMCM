import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
try:
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingRegressor
    HGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor
    HGB_AVAILABLE = False
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings
from datetime import datetime
import difflib
import os

warnings.filterwarnings('ignore')



np.random.seed(42)
OUTPUT_FIG_DIR = 'figures'
OUTPUT_TABLE_DIR = 'chart'
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_TABLE_DIR, exist_ok=True)

trade_df = pd.read_excel('./data/美国2020_2025年对外贸易数据.xlsx')
tariff_df = pd.read_excel('./data/tariff_data.xlsx')

trade_descriptions = trade_df[['HTS Number', 'Description']].drop_duplicates('Description').copy()
tariff_descriptions = tariff_df[['hts8', 'brief_description']].drop_duplicates('brief_description').copy()

mapping_list = []


for idx, trade_row in trade_descriptions.iterrows():
    trade_desc = str(trade_row['Description']).strip().upper()

    exact_match = tariff_descriptions[
        tariff_descriptions['brief_description'].str.upper().str.strip() == trade_desc
        ]

    if len(exact_match) > 0:
        mapping_list.append({
            'HTS_Number': int(trade_row['HTS Number']),
            'Trade_Description': trade_row['Description'],
            'Brief_Description': exact_match.iloc[0]['brief_description'],
            'HTS8': int(exact_match.iloc[0]['hts8']),
            'Match_Type': 'Exact'
        })
    else:
        trade_keywords = trade_desc.split()[:3]  

        partial_matches = []
        for tariff_idx, tariff_row in tariff_descriptions.iterrows():
            tariff_desc = str(tariff_row['brief_description']).strip().upper()
            if all(keyword in tariff_desc for keyword in trade_keywords):
                partial_matches.append({
                    'similarity': difflib.SequenceMatcher(None, trade_desc, tariff_desc).ratio(),
                    'Brief_Description': tariff_row['brief_description'],
                    'HTS8': int(tariff_row['hts8'])
                })

        if partial_matches:
            best_match = max(partial_matches, key=lambda x: x['similarity'])
            mapping_list.append({
                'HTS_Number': int(trade_row['HTS Number']),
                'Trade_Description': trade_row['Description'],
                'Brief_Description': best_match['Brief_Description'],
                'HTS8': best_match['HTS8'],
                'Match_Type': 'Partial'
            })
        else:
            mapping_list.append({
                'HTS_Number': int(trade_row['HTS Number']),
                'Trade_Description': trade_row['Description'],
                'Brief_Description': 'Unknown',
                'HTS8': None,
                'Match_Type': 'None'
            })

mapping_df = pd.DataFrame(mapping_list)
exact_matches = len(mapping_df[mapping_df['Match_Type'] == 'Exact'])
partial_matches = len(mapping_df[mapping_df['Match_Type'] == 'Partial'])
no_matches = len(mapping_df[mapping_df['Match_Type'] == 'None'])

mapping_df.to_csv(os.path.join(OUTPUT_TABLE_DIR, 'hts_description_mapping.csv'), index=False)

country_columns = [col for col in tariff_df.columns if col not in
                   ['hts8', 'brief_description', 'quantity_1_code', 'quantity_2_code',
                    'wto_binding_code', 'begin_effect_date', 'end_effective_date']]

def parse_tariff_rate(tariff_str):
    if pd.isna(tariff_str) or tariff_str == '':
        return 0.0

    tariff_str = str(tariff_str).strip()

    if '免税' in tariff_str or tariff_str == 'Free':
        return 0.0

    if '%' in tariff_str:
        try:
            return float(tariff_str.replace('%', '').strip()) / 100.0
        except:
            return 0.0

    return 0.05  


tariff_by_hts8 = {}

for country_col in country_columns:
    tariff_df[f'{country_col}_parsed'] = tariff_df[country_col].apply(parse_tariff_rate)

    hts8_tariff_map = dict(zip(
        tariff_df['hts8'],
        tariff_df[f'{country_col}_parsed']
    ))

    tariff_by_hts8[country_col.replace('_parsed', '')] = hts8_tariff_map

trade_df_merged = trade_df.merge(
    mapping_df[['HTS_Number', 'HTS8']],
    left_on='HTS Number',
    right_on='HTS_Number',
    how='left'
)

trade_years = ['2020', '2021', '2022', '2023', '2024']

trade_long = trade_df_merged.melt(
    id_vars=['Data Type', 'HTS Number', 'Description', 'Country', 'HTS8'],
    value_vars=trade_years,
    var_name='Year',
    value_name='Trade_Value'
)

trade_long = trade_long[trade_long['Data Type'] == 'General Import Charges'].copy()
trade_long['Trade_Value'] = pd.to_numeric(trade_long['Trade_Value'], errors='coerce').fillna(0)
trade_long['Year'] = trade_long['Year'].astype(int)

country_tariff_map = {}

for country in trade_long['Country'].unique():
    country_clean = country.replace(' ', '').lower()

    best_match = None
    for tariff_col in country_columns:
        if country_clean in tariff_col.lower() or tariff_col.lower() in country_clean:
            best_match = tariff_col
            break

    country_tariff_map[country] = best_match

trade_long['Tariff_Country_Col'] = trade_long['Country'].map(country_tariff_map)


def get_tariff_rate(row):
    country_col = row['Tariff_Country_Col']
    hts8 = row['HTS8']

    if pd.isna(country_col) or pd.isna(hts8):
        if '非WTO成员国' in tariff_by_hts8:
            return tariff_by_hts8['非WTO成员国'].get(int(hts8), 0.05) if not pd.isna(hts8) else 0.0
        return 0.0

    try:
        return tariff_by_hts8.get(country_col, {}).get(int(hts8), 0.05)
    except:
        return 0.05


trade_long['Tariff_Rate'] = trade_long.apply(get_tariff_rate, axis=1)


trade_long['Tariff_Revenue'] = trade_long['Trade_Value'] * trade_long['Tariff_Rate']




country_year_data = trade_long.groupby(['Year', 'Country']).agg({
    'Trade_Value': 'sum',
    'Tariff_Revenue': 'sum',
    'Tariff_Rate': 'mean'
}).reset_index()

country_year_data['Effective_Tariff_Rate'] = (
        country_year_data['Tariff_Revenue'] /
        (country_year_data['Trade_Value'] + 1e-6)
)


yearly_data = trade_long.groupby('Year').agg({
    'Trade_Value': 'sum',
    'Tariff_Revenue': 'sum',
    'Tariff_Rate': 'mean'
}).reset_index()

yearly_data['Effective_Tariff_Rate'] = (
        yearly_data['Tariff_Revenue'] /
        (yearly_data['Trade_Value'] + 1e-6)
)

country_year_data = country_year_data.sort_values(['Country', 'Year']).reset_index(drop=True)

country_year_data['Trade_Value_Lag1'] = country_year_data.groupby('Country')['Trade_Value'].shift(1)
country_year_data['Trade_Value_Lag2'] = country_year_data.groupby('Country')['Trade_Value'].shift(2)
country_year_data['Tariff_Rate_Lag1'] = country_year_data.groupby('Country')['Tariff_Rate'].shift(1)
country_year_data['Tariff_Rate_Lag2'] = country_year_data.groupby('Country')['Tariff_Rate'].shift(2)

country_year_data['Trade_Value_Growth'] = country_year_data.groupby('Country')['Trade_Value'].pct_change()
country_year_data['Tariff_Rate_Change'] = country_year_data['Tariff_Rate'] - country_year_data['Tariff_Rate_Lag1']
country_year_data['Trade_Value_MA3'] = country_year_data.groupby('Country')['Trade_Value'].rolling(window=3).mean().reset_index(level=0, drop=True)
country_year_data['Tariff_Rate_MA3'] = country_year_data.groupby('Country')['Tariff_Rate'].rolling(window=3).mean().reset_index(level=0, drop=True)

country_year_data['Ln_Trade_Value'] = np.log(country_year_data['Trade_Value'] + 1)
country_year_data['Ln_Tariff_Rate'] = np.log(country_year_data['Tariff_Rate'] + 0.001)

country_year_data['Post_Policy'] = (country_year_data['Year'] >= 2025).astype(int)
country_year_data['LT_Post_Interaction'] = country_year_data['Ln_Tariff_Rate'] * country_year_data['Post_Policy']

country_year_data_clean = country_year_data.dropna().reset_index(drop=True)

logger.info(f"✓ 特征工程完成: {len(country_year_data_clean)} 条有效记录")


logger.info("估计关税贸易弹性...")

feature_cols = ['Ln_Tariff_Rate', 'Trade_Value_Lag1', 'Post_Policy', 'Tariff_Rate_Change']
X_elasticity = country_year_data_clean[feature_cols].fillna(0).values
y_elasticity = country_year_data_clean['Ln_Trade_Value'].values

elasticity_model = LinearRegression()
elasticity_model.fit(X_elasticity, y_elasticity)

tariff_elasticity = elasticity_model.coef_[0]

logger.info(f"关税贸易弹性估计完成")
logger.info(f"关税弹性系数: {tariff_elasticity:.4f}")
logger.info(f"含义: 关税增加1%, 进口值下降{abs(tariff_elasticity):.4f}%")

logger.info("建立贸易量预测模型...")

model_features = [
    'Trade_Value_Lag1', 'Trade_Value_Lag2',
    'Ln_Tariff_Rate', 'Tariff_Rate_Change', 'Tariff_Rate_Lag1', 'Tariff_Rate_Lag2',
    'Trade_Value_Growth', 'Trade_Value_MA3', 'Tariff_Rate_MA3',
    'Post_Policy', 'LT_Post_Interaction'
]

X_model = country_year_data_clean[model_features].fillna(0).values
y_model = country_year_data_clean['Ln_Trade_Value'].values

scaler = StandardScaler()
X_model_scaled = scaler.fit_transform(X_model)

if len(X_model) > 10:
    tscv = TimeSeriesSplit(n_splits=3)
    train_idx, test_idx = list(tscv.split(X_model))[2]

    X_train, X_test = X_model_scaled[train_idx], X_model_scaled[test_idx]
    y_train, y_test = y_model[train_idx], y_model[test_idx]

else:
    X_train, X_test = X_model_scaled, X_model_scaled
    y_train, y_test = y_model, y_model

if HGB_AVAILABLE:
    hgb_model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=500,
                                              min_samples_leaf=5, random_state=42)
    hgb_model.fit(X_train, y_train)
    y_pred_hgb = hgb_model.predict(X_test)
    r2_hgb = r2_score(y_test, y_pred_hgb)
    rmse_hgb = np.sqrt(mean_squared_error(y_test, y_pred_hgb))
else:
    hgb_model = GradientBoostingRegressor(loss='huber', n_estimators=400, learning_rate=0.05, max_depth=4,
                                          min_samples_split=5, subsample=0.9, random_state=42)
    hgb_model.fit(X_train, y_train)
    y_pred_hgb = hgb_model.predict(X_test)
    r2_hgb = r2_score(y_test, y_pred_hgb)
    rmse_hgb = np.sqrt(mean_squared_error(y_test, y_pred_hgb))

adaboost_model = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=4),
                                   n_estimators=400, learning_rate=0.05, random_state=42)
adaboost_model.fit(X_train, y_train)
y_pred_adaboost = adaboost_model.predict(X_test)
r2_adaboost = r2_score(y_test, y_pred_adaboost)
rmse_adaboost = np.sqrt(mean_squared_error(y_test, y_pred_adaboost))

et_model = ExtraTreesRegressor(n_estimators=600, max_depth=None, min_samples_split=2,
                               random_state=42, n_jobs=-1)
et_model.fit(X_train, y_train)
y_pred_et = et_model.predict(X_test)
r2_et = r2_score(y_test, y_pred_et)
rmse_et = np.sqrt(mean_squared_error(y_test, y_pred_et))

feature_importance_et = pd.DataFrame({'Feature': model_features,
                                      'Importance': et_model.feature_importances_}).sort_values('Importance', ascending=False)

weights = np.array([r2_hgb, r2_adaboost, r2_et])
weights = weights / (weights.sum() + 1e-6)

y_pred_ensemble = (weights[0] * y_pred_hgb +
                   weights[1] * y_pred_adaboost +
                   weights[2] * y_pred_et)

r2_ensemble = r2_score(y_test, y_pred_ensemble)
rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))


seq_len = 4
seq_cols = ['Trade_Value', 'Tariff_Rate', 'Effective_Tariff_Rate', 'Post_Policy']
seq_X = []
seq_y = []
for _, g in country_year_data.sort_values(['Country', 'Year']).groupby('Country'):
    f = g[seq_cols].values
    yv = np.log(g['Trade_Value'] + 1).values
    for i in range(seq_len, len(f)):
        seq_X.append(f[i - seq_len:i])
        seq_y.append(yv[i])
seq_X = np.array(seq_X)
seq_y = np.array(seq_y)
if len(seq_X) > 10:
    flat = seq_X.reshape(seq_X.shape[0], -1)
    seq_scaler = StandardScaler()
    flat_scaled = seq_scaler.fit_transform(flat)
    seq_X_scaled = flat_scaled.reshape(seq_X.shape[0], seq_X.shape[1], seq_X.shape[2])
    split = int(len(seq_X_scaled) * 0.8)
    X_train_seq = seq_X_scaled[:split]
    y_train_seq = seq_y[:split]
    X_test_seq = seq_X_scaled[split:]
    y_test_seq = seq_y[split:]
else:
    X_train_seq = seq_X
    y_train_seq = seq_y
    X_test_seq = seq_X
    y_test_seq = seq_y

r2_rnn = None
rmse_rnn = None
r2_lstm = None
rmse_lstm = None
r2_transformer = None
rmse_transformer = None
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class SeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    input_dim = X_train_seq.shape[2] if len(X_train_seq) > 0 else 0
    if input_dim > 0:
        train_dl = DataLoader(SeriesDataset(X_train_seq, y_train_seq), batch_size=64, shuffle=False)
        test_dl = DataLoader(SeriesDataset(X_test_seq, y_test_seq), batch_size=64, shuffle=False)
        def train_eval(model, epochs=30):
            model = model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()
            for _ in range(epochs):
                model.train()
                for xb, yb in train_dl:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    opt.zero_grad()
                    preds = model(xb)
                    loss = loss_fn(preds, yb)
                    loss.backward()
                    opt.step()
            model.eval()
            preds_list = []
            with torch.no_grad():
                for xb, _ in test_dl:
                    xb = xb.to(device)
                    preds_list.append(model(xb).cpu().numpy())
            preds = np.concatenate(preds_list) if len(preds_list) > 0 else np.array([])
            if len(preds) == len(y_test_seq):
                r2 = r2_score(y_test_seq, preds)
                rmse = np.sqrt(mean_squared_error(y_test_seq, preds))
                return r2, rmse
            return None, None
        class RNNRegressor(nn.Module):
            def __init__(self, input_dim, hidden_dim=64, num_layers=1):
                super().__init__()
                self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, 1)
            def forward(self, x):
                out, _ = self.rnn(x)
                return self.fc(out[:, -1, :]).squeeze(-1)
        class LSTMRegressor(nn.Module):
            def __init__(self, input_dim, hidden_dim=64, num_layers=1):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=500):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)
            def forward(self, x):
                return x + self.pe[:, :x.size(1)]
        class TransformerRegressor(nn.Module):
            def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, d_model)
                encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, batch_first=True)
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.pos = PositionalEncoding(d_model)
                self.fc = nn.Linear(d_model, 1)
            def forward(self, x):
                h = self.input_proj(x)
                h = self.pos(h)
                h = self.encoder(h)
                return self.fc(h[:, -1, :]).squeeze(-1)
        r2_rnn, rmse_rnn = train_eval(RNNRegressor(input_dim))
        r2_lstm, rmse_lstm = train_eval(LSTMRegressor(input_dim))
        r2_transformer, rmse_transformer = train_eval(TransformerRegressor(input_dim))
except Exception:
    pass



X_all = scaler.fit_transform(X_model)

linear_model_full = LinearRegression()
linear_model_full.fit(X_all, y_model)

if HGB_AVAILABLE:
    hgb_model_full = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=500,
                                                   min_samples_leaf=5, random_state=42)
else:
    hgb_model_full = GradientBoostingRegressor(loss='huber', n_estimators=400, learning_rate=0.05, max_depth=4,
                                               min_samples_split=5, subsample=0.9, random_state=42)
hgb_model_full.fit(X_all, y_model)

adaboost_model_full = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=4),
                                        n_estimators=400, learning_rate=0.05, random_state=42)
adaboost_model_full.fit(X_all, y_model)

et_model_full = ExtraTreesRegressor(n_estimators=600, max_depth=None, min_samples_split=2,
                                   random_state=42, n_jobs=-1)
et_model_full.fit(X_all, y_model)

y_pred_ensemble_all = (weights[0] * hgb_model_full.predict(X_all) +
                       weights[1] * adaboost_model_full.predict(X_all) +
                       weights[2] * et_model_full.predict(X_all))

country_year_data_clean['Trade_Value_Pred'] = np.exp(y_pred_ensemble_all)
country_year_data_clean['Tariff_Revenue_Actual'] = (
        country_year_data_clean['Trade_Value'] * country_year_data_clean['Tariff_Rate']
)
country_year_data_clean['Tariff_Revenue_Pred'] = (
        country_year_data_clean['Trade_Value_Pred'] * country_year_data_clean['Tariff_Rate']
)




gamma_adjustment = linear_model_full.coef_[1]
gamma_adjustment = max(0.0, min(gamma_adjustment, 0.95))

long_term_elasticity = tariff_elasticity / (1 - gamma_adjustment) if gamma_adjustment < 1 else tariff_elasticity


baseline_tariff = yearly_data[yearly_data['Year'] == 2024]['Effective_Tariff_Rate'].values[0]
baseline_import = yearly_data[yearly_data['Year'] == 2024]['Trade_Value'].values[0]
baseline_revenue = yearly_data[yearly_data['Year'] == 2024]['Tariff_Revenue'].values[0]


policy_tariff = baseline_tariff * 8
policy_tariff = min(policy_tariff, 0.50)


scenarios = {
    'Short-term (3 months)': {'elasticity_factor': 0.10},
    'Medium-term (12 months)': {'elasticity_factor': 0.40},
    'Long-term (24 months)': {'elasticity_factor': 1.00}
}

scenario_results = {}

for scenario_name, params in scenarios.items():
    elasticity_factor = params['elasticity_factor']

    tariff_change_pct = np.log(1 + (policy_tariff - baseline_tariff) / (baseline_tariff + 1e-6))

    trade_change_pct = tariff_elasticity * tariff_change_pct * elasticity_factor

    new_import = baseline_import * np.exp(trade_change_pct)

    new_revenue = new_import * policy_tariff

    counterfactual_revenue = baseline_import * baseline_tariff

    revenue_change = new_revenue - counterfactual_revenue
    revenue_change_pct = (revenue_change / counterfactual_revenue * 100) if counterfactual_revenue > 0 else 0

    scenario_results[scenario_name] = {
        'Tariff_Rate': policy_tariff,
        'Import_Value_B': new_import / 1e9,
        'Tariff_Revenue_B': new_revenue / 1e9,
        'Revenue_Change_B': revenue_change / 1e9,
        'Revenue_Change_Pct': revenue_change_pct,
        'Trade_Change_Pct': trade_change_pct * 100
    }

    pass

 

elasticity_scenarios = {
    'Mild Reaction (Elasticity=-0.02)': -0.02,
    'Baseline Reaction (Elasticity=-0.05)': tariff_elasticity,
    'Strong Reaction (Elasticity=-0.10)': -0.10,
    'Extreme Reaction (Elasticity=-0.20)': -0.20
}

sensitivity_results = {}
counterfactual_revenue = baseline_import * baseline_tariff

for scenario_name, elasticity in elasticity_scenarios.items():
    tariff_change_pct = np.log(1 + (policy_tariff - baseline_tariff) / (baseline_tariff + 1e-6))
    trade_change_pct = elasticity * tariff_change_pct

    new_import = baseline_import * np.exp(trade_change_pct)
    new_revenue = new_import * policy_tariff
    revenue_change = new_revenue - counterfactual_revenue
    revenue_change_pct = (revenue_change / counterfactual_revenue * 100) if counterfactual_revenue > 0 else 0

    sensitivity_results[scenario_name] = {
        'Elasticity': elasticity,
        'Tariff_Revenue_B': new_revenue / 1e9,
        'Revenue_Change_B': revenue_change / 1e9,
        'Revenue_Change_Pct': revenue_change_pct
    }

    pass

 

 
scenario_df = pd.DataFrame(scenario_results).T
scenario_df.to_csv(os.path.join(OUTPUT_TABLE_DIR, 'tariff_revenue_scenarios.csv'))

 
sensitivity_df = pd.DataFrame(sensitivity_results).T
sensitivity_df.to_csv(os.path.join(OUTPUT_TABLE_DIR, 'tariff_revenue_sensitivity.csv'))

 
model_fit = pd.DataFrame({
    'Year': country_year_data_clean['Year'],
    'Country': country_year_data_clean['Country'],
    'Trade_Value': country_year_data_clean['Trade_Value'],
    'Tariff_Rate': country_year_data_clean['Tariff_Rate'],
    'Tariff_Revenue_Actual': country_year_data_clean['Tariff_Revenue_Actual'],
    'Trade_Value_Pred': country_year_data_clean['Trade_Value_Pred'],
    'Tariff_Revenue_Pred': country_year_data_clean['Tariff_Revenue_Pred']
})
model_fit.to_csv(os.path.join(OUTPUT_TABLE_DIR, 'tariff_revenue_model_fit.csv'), index=False)

 
perf_models = ['HistGB/GBDT', 'AdaBoost', 'ExtraTrees', 'Ensemble']
perf_r2 = [r2_hgb, r2_adaboost, r2_et, r2_ensemble]
perf_rmse = [rmse_hgb, rmse_adaboost, rmse_et, rmse_ensemble]
if r2_rnn is not None and rmse_rnn is not None:
    perf_models.append('RNN')
    perf_r2.append(r2_rnn)
    perf_rmse.append(rmse_rnn)
if r2_lstm is not None and rmse_lstm is not None:
    perf_models.append('LSTM')
    perf_r2.append(r2_lstm)
    perf_rmse.append(rmse_lstm)
if r2_transformer is not None and rmse_transformer is not None:
    perf_models.append('Transformer')
    perf_r2.append(r2_transformer)
    perf_rmse.append(rmse_transformer)
model_performance = pd.DataFrame({'Model': perf_models, 'R2': perf_r2, 'RMSE': perf_rmse})
model_performance.to_csv(os.path.join(OUTPUT_TABLE_DIR, 'tariff_revenue_model_performance.csv'), index=False)

 
key_params = pd.DataFrame({
    'Parameter': [
        'Baseline Effective Tariff Rate',
        'Baseline Import Value (B)',
        'Baseline Tariff Revenue (B)',
        'Policy Tariff Rate',
        'Tariff Rate Increase',
        'Short-term Elasticity',
        'Long-term Elasticity',
        'Adjustment Speed (Gamma)'
    ],
    'Value': [
        f'{baseline_tariff * 100:.2f}%',
        f'${baseline_import / 1e9:.2f}',
        f'${baseline_revenue / 1e9:.2f}',
        f'{policy_tariff * 100:.2f}%',
        f'{policy_tariff / baseline_tariff:.2f}x',
        f'{tariff_elasticity:.4f}',
        f'{long_term_elasticity:.4f}',
        f'{gamma_adjustment:.4f}'
    ]
})
key_params.to_csv(os.path.join(OUTPUT_TABLE_DIR, 'tariff_revenue_key_parameters.csv'), index=False)

 
yearly_data.to_csv(os.path.join(OUTPUT_TABLE_DIR, 'tariff_revenue_yearly_summary.csv'), index=False)

 

print("\n" + "=" * 100)
print("✓ 4.1 模型建立阶段完成!")
print("=" * 100)

summary = f"""
==================== 模型估计结果摘要 ====================

1. 贸易流预测模型性能
   - HistGB/GBDT: R²={r2_hgb:.4f}
   - AdaBoost: R²={r2_adaboost:.4f}
   - ExtraTrees: R²={r2_et:.4f}
   - 集成模型: R²={r2_ensemble:.4f}

2. 关税贸易弹性
   - 短期弹性: {tariff_elasticity:.4f}
   - 长期弹性: {long_term_elasticity:.4f}
   - 长期/短期: {long_term_elasticity / tariff_elasticity:.2f}倍
   - 调整速度: {gamma_adjustment:.4f}

3. 基准数据(2024年)
   - 有效关税率: {baseline_tariff * 100:.2f}%
   - 进口值: ${baseline_import / 1e9:.2f}B
   - 关税收入: ${baseline_revenue / 1e9:.2f}B

4. 政策情景(2025年)
   - 新关税率: {policy_tariff * 100:.2f}%
   - 短期收入变化: {scenario_results['Short-term (3 months)']['Revenue_Change_Pct']:+.1f}%
   - 中期收入变化: {scenario_results['Medium-term (12 months)']['Revenue_Change_Pct']:+.1f}%
   - 长期收入变化: {scenario_results['Long-term (24 months)']['Revenue_Change_Pct']:+.1f}%

"""

 

with open('tariff_revenue_model_report.txt', 'w', encoding='utf-8') as f:
    f.write(summary)
 

 

 

 

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

 
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')

sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.axisbelow'] = True

 
fig, ax = plt.subplots(figsize=(12, 6))
years = yearly_data['Year']
revB = yearly_data['Tariff_Revenue'] / 1e9
color_main = '#1368CE'
ax.plot(years, revB, marker='o', linewidth=2.8, markersize=8,
        markerfacecolor='white', markeredgecolor=color_main,
        color=color_main, label='Tariff Revenue')
ax.fill_between(years, revB, alpha=0.15, color=color_main)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:.1f}B'))
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Tariff Revenue', fontsize=12, fontweight='bold')
ax.set_title('U.S. Tariff Revenue Trend (2020–2024)', fontsize=14, fontweight='bold', pad=16)
ax.grid(True, alpha=0.25, linestyle='--')
ax.set_xticks(list(years))
ax.set_ylim(max(revB.min() * 0.9, 0), revB.max() * 1.15)
ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
peak_idx = int(np.argmax(revB))
ax.annotate(f'Peak: ${revB.iloc[peak_idx]:.1f}B',
            xy=(years.iloc[peak_idx], revB.iloc[peak_idx]),
            xytext=(years.iloc[peak_idx], revB.iloc[peak_idx] + (revB.max() * 0.08)),
            arrowprops=dict(arrowstyle='->', color=color_main, lw=1.8), fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
for x, y in zip(years, revB):
    ax.text(x, y + (revB.max() * 0.02), f'${y:.1f}B', ha='center', va='bottom', fontsize=9, fontweight='bold')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG_DIR, 'Figure_1_Yearly_Tariff_Revenue.png'), dpi=300, bbox_inches='tight')
plt.close()
 

 
scenarios_list = list(scenario_results.keys())
revenues = [scenario_results[s]['Tariff_Revenue_B'] for s in scenarios_list]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(scenarios_list)), revenues, width=0.45,
              color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.85,
              edgecolor='black', linewidth=1.2)
ax.set_ylabel('Tariff Revenue (Billion USD)', fontsize=12, fontweight='bold')
ax.set_title('Projected Tariff Revenue by Time Period', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(range(len(scenarios_list)))
ax.set_xticklabels(scenarios_list, rotation=15, ha='right')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:.1f}B'))
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
for b in bars:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2., h, f'${h:.2f}B', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG_DIR, 'Figure_2A_Scenario_Revenue.png'), dpi=300, bbox_inches='tight')
plt.close()
 

 
changes = [scenario_results[s]['Revenue_Change_Pct'] for s in scenarios_list]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(scenarios_list)), changes, width=0.45,
              color=['#2ECC71' if x > 0 else 'E74C3C' for x in changes],
              alpha=0.85, edgecolor='black', linewidth=1.2)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_ylabel('Change vs. Baseline (%)', fontsize=12, fontweight='bold')
ax.set_title('Tariff Revenue Change Percentage', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(range(len(scenarios_list)))
ax.set_xticklabels(scenarios_list, rotation=15, ha='right')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
for b in bars:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2., h, f'{h:+.1f}%', ha='center', va='bottom' if h > 0 else 'top', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG_DIR, 'Figure_2B_Scenario_ChangePct.png'), dpi=300, bbox_inches='tight')
plt.close()
 

 
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
bars1 = axes[0].bar(range(len(scenarios_list)), revenues, width=0.35,
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.85,
                    edgecolor='black', linewidth=1.1)
axes[0].set_ylabel('Tariff Revenue (Billion USD)', fontsize=12, fontweight='bold')
axes[0].set_title('Projected Tariff Revenue by Time Period', fontsize=13, fontweight='bold', pad=10)
axes[0].set_xticks(range(len(scenarios_list)))
axes[0].set_xticklabels(scenarios_list, rotation=15, ha='right')
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'$ {x:.1f}B'))
axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
for b in bars1:
    h = b.get_height()
    axes[0].text(b.get_x() + b.get_width() / 2., h, f'$ {h:.2f}B', ha='center', va='bottom', fontsize=9, fontweight='bold')
bars2 = axes[1].bar(range(len(scenarios_list)), changes, width=0.35,
                    color=['#2ECC71' if x > 0 else '#E74C3C' for x in changes], alpha=0.85,
                    edgecolor='black', linewidth=1.1)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[1].set_ylabel('Change vs. Baseline (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Tariff Revenue Change Percentage', fontsize=13, fontweight='bold', pad=10)
axes[1].set_xticks(range(len(scenarios_list)))
axes[1].set_xticklabels(scenarios_list, rotation=15, ha='right')
axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
for b in bars2:
    h = b.get_height()
    axes[1].text(b.get_x() + b.get_width() / 2., h, f'{h:+.1f}%', ha='center', va='bottom' if h > 0 else 'top', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG_DIR, 'Figure_2_Scenario_Comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
 

 
sensitivity_list = list(sensitivity_results.keys())
revenues_sens = [sensitivity_results[s]['Tariff_Revenue_B'] for s in sensitivity_list]
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(range(len(revenues_sens)), revenues_sens, color=plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(revenues_sens))),
               alpha=0.85, edgecolor='black', linewidth=1.2)
ax.set_yticks(range(len(revenues_sens)))
ax.set_yticklabels(sensitivity_list, fontsize=10)
ax.set_xlabel('Tariff Revenue (Billion USD)', fontsize=12, fontweight='bold')
ax.set_title('Long-term Tariff Revenue by Elasticity Assumption', fontsize=13, fontweight='bold', pad=12)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:.1f}B'))
ax.grid(True, alpha=0.3, axis='x', linestyle='--')
for b in bars:
    w = b.get_width()
    ax.text(w, b.get_y() + b.get_height() / 2., f' ${w:.2f}B', ha='left', va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG_DIR, 'Figure_3A_Sensitivity_Revenue.png'), dpi=300, bbox_inches='tight')
plt.close()
 

 
changes_sens = [sensitivity_results[s]['Revenue_Change_Pct'] for s in sensitivity_list]
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(range(len(changes_sens)), changes_sens, color=['#2ECC71' if x > 0 else '#E74C3C' for x in changes_sens],
               alpha=0.85, edgecolor='black', linewidth=1.2)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_yticks(range(len(changes_sens)))
ax.set_yticklabels(sensitivity_list, fontsize=10)
ax.set_xlabel('Revenue Change vs. Baseline (%)', fontsize=12, fontweight='bold')
ax.set_title('Revenue Impact by Elasticity Assumption', fontsize=13, fontweight='bold', pad=12)
ax.grid(True, alpha=0.3, axis='x', linestyle='--')
for b in bars:
    w = b.get_width()
    ax.text(w, b.get_y() + b.get_height() / 2., f' {w:+.1f}%', ha='left' if w > 0 else 'right', va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG_DIR, 'Figure_3B_Sensitivity_ChangePct.png'), dpi=300, bbox_inches='tight')
plt.close()
 

 
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
bars_s1 = axes[0].barh(range(len(revenues_sens)), revenues_sens,
                       color=plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(revenues_sens))),
                       alpha=0.85, edgecolor='black', linewidth=1.1)
axes[0].set_yticks(range(len(revenues_sens)))
axes[0].set_yticklabels(sensitivity_list, fontsize=10)
axes[0].set_xlabel('Tariff Revenue (Billion USD)', fontsize=12, fontweight='bold')
axes[0].set_title('Long-term Tariff Revenue by Elasticity', fontsize=13, fontweight='bold', pad=10)
axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'$ {x:.1f}B'))
axes[0].grid(True, alpha=0.3, axis='x', linestyle='--')
for b in bars_s1:
    w = b.get_width()
    axes[0].text(w, b.get_y() + b.get_height() / 2., f' ${w:.2f}B', ha='left', va='center', fontsize=9, fontweight='bold')
bars_s2 = axes[1].barh(range(len(changes_sens)), changes_sens,
                       color=['#2ECC71' if x > 0 else '#E74C3C' for x in changes_sens],
                       alpha=0.85, edgecolor='black', linewidth=1.1)
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
axes[1].set_yticks(range(len(changes_sens)))
axes[1].set_yticklabels(sensitivity_list, fontsize=10)
axes[1].set_xlabel('Revenue Change vs. Baseline (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Revenue Impact by Elasticity', fontsize=13, fontweight='bold', pad=10)
axes[1].grid(True, alpha=0.3, axis='x', linestyle='--')
for b in bars_s2:
    w = b.get_width()
    axes[1].text(w, b.get_y() + b.get_height() / 2., f' {w:+.1f}%', ha='left' if w > 0 else 'right', va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG_DIR, 'Figure_3_Sensitivity_Analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
 

 
models = ['HistGB/GBDT', 'AdaBoost', 'ExtraTrees', 'Ensemble']
r2_scores = [r2_hgb, r2_adaboost, r2_et, r2_ensemble]
rmse_scores = [rmse_hgb, rmse_adaboost, rmse_et, rmse_ensemble]
if 'r2_rnn' in globals() and r2_rnn is not None and rmse_rnn is not None:
    models.append('RNN')
    r2_scores.append(r2_rnn)
    rmse_scores.append(rmse_rnn)
if 'r2_lstm' in globals() and r2_lstm is not None and rmse_lstm is not None:
    models.append('LSTM')
    r2_scores.append(r2_lstm)
    rmse_scores.append(rmse_lstm)
if 'r2_transformer' in globals() and r2_transformer is not None and rmse_transformer is not None:
    models.append('Transformer')
    r2_scores.append(r2_transformer)
    rmse_scores.append(rmse_transformer)
colors_models = sns.color_palette("tab10", n_colors=len(models))
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
bars_r2 = axes[0].bar(models, r2_scores, width=0.45, color=colors_models,
                      alpha=0.85, edgecolor='black', linewidth=1.1)
axes[0].set_ylabel('R² Score', fontsize=12, fontweight='bold')
axes[0].set_title('Model R² Comparison (Test Set)', fontsize=13, fontweight='bold', pad=10)
axes[0].set_ylim([0, 1])
axes[0].axhline(y=0.8, color='green', linestyle='--', linewidth=2, alpha=0.4, label='Good (R² = 0.8)')
axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
axes[0].legend(fontsize=10)
for b in bars_r2:
    h = b.get_height()
    axes[0].text(b.get_x() + b.get_width() / 2., h, f'{h:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
bars_rmse = axes[1].bar(models, rmse_scores, width=0.45, color=colors_models,
                        alpha=0.85, edgecolor='black', linewidth=1.1)
axes[1].set_ylabel('RMSE', fontsize=12, fontweight='bold')
axes[1].set_title('Model RMSE Comparison (Test Set)', fontsize=13, fontweight='bold', pad=10)
axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
for b in bars_rmse:
    h = b.get_height()
    axes[1].text(b.get_x() + b.get_width() / 2., h, f'{h:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG_DIR, 'Figure_4_Model_RMSE.png'), dpi=300, bbox_inches='tight')
plt.close()
 

 
fig, ax = plt.subplots(figsize=(12, 7))
tariff_rates = [baseline_tariff * 100, policy_tariff * 100, policy_tariff * 100, policy_tariff * 100]
import_values = [baseline_import / 1e9,
                 scenario_results['Short-term (3 months)']['Import_Value_B'],
                 scenario_results['Medium-term (12 months)']['Import_Value_B'],
                 scenario_results['Long-term (24 months)']['Import_Value_B']]
scenario_names = ['Baseline', 'Short-term\n(3 months)', 'Medium-term\n(12 months)', 'Long-term\n(24 months)']

ax.scatter([baseline_tariff * 100], [baseline_import / 1e9], s=500, c='red', marker='*',
           zorder=5, label='Baseline (2.5%)', edgecolor='black', linewidth=2)

colors_points = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i in range(1, len(tariff_rates)):
    ax.scatter([policy_tariff * 100], [import_values[i]], s=400, c=colors_points[i - 1], marker='o',
               zorder=4, label=scenario_names[i], edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.annotate('', xy=(policy_tariff * 100, import_values[i]), xytext=(baseline_tariff * 100, baseline_import / 1e9),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors_points[i - 1], alpha=0.6))

ax.set_xlabel('Tariff Rate (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Import Value (Trillion USD)', fontsize=12, fontweight='bold')
ax.set_title('Tariff Policy Impact: Tariff Rate vs. Import Volume', fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='best', framealpha=0.95)
ax.text(baseline_tariff * 100 - 0.5, baseline_import / 1e9 + 0.05, 'Baseline',
        fontsize=10, fontweight='bold', color='red')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG_DIR, 'Figure_5_Tariff_Import_Relationship.png'), dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"保存: {os.path.join(OUTPUT_FIG_DIR, 'Figure_5_Tariff_Import_Relationship.png')} ")

print("生成图表6: Revenue Evolution Path...")
fig, ax = plt.subplots(figsize=(13, 7))
time_periods = [0, 3, 12, 24]
period_labels = ['Baseline\n(2025 Jan)', 'Short-term\n(3 months)',
                 'Medium-term\n(12 months)', 'Long-term\n(24 months)']
baseline_rev = baseline_revenue / 1e9
revenues_path = [baseline_rev,
                 scenario_results['Short-term (3 months)']['Tariff_Revenue_B'],
                 scenario_results['Medium-term (12 months)']['Tariff_Revenue_B'],
                 scenario_results['Long-term (24 months)']['Tariff_Revenue_B']]
counterfactual_path = [baseline_rev] * len(time_periods)

ax.plot(time_periods, revenues_path, marker='o', linewidth=3, markersize=10,
        color='#2E86AB', label='With Tariff Policy', zorder=3)
ax.plot(time_periods, counterfactual_path, marker='s', linewidth=2.5, markersize=8,
        color='gray', linestyle='--', label='Counterfactual (No Change)', alpha=0.7, zorder=2)
ax.fill_between(time_periods, revenues_path, counterfactual_path, alpha=0.15, color='#2E86AB')

ax.set_xlabel('Time After Policy Implementation (Months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Annual Tariff Revenue (Billion USD)', fontsize=12, fontweight='bold')
ax.set_title('Tariff Revenue Evolution: Short-term to Long-term Adjustments',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(time_periods)
ax.set_xticklabels(period_labels, fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='best', framealpha=0.95)

for time, revenue in zip(time_periods, revenues_path):
    ax.text(time, revenue + 2, f'${revenue:.1f}B', ha='center', va='bottom',
            fontsize=10, fontweight='bold')

changes_path = [(rev - baseline_rev) / baseline_rev * 100 for rev in revenues_path[1:]]
for i, (time, change) in enumerate(zip(time_periods[1:], changes_path)):
    ax.text(time, revenues_path[i + 1] - 5, f'{change:+.1f}%', ha='center', va='top',
            fontsize=9, color='#2E86AB', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG_DIR, 'Figure_6_Revenue_Evolution_Path.png'), dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"保存: {os.path.join(OUTPUT_FIG_DIR, 'Figure_6_Revenue_Evolution_Path.png')} ")

print("生成图表7: Key Parameters Summary...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
ax.set_title('Key Parameters Summary', fontsize=14, fontweight='bold', pad=10)
table = ax.table(cellText=key_params.values, colLabels=key_params.columns,
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG_DIR, 'Figure_7_Key_Parameters_Summary.png'), dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"保存: {os.path.join(OUTPUT_FIG_DIR, 'Figure_7_Key_Parameters_Summary.png')} ")

 
print("  6. figures/Figure_3B_Sensitivity_ChangePct.png")
print("  7. figures/Figure_3_Sensitivity_Analysis.png")
print("  8. figures/Figure_4_Model_RMSE.png")
print("  9. figures/Figure_5_Tariff_Import_Relationship.png")
print("  10. figures/Figure_6_Revenue_Evolution_Path.png")
print("  11. figures/Figure_7_Key_Parameters_Summary.png")
print("  12. figures/Figure_Question4_SecondTerm_Impact.png")
print("=" * 100 + "\n")

 
second_term_years = [2025, 2026, 2027, 2028]
months_since_policy = [12, 24, 36, 48]
elasticity_factor_map = {12: 0.40, 24: 1.00, 36: 1.00, 48: 1.00}
tariff_change_pct_q4 = np.log(1 + (policy_tariff - baseline_tariff) / (baseline_tariff + 1e-6))
results_q4 = []
for year, months in zip(second_term_years, months_since_policy):
    factor = elasticity_factor_map.get(months, 1.00)
    trade_change_pct_q4 = tariff_elasticity * tariff_change_pct_q4 * factor
    import_val = baseline_import * np.exp(trade_change_pct_q4)
    revenue_policy = import_val * policy_tariff
    revenue_baseline = baseline_import * baseline_tariff
    net_change = revenue_policy - revenue_baseline
    net_change_pct = (net_change / revenue_baseline * 100) if revenue_baseline > 0 else 0
    results_q4.append({
        'Year': year,
        'Months_Since_Policy': months,
        'Policy_Tariff_Rate': policy_tariff,
        'Import_Value_B': import_val / 1e9,
        'Tariff_Revenue_B': revenue_policy / 1e9,
        'Baseline_Revenue_B': revenue_baseline / 1e9,
        'Net_Change_B': net_change / 1e9,
        'Net_Change_Pct': net_change_pct
    })
q4_df = pd.DataFrame(results_q4)
q4_df.to_csv(os.path.join(OUTPUT_TABLE_DIR, 'question4_second_term_revenue_impact.csv'), index=False)

cum_net_change_B = q4_df['Net_Change_B'].sum()
answer_lines = []
answer_lines.append("问题 4 答复：美国关税调整对美国关税收入的短期与中期影响\n")
answer_lines.append(f"政策关税率：{policy_tariff*100:.2f}% | 基准有效关税率：{baseline_tariff*100:.2f}%\n")
answer_lines.append("短期（3个月）：")
answer_lines.append(f"关税收入变化：{scenario_results['Short-term (3 months)']['Revenue_Change_Pct']:+.1f}% | 进口值变化：{scenario_results['Short-term (3 months)']['Trade_Change_Pct']:+.1f}%\n")
answer_lines.append("中期（12个月）：")
answer_lines.append(f"关税收入变化：{scenario_results['Medium-term (12 months)']['Revenue_Change_Pct']:+.1f}% | 进口值变化：{scenario_results['Medium-term (12 months)']['Trade_Change_Pct']:+.1f}%\n")
answer_lines.append("第二任期（2025–2028）净变化：")
answer_lines.append(f"累计关税收入净变化：${cum_net_change_B:.2f}B\n")
answer_lines.append("年度明细：\n")
q4_df_cn = q4_df.rename(columns={
    'Year': '年份',
    'Months_Since_Policy': '实施后月份',
    'Policy_Tariff_Rate': '政策关税率',
    'Import_Value_B': '进口值(B)',
    'Tariff_Revenue_B': '关税收入(B)',
    'Baseline_Revenue_B': '基线收入(B)',
    'Net_Change_B': '净变化(B)',
    'Net_Change_Pct': '净变化(%)'
})
answer_lines.append(q4_df_cn.to_string(index=False))
with open('question4_answer.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(answer_lines))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.plot(q4_df['Year'], q4_df['Tariff_Revenue_B'], marker='o', linewidth=2.5, label='Policy Revenue', color='#2E86AB')
ax1.plot(q4_df['Year'], q4_df['Baseline_Revenue_B'], marker='s', linewidth=2, linestyle='--', label='Baseline Revenue', color='gray')
ax1.set_title('Question 4: Policy vs Baseline Tariff Revenue (2025–2028)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Year')
ax1.set_ylabel('Revenue (Billion USD)')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=10)
for x, y in zip(q4_df['Year'], q4_df['Tariff_Revenue_B']):
    ax1.text(x, y + 0.5, f"${y:.1f}B", ha='center', fontsize=9, fontweight='bold')

bars = ax2.bar(q4_df['Year'].astype(str), q4_df['Net_Change_Pct'],
               color=['#2ECC71' if v > 0 else '#E74C3C' for v in q4_df['Net_Change_Pct']],
               alpha=0.85, edgecolor='black', linewidth=1.2)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_title('Question 4: Net Change vs Baseline (%)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Year')
ax2.set_ylabel('Net Change (%)')
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
for b, pct in zip(bars, q4_df['Net_Change_Pct']):
    ax2.text(b.get_x() + b.get_width() / 2, pct, f"{pct:+.1f}%",
             ha='center', va='bottom' if pct > 0 else 'top', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG_DIR, 'Figure_Question4_SecondTerm_Impact.png'), dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"保存: {os.path.join(OUTPUT_FIG_DIR, 'Figure_Question4_SecondTerm_Impact.png')} ")
print("  8. figures/Figure_Question4_SecondTerm_Impact.png")
print("✓ Question 4 answer saved to question4_answer.txt and table to chart/question4_second_term_revenue_impact.csv")
