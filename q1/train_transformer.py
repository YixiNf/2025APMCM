import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

HAS_TORCH = True
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except Exception:
    HAS_TORCH = False

DATA_DIR = 'preprocessed_data'
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(DATA_DIR, 'train_data.csv'), parse_dates=['Date'])
val = pd.read_csv(os.path.join(DATA_DIR, 'val_data.csv'), parse_dates=['Date'])
test = pd.read_csv(os.path.join(DATA_DIR, 'test_data.csv'), parse_dates=['Date'])
 

targets_q = ['US_Bilateral_Q','AR_Bilateral_Q','BR_Bilateral_Q']
targets_v = ['US_Bilateral_V','AR_Bilateral_V','BR_Bilateral_V']
drop_cols = ['Date'] + targets_q + targets_v
feature_cols = [c for c in train.columns if c not in drop_cols]
 

if not HAS_TORCH:
    raise RuntimeError('PyTorch not available for transformer model')

class SeqDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, window=12):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)
        self.window = window
    def __len__(self):
        return max(0, len(self.X) - self.window)
    def __getitem__(self, idx):
        x = self.X[idx:idx+self.window]
        y = self.y[idx+self.window-1]
        return torch.from_numpy(x), torch.tensor(y)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        l = x.size(1)
        return x + self.pe[:, :l, :]

class TransformerReg(nn.Module):
    def __init__(self, in_dim, d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x):
        z = self.proj(x)
        z = self.pos(z)
        h = self.encoder(z)
        h = h[:,-1,:]
        return self.head(h).squeeze(-1)

def train_model(target_col, epochs=10, window=12):
    ds_train = SeqDataset(train, feature_cols, target_col, window)
    ds_val = SeqDataset(val, feature_cols, target_col, window)
    dl_train = DataLoader(ds_train, batch_size=64, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=64, shuffle=False)
    model = TransformerReg(in_dim=len(feature_cols))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    tr_losses, val_losses = [], []
    
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in dl_train:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        tr_losses.append(ep_loss / max(1,len(ds_train)))
        model.eval()
        ev = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                ev += loss_fn(model(xb), yb).item() * len(xb)
        val_losses.append(ev / max(1,len(ds_val)))
        
    return model, tr_losses, val_losses

def predict_series_with_context(model, df, prev_df=None, window=12):
    if prev_df is not None and len(prev_df) > 0:
        k = min(window, len(prev_df))
        ctx = pd.concat([prev_df.tail(k), df], ignore_index=True)
    else:
        ctx = df.copy()
    Xc = ctx[feature_cols].values.astype(np.float32)
    w = min(window, max(3, len(Xc) - 1))
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(len(Xc) - w):
            xb = torch.from_numpy(Xc[i:i+w]).unsqueeze(0)
            preds.append(model(xb).item())
    preds = np.array(preds)
    preds = preds[-len(df):] if len(preds) >= len(df) else preds
    
    return preds

def plot_test(model, df, target_col, name):
    preds = predict_series_with_context(model, df, prev_df=val, window=12)
    if preds.size == 0:
        preds = predict_series_with_context(model, df, prev_df=val, window=max(6, len(df)))
    y = df[target_col].values[-len(preds):] if preds.size > 0 else df[target_col].values
    plt.figure(figsize=(10,4))
    plt.plot(y, label='actual', marker='o')
    plt.plot(preds, label='pred', marker='o')
    plt.legend()
    plt.title(name)
    out_path = os.path.join(FIG_DIR, f'transformer_pred_{name}.png')
    plt.savefig(out_path)
    plt.close()
    

def load_scaler():
    p = os.path.join(DATA_DIR, 'minmax_scaler.pkl')
    return joblib.load(p) if os.path.exists(p) else None

def denorm_array(a, scaler, col):
    if scaler is None:
        return a
    names = list(getattr(scaler, 'feature_names_in_', []))
    if col not in names:
        return a
    idx = names.index(col)
    mn = float(scaler.data_min_[idx])
    mx = float(scaler.data_max_[idx])
    return a * (mx - mn) + mn if mx > mn else a

def export_predictions(models, df, targets, metric_tag, fname):
    scaler = load_scaler()
    rows = []
    for tgt in targets:
        preds = predict_series_with_context(models[tgt], df, prev_df=val, window=12)
        if preds.size == 0:
            continue
        k = len(preds)
        dts = df['Date'].values[-k:]
        act_n = df[tgt].values[-k:].astype(float)
        pred_n = preds.astype(float)
        act = denorm_array(act_n, scaler, tgt)
        pred = denorm_array(pred_n, scaler, tgt)
        key = tgt.split('_')[0]
        for i in range(k):
            rows.append({'Date': dts[i], 'Country': key, 'Metric': metric_tag, 'ActualNorm': act_n[i], 'PredNorm': pred_n[i], 'Actual': act[i], 'Pred': pred[i]})
    if rows:
        out = pd.DataFrame(rows)
        out_path = os.path.join(FIG_DIR, fname)
        out.to_csv(out_path, index=False)

 
models_q, losses_q = {}, {}
for c in targets_q:
    m, tr, va = train_model(c, epochs=30)
    models_q[c] = m
    losses_q[c] = (tr, va)
 
models_v, losses_v = {}, {}
for c in targets_v:
    m, tr, va = train_model(c, epochs=30)
    models_v[c] = m
    losses_v[c] = (tr, va)
 

 

def plot_group(models, df, targets, name_cn):
    series = {}
    labels = []
    for tgt in targets:
        preds = predict_series_with_context(models[tgt], df, prev_df=val, window=12)
        key = tgt.split('_')[0]
        labels.append(key)
        series[key] = preds
    m = min([len(v) for v in series.values()]) if series else 0
    plt.figure(figsize=(10,4))
    for key in labels:
        plt.plot(series[key][-m:], label=key, marker='o')
    plt.legend()
    plt.title('Three Countries Together - ' + ('Quantity' if name_cn == '数量' else 'Value'))
    out_path = os.path.join(FIG_DIR, 'transfomer_quantity.png' if name_cn == '数量' else 'transfomer_value.png')
    plt.savefig(out_path)
    plt.close()
    # also save as pred_xxx style
    plt.figure(figsize=(10,4))
    for key in labels:
        plt.plot(series[key][-m:], label=key, marker='o')
    plt.legend()
    plt.title('Three Countries Together - ' + ('Quantity' if name_cn == '数量' else 'Value'))
    pred_out = os.path.join(FIG_DIR, 'pred_transformer_quantity.png' if name_cn == '数量' else 'pred_transformer_value.png')
    plt.savefig(pred_out)
    plt.close()

print('plot_group_start')
plot_group(models_q, test, targets_q, '数量')
plot_group(models_v, test, targets_v, '金额')
print('plot_group_done')

export_predictions(models_q, test, targets_q, 'Quantity', 'predictions_transformer_quantity.csv')
export_predictions(models_v, test, targets_v, 'Value', 'predictions_transformer_value.csv')

 
plt.figure(figsize=(10,5))
for k, (tr, va) in losses_q.items():
    plt.plot(tr, label=f'{k.split("_")[0]}-train')
    plt.plot(va, label=f'{k.split("_")[0]}-val', linestyle='--')
plt.legend()
plt.title('Transformer Loss - Quantity (US/AR/BR)')
out_loss_q = os.path.join(FIG_DIR, 'transfomer_loss_quantity.png')
plt.savefig(out_loss_q)
plt.close()

plt.figure(figsize=(10,5))
for k, (tr, va) in losses_v.items():
    plt.plot(tr, label=f'{k.split("_")[0]}-train')
    plt.plot(va, label=f'{k.split("_")[0]}-val', linestyle='--')
plt.legend()
plt.title('Transformer Loss - Value (US/AR/BR)')
out_loss_v = os.path.join(FIG_DIR, 'transfomer_loss_value.png')
plt.savefig(out_loss_v)
plt.close()
