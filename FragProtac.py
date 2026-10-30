# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch.utils.data import Dataset as TorchDataset
from rdkit import Chem
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, roc_curve)
from torch.nn.utils import weight_norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ====================== 三元注意力 TAN + FCNet ======================
class FCNet(nn.Module):
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()
        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())
        self.fcnet = nn.Sequential(*layers)

    def forward(self, x):
        return self.fcnet(x)

class TAN(nn.Module):
    def __init__(self, in_dims, h_dim, h_out, dropouts=(0.2,0.2,0.2), act='ReLU', k=1):
        super(TAN, self).__init__()
        self.k = k
        self.x_dim, self.y_dim, self.z_dim = in_dims
        self.h_dim = h_dim
        self.h_out = h_out
        self.x_net = FCNet([self.x_dim, h_dim * k], act=act, dropout=dropouts[0])
        self.y_net = FCNet([self.y_dim, h_dim * k], act=act, dropout=dropouts[1])
        self.z_net = FCNet([self.z_dim, h_dim * k], act=act, dropout=dropouts[2])
        if 1 < k:
            self.p_net = nn.AvgPool1d(kernel_size=k, stride=k)
        self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)
        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, x, y, z, att_map):
        xy = torch.einsum('bxk,byk->bxyk', (x, y))
        xy = xy.permute(0, 2, 1, 3).contiguous()
        fusion_logits = torch.einsum('byxk,bxyz,bzk->bk', (xy, att_map, z))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k
        return fusion_logits

    def forward(self, x, y, z, softmax=True):
        min_batch = min(x.size(0), y.size(0), z.size(0))
        if x.size(0) != min_batch:
            x = x[:min_batch]
        if y.size(0) != min_batch:
            y = y[:min_batch]
        if z.size(0) != min_batch:
            z = z[:min_batch]

        x_num, y_num, z_num = x.size(1), y.size(1), z.size(1)
        _x, _y, _z = self.x_net(x), self.y_net(y), self.z_net(z)
        _xyz = torch.einsum('bxk,byk,bzk->bxyzk', (_x, _y, _z))
        att_maps = self.h_net(_xyz).permute(0,4,1,2,3)
        if softmax:
            p = F.softmax(att_maps.view(-1, self.h_out, x_num*y_num*z_num), 2)
            att_maps = p.view(-1, self.h_out, x_num, y_num, z_num)
        logits = self.attention_pooling(_x,_y,_z,att_maps[:,0,:,:,:])
        for i in range(1,self.h_out):
            logits += self.attention_pooling(_x,_y,_z,att_maps[:,i,:,:,:])
        logits = self.bn(logits)
        return logits, att_maps

# ====================== 全局配置 ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "/FragProtac/data/PROTAC-fine/protac-fine.csv"
OUTPUT_DIR = "/FragProtac"
GCN_LAYERS = 3
GCN_HIDDEN_DIM = 128
FRAG_FEAT_DIM = 256
PATHWAY_DIM = 128
TAN_H_DIM = 256
TAN_H_OUT = 4
FUSION_HIDDEN_DIM = 256
NUM_CLASSES = 2

BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-4

# ====================== 分子图构建（鲁棒） ======================
x_map = {
    'atomic_num': list(range(0, 119)),
    'chirality': ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER','CHI_TETRAHEDRAL','CHI_ALLENE','CHI_SQUAREPLANAR','CHI_TRIGONALBIPYRAMIDAL','CHI_OCTAHEDRAL'],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': ['UNSPECIFIED', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER'],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}
e_map = {
    'bond_type': ['UNSPECIFIED', 'SINGLE', 'DOUBLE', 'TRIPLE', 'QUADRUPLE', 'QUINTUPLE', 'HEXTUPLE','ONEANDAHALF','TWOANDAHALF','THREEANDAHALF','FOURANDAHALF','FIVEANDAHALF','AROMATIC','IONIC','HYDROGEN','THREECENTER','DATIVEONE','DATIVE','DATIVEL','DATIVER','OTHER','ZERO'],
    'stereo': ['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS'],
    'is_conjugated': [False, True],
}

def smiles_to_graph(smiles):
    if pd.isna(smiles) or smiles.strip() == "":
        smiles = "C"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        mol = Chem.MolFromSmiles('C')
    xs = []
    for atom in mol.GetAtoms():
        try:
            x = [
                x_map['atomic_num'].index(atom.GetAtomicNum()),
                x_map['chirality'].index(str(atom.GetChiralTag())),
                x_map['degree'].index(atom.GetTotalDegree()),
                x_map['formal_charge'].index(atom.GetFormalCharge()),
                x_map['num_hs'].index(atom.GetTotalNumHs()),
                x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()),
                x_map['hybridization'].index(str(atom.GetHybridization())),
                x_map['is_aromatic'].index(atom.GetIsAromatic()),
                x_map['is_in_ring'].index(atom.IsInRing())
            ]
        except ValueError:
            x = [0] * 9
        xs.append(x)
    x = torch.tensor(xs, dtype=torch.float)
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        try:
            e = [
                e_map['bond_type'].index(str(bond.GetBondType())),
                e_map['stereo'].index(str(bond.GetStereo())),
                e_map['is_conjugated'].index(bond.GetIsConjugated())
            ]
        except ValueError:
            e = [0, 0, 0]
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]
    if edge_indices:
        edge_index = torch.tensor(edge_indices).t().long()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
    return x, edge_index, edge_attr

# ====================== 片段GCN编码器 ======================
class FragmentGCN(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=GCN_HIDDEN_DIM, out_dim=FRAG_FEAT_DIM, num_layers=GCN_LAYERS):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers-2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.relu = nn.ReLU()
        self.proj = nn.Linear(hidden_dim*2, out_dim)

    def forward(self, x, edge_index, edge_attr, batch_idx):
        for conv in self.convs:
            x = self.relu(conv(x, edge_index))
        mean_pool = global_mean_pool(x, batch_idx)
        max_pool = global_max_pool(x, batch_idx)
        pool_feat = torch.cat([mean_pool, max_pool], dim=1)
        out = self.proj(pool_feat)
        return out

# ====================== 数据集类 ======================
class PROTACFragmentDataset(TorchDataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig', sep='\t')
        self.df.columns = self.df.columns.str.strip()
        print("CSV列名:", self.df.columns.tolist())
        
        required_cols = ['label', 'warhead smiles', 'e3 ligand smiles', 'linker smiles', 'Pathway_Combined_128d']
        for col in required_cols:
            if col not in self.df.columns:
                raise KeyError(f"缺少必要列: {col}，可用列: {self.df.columns.tolist()}")
        
        self.df['label'] = self.df['label'].map({True: 1, False: 0})
        self.df = self.df[self.df['label'].notna()].reset_index(drop=True)
        
        self.warhead_smi = self.df["warhead smiles"].tolist()
        self.linker_smi = self.df["linker smiles"].tolist()
        self.e3_ligand_smi = self.df["e3 ligand smiles"].tolist()
        self.pathway_vec = self._parse_pathway_vec(self.df["Pathway_Combined_128d"])
        self.labels = torch.tensor(self.df["label"].astype(int).tolist(), dtype=torch.long)

    def _parse_pathway_vec(self, series):
        vec_list = []
        for s in series:
            if pd.isna(s) or s == '':
                vec = torch.zeros(PATHWAY_DIM, dtype=torch.float)
            else:
                parts = s.split(',')
                vec = torch.tensor([float(i) for i in parts[:PATHWAY_DIM]], dtype=torch.float)
            vec_list.append(vec)
        return vec_list

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        warhead_x, warhead_edge, warhead_edge_attr = smiles_to_graph(self.warhead_smi[idx])
        linker_x, linker_edge, linker_edge_attr = smiles_to_graph(self.linker_smi[idx])
        e3_x, e3_edge, e3_edge_attr = smiles_to_graph(self.e3_ligand_smi[idx])
        return {
            "warhead": (warhead_x, warhead_edge, warhead_edge_attr),
            "linker": (linker_x, linker_edge, linker_edge_attr),
            "e3_ligand": (e3_x, e3_edge, e3_edge_attr),
            "pathway": self.pathway_vec[idx],
            "label": self.labels[idx]
        }

# ====================== 批次拼接collate_fn ======================
def fragment_collate_fn(batch):
    def collate_graph(graph_list):
        all_x, all_edge, all_edge_attr, batch_idx = [], [], [], []
        cur = 0
        sample_idx = 0
        for g in graph_list:
            x, e_idx, e_attr = g
            if x.shape[0] == 0:
                x = torch.zeros((1, 9), dtype=torch.float)
                e_idx = torch.empty((2, 0), dtype=torch.long)
                e_attr = torch.empty((0, 3), dtype=torch.float)
            n = x.shape[0]
            all_x.append(x)
            all_edge.append(e_idx + cur if e_idx.numel() > 0 else e_idx)
            all_edge_attr.append(e_attr)
            batch_idx.extend([sample_idx] * n)
            cur += n
            sample_idx += 1
        batch_x = torch.cat(all_x, dim=0).to(DEVICE)
        batch_edge = torch.cat(all_edge, dim=1).to(DEVICE) if all_edge else torch.empty((2,0), dtype=torch.long).to(DEVICE)
        batch_edge_attr = torch.cat(all_edge_attr, dim=0).to(DEVICE) if all_edge_attr else torch.empty((0,3), dtype=torch.float).to(DEVICE)
        batch_idx = torch.tensor(batch_idx, dtype=torch.long).to(DEVICE)
        return batch_x, batch_edge, batch_edge_attr, batch_idx

    warhead_graphs = [item["warhead"] for item in batch]
    linker_graphs = [item["linker"] for item in batch]
    e3_graphs = [item["e3_ligand"] for item in batch]

    w_x, w_ei, w_ea, w_batch = collate_graph(warhead_graphs)
    l_x, l_ei, l_ea, l_batch = collate_graph(linker_graphs)
    e_x, e_ei, e_ea, e_batch = collate_graph(e3_graphs)

    pathway = torch.stack([item["pathway"] for item in batch]).to(DEVICE)
    labels = torch.stack([item["label"] for item in batch]).to(DEVICE)

    return {
        "warhead_x": w_x, "warhead_edge": w_ei, "warhead_edge_attr": w_ea, "warhead_batch": w_batch,
        "linker_x": l_x, "linker_edge": l_ei, "linker_edge_attr": l_ea, "linker_batch": l_batch,
        "e3_x": e_x, "e3_edge": e_ei, "e3_edge_attr": e_ea, "e3_batch": e_batch,
        "pathway": pathway,
        "label": labels
    }

# ====================== 主模型 ======================
class FragmentPathwayModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.frag_gcn = FragmentGCN()
        self.tan = TAN(in_dims=(FRAG_FEAT_DIM, FRAG_FEAT_DIM, FRAG_FEAT_DIM), h_dim=TAN_H_DIM, h_out=TAN_H_OUT)
        self.pathway_proj = nn.Linear(PATHWAY_DIM, TAN_H_DIM)
        
        self.query_proj = nn.Linear(TAN_H_DIM, TAN_H_DIM)
        self.key_proj = nn.Linear(TAN_H_DIM, TAN_H_DIM)
        self.value_proj = nn.Linear(TAN_H_DIM, TAN_H_DIM)
        self.scale = (TAN_H_DIM) ** -0.5
        self.norm = nn.LayerNorm(TAN_H_DIM)
        
        self.classifier = nn.Sequential(
            nn.Linear(TAN_H_DIM, FUSION_HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(FUSION_HIDDEN_DIM, FUSION_HIDDEN_DIM//2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(FUSION_HIDDEN_DIM//2, FUSION_HIDDEN_DIM//4), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(FUSION_HIDDEN_DIM//4, NUM_CLASSES)
        )

    def forward(self, batch):
        warhead_feat = self.frag_gcn(batch["warhead_x"], batch["warhead_edge"], batch["warhead_edge_attr"], batch["warhead_batch"])
        linker_feat = self.frag_gcn(batch["linker_x"], batch["linker_edge"], batch["linker_edge_attr"], batch["linker_batch"])
        e3_feat = self.frag_gcn(batch["e3_x"], batch["e3_edge"], batch["e3_edge_attr"], batch["e3_batch"])
        
        min_batch = min(warhead_feat.size(0), linker_feat.size(0), e3_feat.size(0))
        warhead_feat = warhead_feat[:min_batch]
        linker_feat = linker_feat[:min_batch]
        e3_feat = e3_feat[:min_batch]
        batch["pathway"] = batch["pathway"][:min_batch]
        batch["label"] = batch["label"][:min_batch]
        
        w_exp = warhead_feat.unsqueeze(1)
        l_exp = linker_feat.unsqueeze(1)
        e_exp = e3_feat.unsqueeze(1)
        frag_fused, _ = self.tan(w_exp, l_exp, e_exp)
        
        pathway_proj = self.pathway_proj(batch["pathway"])
        
        Q = self.query_proj(frag_fused).unsqueeze(1)
        K = self.key_proj(pathway_proj).unsqueeze(1)
        V = self.value_proj(pathway_proj).unsqueeze(1)
        
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, V).squeeze(1)
        
        fused = self.norm(frag_fused + attn_out)
        logits = self.classifier(fused)
        return logits

# ====================== 评估函数 ======================
def evaluate_and_plot(model, data_loader, dataset_name, output_dir):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            logits = model(batch)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            labels = batch["label"].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)
    
    acc = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    auc_score = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    
    print(f"\n========== {dataset_name} 评估结果 ==========")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc_score:.4f}")
    
    with open(os.path.join(output_dir, f"{dataset_name}_metrics.txt"), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\nAUC: {auc_score:.4f}")
    
    plt.figure(figsize=(6,5))
    if auc_score > 0:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.plot(fpr, tpr, label=f'AUC={auc_score:.3f}', color='darkorange')
        plt.plot([0,1],[0,1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC - {dataset_name}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_roc.png"), dpi=300)
    plt.close()
    
    cm = confusion_matrix(all_labels, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.title(f'CM - {dataset_name}')
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_cm.png"), dpi=300)
    plt.close()
    return acc, precision, recall, f1, auc_score

# ====================== 训练函数 ======================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, output_dir):
    best_auc = 0.0
    best_model_path = os.path.join(output_dir, "best_fragment_pathway_model.pth")
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        train_probs, train_labs = [], []
        for b in train_loader:
            optimizer.zero_grad()
            logits = model(b)
            loss = criterion(logits, b["label"])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            prob = F.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
            train_probs.extend(prob)
            train_labs.extend(b["label"].cpu().numpy())
        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_labs, train_probs) if len(np.unique(train_labs))>1 else 0.0
        
        model.eval()
        val_loss = 0.0
        val_probs, val_labs = [], []
        with torch.no_grad():
            for b in val_loader:
                logits = model(b)
                val_loss += criterion(logits, b["label"]).item()
                val_probs.extend(F.softmax(logits, dim=1)[:,1].cpu().numpy())
                val_labs.extend(b["label"].cpu().numpy())
        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_labs, val_probs) if len(np.unique(val_labs))>1 else 0.0
        
        if (ep+1) % 10 == 0:
            print(f"Epoch{ep+1:3d} | TrainLoss:{train_loss:.4f} TrainAUC:{train_auc:.4f} | ValLoss:{val_loss:.4f} ValAUC:{val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
    
    model.load_state_dict(torch.load(best_model_path))
    return evaluate_and_plot(model, val_loader, "Validation", output_dir)

# ====================== 主函数 ======================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = PROTACFragmentDataset(CSV_PATH)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=fragment_collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=fragment_collate_fn)
    
    print("="*50)
    print("开始训练深度学习模型 (GCN-TAN + Pathway)")
    print("="*50)
    model = FragmentPathwayModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dl_metrics = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, OUTPUT_DIR)

if __name__ == "__main__":
    main()
