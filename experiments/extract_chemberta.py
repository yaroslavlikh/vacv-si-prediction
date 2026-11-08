from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DATA = ROOT / "data/molecular_activity"
MODEL_ID = "DeepChem/ChemBERTa-77M-MTR"
MODEL_DIR = OUT / "models" / "ChemBERTa-77M-MTR"
EMB_PATH = OUT / "chemberta_embeddings.npz"

if EMB_PATH.exists():
    print("Embeddings already exist:", EMB_PATH, flush=True)
    raise SystemExit(0)

start = time.time()
MODEL_DIR.mkdir(parents=True, exist_ok=True)
print("Downloading/loading", MODEL_ID, flush=True)
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=str(MODEL_DIR),
    allow_patterns=["*.json", "*.txt", "*.model", "*.bin", "*.safetensors", "merges.txt", "vocab.json", "tokenizer*"],
)
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
model = AutoModel.from_pretrained(str(MODEL_DIR), local_files_only=True)
model.eval()
torch.set_num_threads(4)

tr = pd.read_csv(DATA / "train.csv")
te = pd.read_csv(DATA / "test.csv")
smiles = tr.SMILES.astype(str).tolist() + te.SMILES.astype(str).tolist()
canonical = []
for s in smiles:
    m = Chem.MolFromSmiles(s)
    try:
        m = rdMolStandardize.FragmentParent(m)
    except Exception:
        pass
    canonical.append(Chem.MolToSmiles(m) if m is not None else s)

cls_rows = []
mean_rows = []
with torch.inference_mode():
    for start_i in range(0, len(canonical), 32):
        batch = canonical[start_i:start_i + 32]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        hidden = model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        cls_rows.append(hidden[:, 0].cpu().numpy().astype(np.float32))
        mean_rows.append(mean.cpu().numpy().astype(np.float32))
        print(f"Embedded {min(start_i + 32, len(canonical))}/{len(canonical)}", flush=True)

cls = np.vstack(cls_rows)
mean = np.vstack(mean_rows)
np.savez_compressed(EMB_PATH, cls_train=cls[:len(tr)], cls_test=cls[len(tr):],
                    mean_train=mean[:len(tr)], mean_test=mean[len(tr):])
metadata = {
    "model": MODEL_ID,
    "n_train": len(tr), "n_test": len(te), "embedding_dim": int(cls.shape[1]),
    "pooling": ["first_token", "attention_masked_mean"],
    "seconds": time.time() - start,
    "targets_used_for_embedding": False,
    "test_concentrations_used": False,
}
(OUT / "chemberta_embedding_report.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps(metadata, indent=2), flush=True)
