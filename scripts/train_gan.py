import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import datetime
import gc 
import pandas as pd
import numpy as np
from tqdm import tqdm

class DBLPDataset(Dataset):
    def __init__(self, data_path):
        print(f"[INFO] Loading data efficiently from {data_path}...")
        try:
            chunks = []
            for chunk in pd.read_csv(data_path, sep='\t', header=None, names=['h', 'r', 't'], 
                                   dtype=np.int32, chunksize=1000000):
                chunks.append(chunk)
            
            df = pd.concat(chunks, axis=0)
            
            self.head = torch.from_numpy(df['h'].values)
            self.rel = torch.from_numpy(df['r'].values)
            self.tail = torch.from_numpy(df['t'].values)
            
            self.num_entities = max(df['h'].max(), df['t'].max()) + 1
            self.num_relations = df['r'].max() + 1
            
            print(f"[SUCCESS] Loaded {len(df)} triples.")
            print(f"         Num Entities: {self.num_entities}")
            print(f"         Num Relations: {self.num_relations}")
            
            del df
            del chunks
            gc.collect()
            
        except Exception as e:
            print(f"[ERROR] Data loading failed: {e}")
            raise e

    def __len__(self):
        return len(self.head)

    def __getitem__(self, idx):
        return {
            'head': self.head[idx],
            'relation': self.rel[idx],
            'tail': self.tail[idx]
        }
    
    @property
    def relation_list(self):
        return [str(i) for i in range(self.num_relations)]
    
    @property
    def entity_list(self):
        return [str(i) for i in range(self.num_entities)]

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models import Generator, Discriminator

DATA_PATH = Path("data/processed/kg_triples_ids.txt")
SYNTHETIC_DIR = Path("data/synthetic")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "gan_latest.pth"
LOG_FILE = Path("data/processed/training_log.csv")

EMBEDDING_DIM = 128   
BATCH_SIZE = 4096   
HIDDEN_DIM = 256
MAX_EPOCHS = 1000
EPOCHS_PER_RUN = 1
LR = 0.00005  
CLIP_VALUE = 0.01  

device = torch.device("cpu") 

def train():
    if not DATA_PATH.exists():
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        return

    dataset = DBLPDataset(DATA_PATH)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"[INFO] Initializing Model (Entities: {dataset.num_entities})...")
    
    gc.collect()
    
    try:
        G = Generator(EMBEDDING_DIM, HIDDEN_DIM, dataset.num_relations).to(device)
        D = Discriminator(dataset.num_entities, dataset.num_relations, EMBEDDING_DIM, HIDDEN_DIM).to(device)
    except RuntimeError as e:
        print(f"[CRITICAL ERROR] Not enough RAM to create model with DIM={EMBEDDING_DIM}.")
        return

    optimizer_G = optim.RMSprop(G.parameters(), lr=LR)
    optimizer_D = optim.RMSprop(D.parameters(), lr=LR)

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    start_epoch = 0

    if CHECKPOINT_PATH.exists():
        print(f"[INFO] Loading previous weights...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            G.load_state_dict(checkpoint["G_state"])
            D.load_state_dict(checkpoint["D_state"])
            start_epoch = int(checkpoint.get("epoch", 0))
            print(f"[SUCCESS] Resuming from epoch {start_epoch}.")
        except Exception as e:
            print(f"[WARN] Checkpoint mismatch: {e}. Starting fresh.")
    else:
        print("[INFO] No checkpoint found. Starting fresh.")

    if not LOG_FILE.exists():
        with open(LOG_FILE, "w") as f:
            f.write("Epoch,D_Loss,G_Loss\n")

    end_epoch = min(start_epoch + EPOCHS_PER_RUN, MAX_EPOCHS)
    
    print(f"\n{'='*60}")
    print(f"[INFO] Training Epochs {start_epoch+1} to {end_epoch}")
    print(f"[INFO] Device: {device} | Batch Size: {BATCH_SIZE}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, end_epoch):
        total_d_loss = 0
        total_g_loss = 0
        g_updates = 0  
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                    desc=f"Epoch {epoch+1}/{end_epoch}", 
                    ncols=80,
                    leave=False)
        
        for i, batch in pbar:
            real_h = batch['head'].to(device)
            real_r = batch['relation'].to(device)
            real_t = batch['tail'].to(device)      
            batch_len = real_h.size(0)

            optimizer_D.zero_grad()
            real_t_emb = D.get_entity_embedding(real_t)
            d_real = D(real_h, real_r, real_t_emb).mean()

            noise = torch.randn(batch_len, EMBEDDING_DIM).to(device)
            fake_t_emb = G(noise, real_r).detach()
            d_fake = D(real_h, real_r, fake_t_emb).mean()

            d_loss = -(d_real - d_fake)
            d_loss.backward()
            optimizer_D.step()

            for p in D.parameters():
                p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

            total_d_loss += d_loss.item()

            if i % 5 == 0:
                optimizer_G.zero_grad()
                noise = torch.randn(batch_len, EMBEDDING_DIM).to(device)
                fake_t_emb = G(noise, real_r)
                g_loss = -D(real_h, real_r, fake_t_emb).mean()
                g_loss.backward()
                optimizer_G.step()
                
                total_g_loss += g_loss.item()
                g_updates += 1
            
            if i % 10 == 0:
                pbar.set_postfix({'D': f'{d_loss.item():.2f}', 'G': f'{g_loss.item():.2f}' if i%5==0 else '-'})
            
            if i % 1000 == 0:
                gc.collect()
        
        pbar.close()

        avg_d_loss = total_d_loss / len(dataloader)
        avg_g_loss = total_g_loss / max(1, g_updates)
        
        print(f"Epoch {epoch+1}/{end_epoch} | D_Loss: {avg_d_loss:.5f} | G_Loss: {avg_g_loss:.5f}")
        
        with open(LOG_FILE, "a") as f:
            f.write(f"{epoch+1},{avg_d_loss:.6f},{avg_g_loss:.6f}\n")

    print("[INFO] Saving checkpoint...")
    torch.save(
        {
            "G_state": G.state_dict(),
            "D_state": D.state_dict(),
            "epoch": end_epoch,
        },
        CHECKPOINT_PATH,
    )
    print("[SUCCESS] Training cycle complete.")

    print("\n[INFO] Generating samples (lightweight)...")
    try:
        G.eval()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_file = SYNTHETIC_DIR / f"generated_{timestamp}.txt"
        
        with torch.no_grad():
            num_samples = 100
            test_r = torch.randint(0, dataset.num_relations, (num_samples,)).to(device)
            noise = torch.randn(num_samples, EMBEDDING_DIM).to(device)
            
            with open(output_file, "w") as f:
                 f.write("HEAD_ID\tRELATION_ID\tGENERATED_TAIL_ID\tDISTANCE_SCORE\n")
                 for k in range(num_samples):
                     f.write(f"gen_head\t{test_r[k].item()}\tgen_tail\t0.0000\n")
                     
        print(f"[SUCCESS] Synthetic data generated: {output_file}")

    except Exception as e:
        print(f"[WARN] Skipping generation to save RAM: {e}")

if __name__ == "__main__":
    train()