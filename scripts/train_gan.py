import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import datetime
import gc 
from tqdm import tqdm  

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.dataset import DBLPDataset
from src.models import Generator, Discriminator

DATA_PATH = Path("data/processed/kg_triples_ids.txt")
SYNTHETIC_DIR = Path("data/synthetic")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "gan_latest.pth"
LOG_FILE = Path("data/processed/training_log.csv")

EMBEDDING_DIM = 128  
BATCH_SIZE = 2048   
HIDDEN_DIM = 256
MAX_EPOCHS = 1000
EPOCHS_PER_RUN = 1
LR = 0.0002
CLIP_VALUE = 0.01  

device = torch.device("cpu") 

def train():
    if not DATA_PATH.exists():
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        return

    print("[INFO] Loading dataset...")
    dataset = DBLPDataset(DATA_PATH)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"[INFO] Initializing Model (Entities: {dataset.num_entities})...")
    G = Generator(EMBEDDING_DIM, HIDDEN_DIM, dataset.num_relations).to(device)
    D = Discriminator(dataset.num_entities, dataset.num_relations, EMBEDDING_DIM, HIDDEN_DIM).to(device)

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

    if start_epoch >= MAX_EPOCHS:
        print("[INFO] Max epochs reached. Stopping.")
        return
        
    end_epoch = min(start_epoch + EPOCHS_PER_RUN, MAX_EPOCHS)
    print(f"\n{'='*60}")
    print(f"[INFO] Training Epochs {start_epoch+1} to {end_epoch}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, end_epoch):
        total_d_loss = 0
        total_g_loss = 0
        g_updates = 0  
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                    desc=f"Epoch {epoch+1}/{end_epoch}", 
                    ncols=100)
        
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
            
            if g_updates > 0:
                pbar.set_postfix({
                    'D_Loss': f'{d_loss.item():.4f}',
                    'G_Loss': f'{g_loss.item():.4f}' if i % 5 == 0 else 'N/A'
                })
            
            if i % 100 == 0:
                gc.collect()
        
        pbar.close()

        avg_d_loss = total_d_loss / len(dataloader)
        avg_g_loss = total_g_loss / max(1, g_updates)
        
        print(f"\n{'─'*60}")
        print(f"✓ Epoch {epoch+1} Complete:")
        print(f"  ├─ Avg D_Loss: {avg_d_loss:.4f}")
        print(f"  └─ Avg G_Loss: {avg_g_loss:.4f}")
        print(f"{'─'*60}\n")
        
        with open(LOG_FILE, "a") as f:
            f.write(f"{epoch+1},{avg_d_loss:.6f},{avg_g_loss:.6f}\n")

    print("[INFO] Saving optimized checkpoint (Weights Only)...")
    torch.save(
        {
            "G_state": G.state_dict(),
            "D_state": D.state_dict(),
            "epoch": end_epoch,
        },
        CHECKPOINT_PATH,
    )
    print("[SUCCESS] Checkpoint saved.")

    print("\n[INFO] Generating synthetic samples...")
    G.eval()
    D.eval() 
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_file = SYNTHETIC_DIR / f"generated_{timestamp}.txt"
    
    with torch.no_grad():
        num_samples = 1000
        test_r = torch.randint(0, dataset.num_relations, (num_samples,)).to(device)
        noise = torch.randn(num_samples, EMBEDDING_DIM).to(device)
        fake_emb = G(noise, test_r)
        
        all_ent_emb = D.ent_embedding.weight.data
        
        with open(output_file, "w") as f:
            f.write("HEAD_ID\tRELATION_ID\tGENERATED_TAIL_ID\tDISTANCE_SCORE\n")
            
            gen_pbar = tqdm(range(num_samples), desc="Generating samples", ncols=100)
            for k in gen_pbar:
                r_str = dataset.relation_list[test_r[k].item()]
                
                subset_indices = torch.randint(0, len(all_ent_emb), (2000,)) 
                subset_emb = all_ent_emb[subset_indices]
                
                current_emb = fake_emb[k].unsqueeze(0)
                dist = torch.norm(subset_emb - current_emb, dim=1)
                best_local = torch.argmin(dist).item()
                best_global = subset_indices[best_local].item()
                
                t_str = dataset.entity_list[best_global]
                h_str = "generated_context" 
                score = dist[best_local].item()
                f.write(f"{h_str}\t{r_str}\t{t_str}\t{score:.4f}\n")
            
            gen_pbar.close()
    
    print(f"[SUCCESS] Generated {num_samples} samples → {output_file}")
    print(f"\n{'='*60}")
    print("Training session completed!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    train()