import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

LOG_FILE = Path("data/processed/training_log.csv")
OUTPUT_IMG = Path("data/processed/loss_curve.png")

def clean_and_plot():
    if not LOG_FILE.exists():
        print("No log file found.")
        return

    try:
        df = pd.read_csv(LOG_FILE)
        df["Epoch"] = df["Epoch"].astype(int)
        df_clean = df.tail(20).sort_values("Epoch")
        df_clean.to_csv(LOG_FILE, index=False)

        plt.rcParams.update({
            "text.usetex": False,  
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm"
        })

        plt.figure(figsize=(10, 5), dpi=300)
        
        plt.plot(df_clean["Epoch"], df_clean["D_Loss"], 
                 label=r"Discriminator Loss ($D$)", color="#1f77b4", linewidth=2)
        plt.plot(df_clean["Epoch"], df_clean["G_Loss"], 
                 label=r"Generator Loss ($G$)", color="#ff7f0e", linewidth=2)
        
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("WGAN Loss", fontsize=12)
        plt.title(r"$\text{GAN Training Convergence}$", fontsize=14, pad=15)
        
        plt.legend(frameon=True, loc='center right', fontsize=10)
        plt.grid(True, which='both', linestyle=':', alpha=0.4)
        plt.tight_layout()
        
        plt.savefig(OUTPUT_IMG)
        print(f"Cleaned plot saved to {OUTPUT_IMG}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    clean_and_plot()