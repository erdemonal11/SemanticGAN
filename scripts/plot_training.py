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
        

        df_clean = df.tail(20) 
        

        df_clean.to_csv(LOG_FILE, index=False)
        print("Log data cleaned for the latest run.")

        plt.figure(figsize=(10, 5))
        plt.plot(df_clean["Epoch"], df_clean["D_Loss"], label="Discriminator Loss", color="#1f77b4")
        plt.plot(df_clean["Epoch"], df_clean["G_Loss"], label="Generator Loss", color="#ff7f0e")
        
        plt.xlabel("Epochs")
        plt.ylabel("WGAN Loss")
        plt.title("GAN Training Convergence (Latest Run)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(OUTPUT_IMG)
        print(f"Cleaned plot saved to {OUTPUT_IMG}")
        
    except Exception as e:
        print(f"Error during cleaning or plotting: {e}")

if __name__ == "__main__":
    clean_and_plot()