import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_smooth(log_dir, window=50):
    file_path = os.path.join(log_dir, "monitor.csv")
    # Skip the first row (metadata) and load
    data = pd.read_csv(file_path, skiprows=1)
    y = data['r'].values # Cumulative rewards per episode
    # Moving average for smoothing
    y_smooth = pd.Series(y).rolling(window=window).mean()
    return y, y_smooth

def plot_training_comparison():
    asac_dir = "./logs/asac_pure/"
    hybrid_dir = "./logs/asac_hybrid/"
    
    plt.figure(figsize=(12, 7))

    # Load data
    try:
        raw_asac, smooth_asac = load_and_smooth(asac_dir)
        raw_hybrid, smooth_hybrid = load_and_smooth(hybrid_dir)

        # Plot ASAC
        plt.plot(raw_asac, color='blue', alpha=0.15)
        plt.plot(smooth_asac, color='blue', label='ASAC (Baseline)', linewidth=2)

        # Plot Hybrid
        plt.plot(raw_hybrid, color='orange', alpha=0.15)
        plt.plot(smooth_hybrid, color='orange', label='ASAC + Hybrid A*', linewidth=2)

        # Formatting in English as requested
        plt.title("Reward comparison of both models in training process", fontsize=16)
        plt.xlabel("Episodes", fontsize=12)
        plt.ylabel("Cumulative Reward", fontsize=12)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig("training_comparison_reward.png")
        plt.show()

    except FileNotFoundError:
        print("Error: monitor.csv files not found. Make sure you trained the models with the Monitor wrapper.")

if __name__ == "__main__":
    plot_training_comparison()