import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "validation" / "results"
PLOTS_DIR = BASE_DIR / "validation" / "plots"
RAW_DATA_DIR = PLOTS_DIR / "raw_data"

# Ensure directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def load_json(filename):
    with open(RESULTS_DIR / filename, 'r') as f:
        return json.load(f)

def plot_correlation():
    data = load_json("similarity_correlation_test.json")
    results = data["results"]
    correlation = data["correlation"]
    
    actual_cos = [r["actual"] for r in results]
    hamming_sim = [r["hamming"] for r in results]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(actual_cos, hamming_sim, color='blue', alpha=0.7, label='Measurements')
    
    # Ideal line
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Ideal (y=x)')
    
    # Threshold lines
    plt.axvline(x=0.85, color='green', linestyle=':', alpha=0.5, label='Cache Threshold (Cosine 0.85)')
    plt.axhline(y=0.85, color='green', linestyle=':', alpha=0.5)
    
    plt.title(f"Cosine vs Hamming Similarity\nPearson r = {correlation:.4f}")
    plt.xlabel("Actual Cosine Similarity (Float)")
    plt.ylabel("Binary Hamming Similarity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    
    output_path = PLOTS_DIR / "correlation_float_vs_binary.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved correlation plot to {output_path}")
    plt.close()

def plot_latency_comparison():
    data = load_json("s1_latency_results_v3.json")
    
    encode_ms = data["encode_ms"]
    lookup_ms = data["lookup_us"] / 1000.0
    
    targets = {
        "Encode": 1.0,  # ms
        "Lookup": 1.0   # ms (kill trigger)
    }
    
    measured = {
        "Encode": encode_ms,
        "Lookup": lookup_ms
    }
    
    labels = list(measured.keys())
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(8, 6))
    
    rects1 = plt.bar(x - width/2, [measured[l] for l in labels], width, label='Measured', color=['#3498db', '#e74c3c'])
    rects2 = plt.bar(x + width/2, [targets[l] for l in labels], width, label='Target (Max)', color='gray', alpha=0.5)
    
    plt.ylabel('Latency (ms)')
    plt.title('Latency Metrics: Measured vs Target')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.2f} ms',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    
    output_path = PLOTS_DIR / "latency_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved latency plot to {output_path}")
    plt.close()

def plot_memory_usage():
    data = load_json("s1_latency_results_v3.json")
    memory_mb = data["memory_mb"]
    target_mb = 4.0
    
    plt.figure(figsize=(6, 6))
    bars = plt.bar(['100k Entries'], [memory_mb], width=0.4, color='#2ecc71', label='Actual')
    plt.axhline(y=target_mb, color='r', linestyle='--', label=f'Limit ({target_mb} MB)')
    
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage for 100k Entries')
    plt.ylim(0, 5)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    # Add label
    for rect in bars:
        height = rect.get_height()
        plt.annotate(f'{height:.2f} MB',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    output_path = PLOTS_DIR / "memory_usage.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved memory plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    print("Generating plots...")
    try:
        plot_correlation()
        plot_latency_comparison()
        plot_memory_usage()
        print("All plots generated successfully.")
    except Exception as e:
        print(f"Error generating plots: {e}")

