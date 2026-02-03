import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def parse_detailed_output(filename):
    """
    Parses the DETAILED RESULTS tables from output.txt.
    Returns: {'PSO': {'Sphere': 1.5e-53, ...}, 'GA': ...}
    """
    if not os.path.exists(filename):
        print(f"Error: {filename} not found")
        sys.exit(1)

    data = {}
    with open(filename, 'r') as f:
        lines = f.readlines()

    current_algo = None
    parsing_table = False
    
    # We look for lines like "Calculate PSO DETAILED RESULTS" or just "PSO DETAILED RESULTS"
    
    for line in lines:
        stripped = line.strip()
        
        # Detect Header
        if "DETAILED RESULTS" in stripped and "=" not in stripped:
            parts = stripped.split()
            if len(parts) >= 1:
                # e.g. "PSO DETAILED RESULTS" -> algo="PSO"
                current_algo = parts[0]
                data[current_algo] = {}
                parsing_table = False
            continue
            
        # Detect Table Start
        # | Rank | Function | Best f | ...
        if "Rank" in stripped and "Function" in stripped and "Best f" in stripped:
            parsing_table = True
            continue
            
        # Parse Table Row
        if parsing_table and stripped.startswith("|"):
            # | 1 | Styblinski_Tang | -195.83 | ...
            cols = [c.strip() for c in stripped.split('|')]
            # cols[0] is empty, cols[1] is Rank, cols[2] is Function, cols[3] is Best f
            
            if len(cols) < 5:
                continue
                
            if "---" in cols[2] or cols[2] == "Function":
                continue
                
            func_name = cols[2]
            best_f_str = cols[3]
            
            try:
                val = float(best_f_str)
                data[current_algo][func_name] = val
            except ValueError:
                continue
                
    return data

def plot_4_subplots(data, output_file="preview.jpeg"):
    if not data:
        print("No data parsed from output.txt. Please check the file format.")
        return

    # Algorithm order
    preferred_order = ["PSO", "GA", "Tabu", "SA"]
    
    # Filter algos that are actually in data
    algos_to_plot = [a for a in preferred_order if a in data]
    # Add any others found
    for a in data.keys():
        if a not in algos_to_plot:
            algos_to_plot.append(a)
    
    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()
    
    # Colors for each plot
    colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#d62728'] # Blue, Orange, Purple, Red
    
    plt.style.use('default') 
    
    print(f"Plotting for algorithms: {algos_to_plot}")
    
    for i, ax in enumerate(axes):
        if i >= len(algos_to_plot):
            ax.set_visible(False)
            continue
            
        algo_name = algos_to_plot[i]
        algo_data = data[algo_name]
        
        # Sort functions by name for consistent x-axis across plots
        sorted_funcs = sorted(list(algo_data.keys()))
        values = [algo_data[Fn] for Fn in sorted_funcs]
        
        x = np.arange(len(sorted_funcs))
        
        # Bar Plot of Best Fitness
        ax.bar(x, values, color=colors[i % len(colors)], alpha=0.7, edgecolor='black')
        
        ax.set_title(f"{algo_name} Analysis", fontsize=14, fontweight='bold')
        ax.set_ylabel("Best Fitness (SymLog)", fontsize=10)
        
        # X-axis
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_funcs, rotation=45, ha='right', fontsize=9)
        
        # Use symlog to handle 0 and negative values nicely
        ax.set_yscale('symlog', linthresh=1e-10)
        
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Graph saved to {output_file}")

if __name__ == "__main__":
    output_file = "preview.jpeg"
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
        
    print(f"Reading from output.txt...")
    data = parse_detailed_output("output.txt")
    plot_4_subplots(data, output_file)
