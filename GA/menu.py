"""
Simple Menu for Genetic Algorithm
"""

import sys
import time
from tabulate import tabulate
from benchmark_functions import benchmark_functions
from ga_algorithm import run_multiple_experiments, run_all_functions_parallel, calculate_statistics
from visualization import plot_individual_function, plot_all_functions_combined
from config import NUM_RUNS

def format_number(value):
    """Convert number to decimal format with sufficient precision"""
    if value == 0:
        return "0.00000000"
    elif abs(value) < 1e-10:
        # Very small numbers - show up to 20 decimal places
        return f"{value:.20f}".rstrip('0').rstrip('.')
    elif abs(value) < 1e-6:
        # Small numbers - show up to 16 decimal places
        return f"{value:.16f}".rstrip('0').rstrip('.')
    elif abs(value) < 0.001:
        # Show up to 12 decimal places
        return f"{value:.12f}".rstrip('0').rstrip('.')
    elif abs(value) < 1:
        return f"{value:.8f}".rstrip('0').rstrip('.')
    elif abs(value) < 100:
        return f"{value:.6f}".rstrip('0').rstrip('.')
    else:
        return f"{value:.4f}".rstrip('0').rstrip('.')

def print_menu():
    print("\n" + "="*60)
    print(" GENETIC ALGORITHM BENCHMARK ".center(60))
    print("="*60)
    print("\n1. Run single function")
    print("2. Run all functions (parallel)")
    print("3. Exit\n")

def show_functions():
    """Display list of available functions"""
    print("\nAvailable Functions:")
    print("-" * 40)
    functions = list(benchmark_functions.keys())
    for i, name in enumerate(functions, 1):
        print(f"{i:2d}. {name}")
    print("-" * 40)
    return functions

def show_single_result(func_name, results, time_taken):
    """Show results for a single function"""
    stats = calculate_statistics(results)
    
    print("\n" + "="*60)
    print(f" {func_name.upper()} ".center(60))
    print("="*60)
    
    table_data = [
        ["Runs", len(results)],
        ["Min", format_number(stats['min'])],
        ["Mean", format_number(stats['mean'])],
        ["Median", format_number(stats['median'])],
        ["Std Dev", format_number(stats['std'])],
        ["Max", format_number(stats['max'])],
        ["Time", f"{time_taken:.2f}s"]
    ]
    
    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))
    print("="*60 + "\n")
    return stats

def show_all_results(all_results, time_taken):
    """Show results for all functions"""
    print("\n" + "="*100)
    print(" BENCHMARK RESULTS - ALL FUNCTIONS ".center(100))
    print("="*100)
    
    # Calculate stats and prepare data
    results_data = []
    for func_name, results in all_results.items():
        stats = calculate_statistics(results)
        results_data.append({
            'name': func_name,
            'min': stats['min'],
            'mean': stats['mean'],
            'median': stats['median'],
            'std': stats['std']
        })
    
    # Sort by mean value (best to worst)
    results_data.sort(key=lambda x: x['mean'])
    
    # Create table with rankings
    table_data = []
    for i, data in enumerate(results_data, 1):
        # Add emoji for top 3
        rank = "1." if i == 1 else "2." if i == 2 else "3." if i == 3 else f"{i}."
        
        table_data.append([
            rank,
            data['name'],
            format_number(data['min']),
            format_number(data['mean']),
            format_number(data['median']),
            format_number(data['std'])
        ])
    
    print(tabulate(
        table_data,
        headers=["Rank", "Function", "Min", "Mean", "Median", "Std Dev"],
        tablefmt="grid",
        disable_numparse=True
    ))
    
    print(f"\n{'='*100}")
    print(f"⏱️  Execution Time: {time_taken:.2f}s")
    
    # Check if target met
    from config import MAX_EXECUTION_TIME
    if time_taken <= MAX_EXECUTION_TIME:
        print(f"✓ Target met: {time_taken:.2f}s ≤ {MAX_EXECUTION_TIME}s")
    else:
        print(f"⚠️  Time: {time_taken:.2f}s (target was {MAX_EXECUTION_TIME}s)")
    
    print(f"📊 Runs per function: {NUM_RUNS}")
    print(f"🎯 Lower values = Better performance")
    print(f"{'='*100}\n")

def run_single_function():
    """Run genetic algorithm on one function"""
    functions = show_functions()
    
    # Get user choice
    try:
        choice = int(input("\nSelect function number (1-15): "))
        if choice < 1 or choice > len(functions):
            print("❌ Invalid choice! Please select 1-15")
            return
    except ValueError:
        print("❌ Please enter a valid number!")
        return
    
    func_name = functions[choice - 1]
    
    # Run experiment
    print(f"\n🚀 Running {func_name}...")
    print(f"   Number of runs: {NUM_RUNS}")
    print()
    
    start_time = time.time()
    results, history = run_multiple_experiments(func_name, NUM_RUNS)
    execution_time = time.time() - start_time
    
    # Show results
    stats = show_single_result(func_name, results, execution_time)
    
    # Generate plots
    print("📊 Generating plots...")
    plot_individual_function(func_name, results, history, stats)
    print("✓ Plots saved successfully!\n")

def run_all_functions():
    """Run genetic algorithm on all functions in parallel"""
    print("\n🚀 Starting parallel execution...")
    print("   Running all 15 functions simultaneously")
    print(f"   Each function: {NUM_RUNS} runs")
    print("   Please wait...\n")
    
    # Run all functions
    all_results, plot_data, execution_time = run_all_functions_parallel(num_runs=NUM_RUNS)
    
    # Show results
    show_all_results(all_results, execution_time)
    
    # Generate plots
    print("📊 Generating comparison plots...")
    plot_all_functions_combined(all_results, plot_data)
    print("✓ Plots saved successfully!\n")

def main():
    """Main program loop"""
    while True:
        print_menu()
        
        try:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == '1':
                run_single_function()
            elif choice == '2':
                run_all_functions()
            elif choice == '3':
                print("\n👋 Goodbye!\n")
                sys.exit(0)
            else:
                print("\n❌ Invalid choice! Please enter 1, 2, or 3.\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error occurred: {e}\n")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()