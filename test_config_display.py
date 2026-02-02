import sys
import os
sys.path.append(os.getcwd())

try:
    from runner import generate_config_report
    print("Generating Config Report...")
    print(generate_config_report())
    print("Success!")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
