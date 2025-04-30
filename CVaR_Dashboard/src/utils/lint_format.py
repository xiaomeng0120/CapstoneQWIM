import subprocess
import sys

def run_command(command):
    """Run a shell command and print its output."""
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode

def main():
    file_path = "src/dashboard/modules/portfolios_module.py"
    
    print(f"Step 1: Checking for issues in {file_path}")
    run_command(f"ruff check {file_path}")
    
    print(f"\nStep 2: Fixing automatically fixable issues in {file_path}")
    run_command(f"ruff check --fix {file_path}")
    
    print(f"\nStep 3: Formatting {file_path}")
    run_command(f"ruff format {file_path}")
    
    print(f"\nStep 4: Final check for remaining issues in {file_path}")
    run_command(f"ruff check {file_path}")
    
    print("\nLinting and formatting completed!")

if __name__ == "__main__":
    main()