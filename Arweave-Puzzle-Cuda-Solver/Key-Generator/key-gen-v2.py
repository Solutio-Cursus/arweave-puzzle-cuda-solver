import itertools
import os
from datetime import datetime
import sys

def count_combinations(lines):
    """Calculates the total number of possible combinations."""
    total = 1
    line_counts = []
    for line_elements in lines:
        count = len(line_elements)
        line_counts.append(count)
        total *= count
    
    print("Number of elements per line:")
    for i, count in enumerate(line_counts, 1):
        print(f"Line {i}: {count} elements")
    
    return total

def format_number(num):
    """Formats large numbers readably with thousand separators."""
    return f"{num:,}".replace(",", ".")

def generate_combinations(input_file, output_file):
    """Generates all combinations and writes them to the output file."""
    try:
        # Read all lines and split by commas
        lines = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                # Split by commas and remove empty elements
                elements = [elem.strip() for elem in line.strip().split(',')]
                # Filter out empty elements
                elements = [elem for elem in elements if elem]
                
                if not elements:
                    print(f"Warning: Line {i} contains no valid elements after filtering.")
                    print("Please ensure that each line contains at least one valid element.")
                    return
                
                lines.append(elements)
                
                # Show details about special characters
                special_elements = [elem for elem in elements if any(c in elem for c in '-/\\.,+*?!@#$%^&')]
                if special_elements:
                    print(f"Note: Line {i} contains elements with special characters: {', '.join(special_elements)}")
        
        # Calculate the total number of combinations
        total_combinations = count_combinations(lines)
        print(f"\nTotal number of combinations to be generated: {format_number(total_combinations)}")
        
        # Calculate the approximate file size
        avg_length = sum(len(''.join(combo)) for combo in 
                        [list(map(lambda x: x[0], lines))])  # Use the first element of each line
        estimated_size_bytes = total_combinations * (avg_length + 1)  # +1 for newline character
        estimated_size_gb = estimated_size_bytes / (1024**3)
        print(f"Estimated output file size: {estimated_size_gb:.2f} GB")
        
        # Get confirmation if file size is large
        if estimated_size_gb > 1:
            response = input(f"\nThe output file will be approx. {estimated_size_gb:.2f} GB. Continue? (y/n): ")
            if response.lower() not in ['j', 'ja', 'y', 'yes']:
                print("Operation cancelled by user.")
                return
        
        print(f"\nStarting generation into file '{output_file}'...")
        start_time = datetime.now()
        
        # Use itertools.product to generate all combinations
        with open(output_file, 'w', encoding='utf-8') as f:
            # Status updates for large files
            update_interval = max(1, total_combinations // 100)  # 100 updates during the process
            combinations_generated = 0
            
            for combination in itertools.product(*lines):
                # Join the elements without a separator
                f.write(''.join(combination) + '\n')
                combinations_generated += 1
                
                # Status update
                if combinations_generated % update_interval == 0:
                    percent_complete = (combinations_generated / total_combinations) * 100
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    
                    # Estimate remaining time
                    if elapsed_time > 0:
                        combinations_per_second = combinations_generated / elapsed_time
                        remaining_combinations = total_combinations - combinations_generated
                        estimated_remaining_seconds = remaining_combinations / combinations_per_second
                        remaining_minutes = estimated_remaining_seconds // 60
                        remaining_seconds = estimated_remaining_seconds % 60
                        
                        sys.stdout.write(f"\rProgress: {percent_complete:.1f}% ({format_number(combinations_generated)} of {format_number(total_combinations)}) "
                                        f"- Estimated time remaining: {int(remaining_minutes)} min {int(remaining_seconds)} sec")
                    else:
                        sys.stdout.write(f"\rProgress: {percent_complete:.1f}% ({format_number(combinations_generated)} of {format_number(total_combinations)})")
                    sys.stdout.flush()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        combinations_per_second = combinations_generated / duration if duration > 0 else 0
        
        print(f"\n\nSuccessfully generated {format_number(combinations_generated)} combinations.")
        print(f"Duration: {duration:.1f} seconds ({format_number(int(combinations_per_second))} combinations/second)")
        print(f"Combinations were written to '{output_file}'")
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except PermissionError:
        print(f"Error: No permission to write to '{output_file}'.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    input_file = "keys-input.txt"
    output_file = "keys-output.txt"
    
    print(f"Key generator started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    generate_combinations(input_file, output_file)