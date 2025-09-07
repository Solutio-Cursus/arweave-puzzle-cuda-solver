import configparser
import itertools
import os
import sys
import subprocess
from datetime import datetime
import re

class ComboBruteManager:
    """
    Combines password generation and batch processing for the CUDA-based brute-force tool.
    Generates combinations "on-the-fly" and feeds them in batches to an external executable.
    """

    def __init__(self, config_file='config.ini'):
        """Loads configuration and initializes state."""
        self.config = configparser.ConfigParser()
        if not os.path.exists(config_file):
            print(f"Error: Configuration file '{config_file}' not found.")
            sys.exit(1)
        self.config.read(config_file)
        
        self._load_config()
        
        self.total_combinations = 0
        self.parts_list = []

    def _load_config(self):
        """Loads settings from the config file into instance variables."""
        self.keys_input_file = self.config.get('main', 'keys_input_file')
        self.cuda_executable = self.config.get('main', 'cuda_executable')
        cuda_args_str = self.config.get('main', 'cuda_args', fallback='')
        self.cuda_args = cuda_args_str.split() if cuda_args_str else []
        self.match_marker = self.config.get('main', 'match_marker', fallback='>>> MATCH FOUND! <<<')

        self.batch_size = self.config.getint('processing', 'batch_size')
        self.temp_batch_file = self.config.get('processing', 'temp_batch_file')

        self.log_file = self.config.get('logging', 'log_file')
        self.found_password_file = self.config.get('logging', 'found_password_file')
        self.save_tested_passwords = self.config.getboolean('logging', 'save_tested_passwords')
        self.tested_passwords_file = self.config.get('logging', 'tested_passwords_file')

        self.full_combo_chars = self.config.get('full_combo', 'characters')

    def _parse_input_file(self):
        """Parses the input file, handling standard parts and "Full-Combo" syntax."""
        if not os.path.exists(self.keys_input_file):
            print(f"Error: The input file '{self.keys_input_file}' was not found.")
            sys.exit(1)

        with open(self.keys_input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                elements = [elem.strip() for elem in line.strip().split(',') if elem.strip()]
                if not elements:
                    continue
                
                processed_elements = []
                for elem in elements:
                    match = re.match(r'^(.*?)("Full-Combo-(\d+)")$', elem)
                    if match:
                        prefix = match.group(1)
                        combo_len = int(match.group(3))
                        processed_elements.append(('FULL_COMBO', prefix, combo_len))
                    else:
                        processed_elements.append(elem)
                self.parts_list.append(processed_elements)

    def _calculate_total_combinations(self):
        """Calculates the total number of possible combinations."""
        total = 1
        print("Number of elements per line:")
        for i, parts in enumerate(self.parts_list, 1):
            count = 0
            has_full_combo = False
            for part in parts:
                if isinstance(part, tuple) and part[0] == 'FULL_COMBO':
                    _, _, combo_len = part
                    count += len(self.full_combo_chars) ** combo_len
                    has_full_combo = True
                else:
                    count += 1
            
            line_info = f"Line {i}: {count:,} elements"
            if has_full_combo:
                line_info += " (contains Full-Combo pattern)"
            print(line_info)
            total *= count
        
        self.total_combinations = total
        return total

    def _get_combination_generator(self):
        """Creates a generator that yields all possible combinations."""
        line_generators = []
        for parts in self.parts_list:
            def create_line_gen(items):
                def line_gen():
                    for item in items:
                        if isinstance(item, tuple) and item[0] == 'FULL_COMBO':
                            _, prefix, length = item
                            for combo in itertools.product(self.full_combo_chars, repeat=length):
                                yield prefix + ''.join(combo)
                        else:
                            yield item
                return line_gen()
            line_generators.append(create_line_gen(parts))

        return (''.join(p) for p in itertools.product(*line_generators))

    def run_cuda_process(self):
        """
        Executes the CUDA program and parses its output for the specific match format.
        Returns (True, found_password) on success, otherwise (False, "").
        """
        print(f"Starting CUDA process for '{self.temp_batch_file}'...")
        
        try:
            # Ensure the `message.b64` file required by the CUDA tool exists.
            if not os.path.exists('message.b64'):
                print("\nWarning: 'message.b64' not found. The CUDA tool may fail.")

            cuda_process = subprocess.Popen(
                [self.cuda_executable] + self.cuda_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            found_hit = False
            match_password = "Could not parse from output line"
            output_lines = []

            # Read output line by line in real-time
            for line in iter(cuda_process.stdout.readline, ''):
                line = line.strip()
                sys.stdout.write(line + '\n')
                sys.stdout.flush()
                output_lines.append(line)

                # Check if the current line contains the specific success marker
                if self.match_marker in line:
                    found_hit = True
                    # The password is on the *next* line. We'll find it after the loop.
            
            cuda_process.stdout.close()
            
            # If a hit was found, parse the collected output
            if found_hit:
                try:
                    # Find the index of the line with the marker
                    marker_index = -1
                    for i, l in enumerate(output_lines):
                        if self.match_marker in l:
                            marker_index = i
                            break
                    
                    # The password is on the next line, prefixed with "The correct password is: "
                    if marker_index != -1 and marker_index + 1 < len(output_lines):
                        next_line = output_lines[marker_index + 1]
                        if next_line.startswith("The correct password is: "):
                            match_password = next_line.replace("The correct password is: ", "").strip()
                except Exception as e:
                    print(f"Error parsing password after match: {e}")

            # Capture any stderr output
            stderr_output = cuda_process.stderr.read()
            if stderr_output:
                sys.stderr.write(stderr_output)
                sys.stderr.flush()

            cuda_process.wait()

            if cuda_process.returncode != 0 and not found_hit:
                print(f"\nCUDA process exited with error code {cuda_process.returncode}.")

            return found_hit, match_password

        except FileNotFoundError:
            print(f"\nError: Executable '{self.cuda_executable}' not found. Please check the path in config.ini.")
            return False, ""
        except Exception as e:
            print(f"\nAn error occurred while running the CUDA process: {e}")
            return False, ""
        finally:
            if os.path.exists(self.temp_batch_file):
                os.remove(self.temp_batch_file)

    def run(self):
        """Main function to control the batch generation and processing."""
        print(f"Key generator and batch manager started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self._parse_input_file()
        total_combos = self._calculate_total_combinations()

        print(f"\nTotal number of combinations to be generated: {total_combos:,}")
        
        try:
            response = input("Do you want to start the process? (y/n): ")
            if response.lower() not in ['y', 'yes']:
                print("Operation cancelled by user.")
                return
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.")
            return

        print("\n--- Starting Process ---")
        
        combination_generator = self._get_combination_generator()
        batch = []
        batch_number = 1
        processed_count = 0
        
        tested_pw_file = None
        if self.save_tested_passwords:
            tested_pw_file = open(self.tested_passwords_file, 'a', encoding='utf-8')

        try:
            for combo in combination_generator:
                batch.append(combo)
                processed_count += 1

                if len(batch) >= self.batch_size:
                    hit_found, match = self._process_batch(batch, batch_number, processed_count, tested_pw_file)
                    if hit_found:
                        self._handle_match_found(match, processed_count)
                        return
                    
                    batch = []
                    batch_number += 1

            if batch:
                hit_found, match = self._process_batch(batch, batch_number, processed_count, tested_pw_file)
                if hit_found:
                    self._handle_match_found(match, processed_count)
                    return
            
            print('\n\nAll batches have been processed. No match found in the entire set of combinations.')

        except (KeyboardInterrupt, EOFError):
             print("\n\nProcess interrupted by user. Stopping automation.")
        finally:
            if tested_pw_file:
                tested_pw_file.close()

    def _process_batch(self, batch, batch_number, processed_count, tested_pw_file):
        """Handles writing, running, and logging for a single batch."""
        batch_start_index = processed_count - len(batch) + 1
        last_password_in_batch = batch[-1]
        
        log_message = (
            f"[{datetime.now().isoformat()}] Starting Batch #{batch_number} "
            f"(Combinations {batch_start_index:,} - {processed_count:,})"
        )
        print(f"\n{log_message}")

        with open(self.temp_batch_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(batch))
        
        if self.save_tested_passwords and tested_pw_file:
            tested_pw_file.write('\n'.join(batch) + '\n')
            tested_pw_file.flush()

        hit_found, match = self.run_cuda_process()
        
        log_completion_message = (
            f"[{datetime.now().isoformat()}] Finished Batch #{batch_number}. "
            f"Last tested password: {last_password_in_batch} (Index: {processed_count:,})\n"
        )
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_completion_message)
        
        return hit_found, match

    def _handle_match_found(self, match_password, final_count):
        """Logs and saves the found password."""
        print('\n\n>>> MATCH FOUND! Stopping automation. <<<')
        print(f"Password: {match_password}")
        print(f"Found at combination index: {final_count:,}")
        
        log_message = (
            f"[{datetime.now().isoformat()}] MATCH FOUND!\n"
            f"Password: {match_password}\n"
            f"Found at global index: {final_count:,}\n"
        )
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message)
        
        with open(self.found_password_file, 'w', encoding='utf-8') as f:
            f.write(f"{match_password}\n")
        
        print(f"The found password has been saved to '{self.found_password_file}'")


if __name__ == "__main__":
    manager = ComboBruteManager()
    manager.run()