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
    Supports both single and multi-GPU setups.
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
        self.gpu_count = 1
        self.found_flag = False
        self.found_password = ""

    def _load_config(self):
        """Loads settings from the config file into instance variables."""
        self.keys_input_file = self.config.get('main', 'keys_input_file')
        self.cuda_executable = self.config.get('main', 'cuda_executable')
        cuda_args_str = self.config.get('main', 'cuda_args', fallback='')
        self.cuda_args = cuda_args_str.split() if cuda_args_str else []
        self.match_marker = self.config.get('main', 'match_marker', fallback='>>> MATCH FOUND! <<<')

        self.multi_gpu = self.config.getboolean('processing', 'multi_gpu', fallback=False)
        self.batch_size_per_gpu = self.config.getint('processing', 'batch_size')
        self.temp_batch_file_base = self.config.get('processing', 'temp_batch_file_base')

        self.log_file = self.config.get('logging', 'log_file')
        self.found_password_file = self.config.get('logging', 'found_password_file')
        self.save_tested_passwords = self.config.getboolean('logging', 'save_tested_passwords')
        self.tested_passwords_file = self.config.get('logging', 'tested_passwords_file')

        self.full_combo_chars = self.config.get('full_combo', 'characters')

    def get_gpu_count(self):
        """Detects the number of available NVIDIA GPUs using nvidia-smi."""
        if not self.multi_gpu:
            return 1
        try:
            # nvidia-smi -L lists all GPUs. We count the lines in the output.
            command = "nvidia-smi -L"
            output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.PIPE)
            gpu_count = len(output.strip().split('\n'))
            if gpu_count > 0:
                print(f"Detected {gpu_count} GPUs.")
                return gpu_count
            else:
                print("Warning: nvidia-smi found 0 GPUs. Defaulting to 1.")
                return 1
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: 'nvidia-smi' command not found or failed. Defaulting to 1 GPU.")
            print("Please ensure NVIDIA drivers are installed and 'nvidia-smi' is in your PATH.")
            return 1

    def _parse_input_file(self):
        # ... (Diese Funktion bleibt unverändert)
        if not os.path.exists(self.keys_input_file):
            print(f"Error: The input file '{self.keys_input_file}' was not found.")
            sys.exit(1)
        with open(self.keys_input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                elements = [elem.strip() for elem in line.strip().split(',') if elem.strip()]
                if not elements: continue
                processed_elements = []
                for elem in elements:
                    match = re.match(r'^(.*?)("Full-Combo-(\d+)")$', elem)
                    if match:
                        prefix, _, combo_len_str = match.groups()
                        processed_elements.append(('FULL_COMBO', prefix, int(combo_len_str)))
                    else:
                        processed_elements.append(elem)
                self.parts_list.append(processed_elements)

    def _calculate_total_combinations(self):
        # ... (Diese Funktion bleibt unverändert)
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
            if has_full_combo: line_info += " (contains Full-Combo pattern)"
            print(line_info)
            total *= count
        self.total_combinations = total
        return total

    def _get_combination_generator(self):
        # ... (Diese Funktion bleibt unverändert)
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

    def run_cuda_process_for_gpu(self, gpu_id, batch_file):
        """
        Executes the CUDA program on a specific GPU for a specific batch file.
        Returns the full output for later parsing.
        """
        print(f"[GPU-{gpu_id}] Starting CUDA process for '{batch_file}'...")
        
        # Set environment variable to select the GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        command = [self.cuda_executable] + self.cuda_args + [batch_file]
        
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env
            )
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"\n[GPU-{gpu_id}] Process exited with error code {process.returncode}.")
            
            return stdout, stderr

        except FileNotFoundError:
            print(f"\nError: Executable '{self.cuda_executable}' not found. Please check the path in config.ini.")
            return "", "Executable not found."
        except Exception as e:
            print(f"\n[GPU-{gpu_id}] An error occurred while running the CUDA process: {e}")
            return "", str(e)

    def run(self):
        """Main function to control batch generation and processing, with multi-GPU support."""
        print(f"Key generator and batch manager started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.gpu_count = self.get_gpu_count()
        self._parse_input_file()
        total_combos = self._calculate_total_combinations()

        print(f"\nTotal combinations to generate: {total_combos:,}")
        print(f"Batch size per GPU: {self.batch_size_per_gpu:,}")
        
        try:
            response = input("Do you want to start the process? (y/n): ")
            if response.lower() not in ['y', 'yes']:
                print("Operation cancelled by user."); return
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user."); return

        print("\n--- Starting Process ---")
        
        combination_generator = self._get_combination_generator()
        mega_batch_size = self.batch_size_per_gpu * self.gpu_count
        batch_number = 1
        processed_count = 0
        
        tested_pw_file = None
        if self.save_tested_passwords:
            tested_pw_file = open(self.tested_passwords_file, 'a', encoding='utf-8')

        try:
            while not self.found_flag:
                # 1. Create a "mega-batch" for all GPUs
                mega_batch = list(itertools.islice(combination_generator, mega_batch_size))
                if not mega_batch:
                    print('\n\nAll combinations have been processed.')
                    break

                # 2. Split the mega-batch into sub-batches for each GPU
                sub_batches = []
                for i in range(self.gpu_count):
                    start = i * self.batch_size_per_gpu
                    end = start + self.batch_size_per_gpu
                    sub_batch = mega_batch[start:end]
                    if sub_batch:
                        sub_batches.append((i, sub_batch))
                
                # 3. Process sub-batches in parallel (conceptually)
                # Here we do it sequentially to manage output, but Popen is non-blocking
                # A more advanced version would use multiprocessing.Pool
                
                batch_start_index = processed_count + 1
                processed_in_run = len(mega_batch)
                processed_count += processed_in_run
                last_password_in_mega_batch = mega_batch[-1]

                log_message = (
                    f"[{datetime.now().isoformat()}] Starting Run #{batch_number} for {self.gpu_count} GPUs "
                    f"(Combinations {batch_start_index:,} - {processed_count:,})"
                )
                print(f"\n{log_message}")

                # Write batch files
                batch_files = []
                for gpu_id, sub_batch in sub_batches:
                    batch_file = f"{self.temp_batch_file_base}{gpu_id}.txt"
                    batch_files.append(batch_file)
                    with open(batch_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(sub_batch))
                    if self.save_tested_passwords and tested_pw_file:
                        tested_pw_file.write('\n'.join(sub_batch) + '\n')
                if self.save_tested_passwords: tested_pw_file.flush()

                # Launch processes
                # For simplicity, we wait for all to finish. A true parallel approach is more complex.
                # This is a good starting point.
                outputs = [self.run_cuda_process_for_gpu(gpu_id, f"{self.temp_batch_file_base}{gpu_id}.txt") for gpu_id, _ in sub_batches]

                # Clean up files
                for bf in batch_files:
                    if os.path.exists(bf): os.remove(bf)

                # 4. Check results from all GPUs
                for gpu_id, (stdout, stderr) in enumerate(outputs):
                    print(f"--- Output from GPU-{gpu_id} ---")
                    sys.stdout.write(stdout)
                    sys.stderr.write(stderr)
                    print(f"--- End of Output from GPU-{gpu_id} ---")

                    if self.match_marker in stdout:
                        self.found_flag = True
                        try:
                            # Extract password
                            match_line_index = stdout.find("The correct password is: ")
                            if match_line_index != -1:
                                start = match_line_index + len("The correct password is: ")
                                end = stdout.find('\n', start)
                                self.found_password = stdout[start:end].strip()
                        except Exception:
                            self.found_password = "Could not parse from output"
                        
                        # Since a match was found, stop further processing
                        break

                log_completion_message = (
                    f"[{datetime.now().isoformat()}] Finished Run #{batch_number}. "
                    f"Last tested password in run: {last_password_in_mega_batch} (Global Index: {processed_count:,})\n"
                )
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_completion_message)
                
                batch_number += 1

                if self.found_flag:
                    self._handle_match_found(self.found_password, processed_count)
                    break
            
        except (KeyboardInterrupt, EOFError):
             print("\n\nProcess interrupted by user. Stopping automation.")
        finally:
            if tested_pw_file:
                tested_pw_file.close()

    def _handle_match_found(self, match_password, final_count_estimate):
        """Logs and saves the found password."""
        print('\n\n>>> MATCH FOUND! Stopping automation. <<<')
        print(f"Password: {match_password}")
        print(f"Found within batch ending at global index: {final_count_estimate:,}")
        
        log_message = (
            f"[{datetime.now().isoformat()}] MATCH FOUND!\n"
            f"Password: {match_password}\n"
            f"Found within batch ending at global index: {final_count_estimate:,}\n"
        )
        with open(self.log_file, 'a', encoding='utf-8') as f: f.write(log_message)
        with open(self.found_password_file, 'w', encoding='utf-8') as f: f.write(f"{match_password}\n")
        
        print(f"The found password has been saved to '{self.found_password_file}'")

if __name__ == "__main__":
    manager = ComboBruteManager()
    manager.run()