# Arweave Puzzle CUDA Brute-Forcer

This repository contains a high-performance, multi-GPU capable CUDA-based brute-force tool specifically designed to solve the Arweave Puzzles. It is engineered to overcome a unique challenge presented by the puzzles' non-standard AES encryption implementation.

## The Challenge: A Non-Standard 128-Byte AES Key

The Arweave puzzles utilize a JavaScript-based encryption setup where `CryptoJS.algo.AES.keySize` is set to `32`. Instead of interpreting this as a standard 32-byte (256-bit) key, CryptoJS erroneously creates a **128-byte (1024-bit)** key.

This "illegal" key size is not supported by most standard cryptographic libraries, which typically either throw an error or truncate the key to 32 bytes. Consequently, decryption can only be performed using the original, slow JavaScript implementation or a custom-built tool that replicates this specific behavior. This peculiarity, while likely unintentional, serves as an additional layer of protection against conventional brute-force attacks.

This project successfully replicates the entire cryptographic pipeline in CUDA, providing a significant speed-up over any JavaScript-based solution.

## Cryptographic Pipeline

The tool automates every step of the process, from a plaintext password to the final decryption check:

1.  **Iterative Hashing:** The plaintext password is first hashed **11,513 times** using **SHA-512**.
2.  **Key & IV Derivation (KDF):** The resulting SHA-512 hash is then fed into an OpenSSL-style `EVP_BytesToKey` function. This Key Derivation Function uses **10,000 iterations of MD5** along with the salt (extracted from bytes 8-16 of the message) to generate a **128-byte encryption key** and a **16-byte IV**.
3.  **AES Decryption:** The derived key and IV are used to decrypt the puzzle's message payload using AES-256-CBC. Since the key is 128 bytes, a custom implementation is used to handle the non-standard key size.
4.  **Validation:** The decryption is considered successful if the decrypted text contains a specific pattern (e.g., `"kty":"RSA"`), which is hardcoded into the kernel.

With this CUDA implementation, remarkable speeds can be achieved. The primary bottleneck is not the AES decryption but the ~21,000 hashing iterations (SHA-512 + MD5) required for each password candidate.

- **NVIDIA RTX 2080:** ~41,000 passwords/second
- **NVIDIA RTX 3090:** ~87,000 passwords/second (with optimized compile flags)

> **Disclaimer:** This tool is highly specialized for the Arweave puzzles. Its logic is tailored to the unique 128-byte key bug and is unlikely to work for other AES-encrypted messages. However, you are welcome to adapt any part of the source code for your own projects.

---

## Prerequisites

*   An NVIDIA GPU with compatible CUDA drivers installed.
*   The CUDA Toolkit (for compiling the source code).
*   **Python 3**: Required for the `combo_brute_manager.py` script.
*   **Windows Users**: You can run the Linux version using WSL 2 (WSL 1 is not supported) or use the provided `.exe` executable in Command Prompt.

---

## Repository Contents

*   **`combo_brute_manager.py`**: The recommended, all-in-one tool for "on-the-fly" key generation and batch processing. It supports both single and multi-GPU setups.
*   **`config.ini`**: The configuration file for the Python manager.
*   **`keys-input.txt`**: The input file where you define password structures and combinations.
*   **`combined_main.cu`**: The main source file containing the host code and the primary CUDA kernel.
*   **`crypto_kernels.cuh`**: Header file with all device-side cryptographic functions (SHA-512, MD5, AES).
*   **`Makefile`**: Used to compile the project on Linux/WSL.
*   **`message.b64`**: The encrypted message file for the puzzle.
*   **Other Folders (e.g., `Key-Generator`, `Debugging-Tools`)**: Contain useful scripts and information for generating inputs and understanding the JS bug.

---

## Usage Guide

First, compile the program from source using the `make` command. For best results, see the **Performance Tuning** section below.
```bash
# Clean previous builds and compile a fresh version
make clean && make
```

### Option 1: Python Manager (Recommended for Single & Multi-GPU)

The `combo_brute_manager.py` script is the most powerful way to use this tool. It generates password combinations on-the-fly based on your rules and feeds them to the CUDA executable in manageable batches. It automatically detects and utilizes all available GPUs for parallel processing.

**How it works:**
1.  **Configure:** Edit `config.ini` to set your `batch_size`. To disable multi-GPU processing, set `multi_gpu = false`.
2.  **Define Structure:** Edit `keys-input.txt` to define the parts that will be combined to create passwords. This file also supports a powerful `Full-Combo-X` syntax to generate all possible character combinations of a certain length.
3.  **Run:** Execute the Python script. It will calculate the total number of possible combinations, ask for confirmation, and then start the process, distributing the workload across all detected GPUs.

**Example Workflow:**

1.  **Set up your password structure in `keys-input.txt`:**
    ```
    head,body,tail
    -v1,-v2
    "Full-Combo-4" 
    ```
    This would generate combinations like `head-v1abcd`, `body-v2wxyz`, etc., where the last part is any 4-character string defined in `config.ini`.

2.  **Run the manager:**
    ```bash
    python3 combo_brute_manager.py
    ```

**Example Output (Multi-GPU):**
```
Detected 2 GPUs.
Total combinations to generate: 10,077,696
Do you want to start the process? (y/n): y

--- Starting Process ---

[2025-09-09T14:00:00.123Z] Starting Run #1 for 2 GPUs (Combinations 1 - 2,000,000)

[GPU-0] Starting CUDA process for 'input_batch_0.txt'...
[GPU-1] Starting CUDA process for 'input_batch_1.txt'...

--- Output from GPU-0 (Finished) ---
...
>>> MATCH FOUND! <<<
The correct password is: *48GCEErisUmberCastlePicasso
--------------------------------------------------
--- End of Output from GPU-0 ---

--- Output from GPU-1 (Finished) ---
...
--- End of Output from GPU-1 ---

>>> MATCH FOUND! Stopping automation. <<<
Password: *48GCEErisUmberCastlePicasso
The found password has been saved to 'FOUND-PW.txt'
```

### Option 2: Standalone CUDA Program (for a single, pre-generated list)

You can run the compiled `ar-brute` executable directly with a single password list. This is useful for testing or if you already have a complete file.

1.  Ensure `message.b64` is in the same directory.
2.  Execute the program and pass the path to your password file as an argument:
    *   **Linux / WSL:** `./ar-brute my_password_list.txt`
    *   **Windows:** `ar-brute.exe my_password_list.txt`

The program will process only the specified file.

---

## Performance Tuning

You can gain significant performance by tuning parameters for your specific GPU.

#### 1. Compilation Flags (Highly Recommended)

The `Makefile` does not specify a GPU architecture by default. For optimal performance, add the `-arch=sm_XX` flag to the `NVCC_FLAGS` in the `Makefile` to target your specific GPU architecture. This can lead to a substantial speed increase (e.g., an RTX 3090 improved from 69k to 87k pw/s).

Find your architecture here: [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus)

**Example for an RTX 3090 (Ampere architecture):**
```makefile
# In your Makefile
NVCC_FLAGS = -O3 -arch=sm_86
```

#### 2. Thread Count
The number of threads per block can be adjusted in `combined_main.cu` on line 271:
```c++
int threads_per_block = 128; // Default
```
Experiment with powers of 2 (e.g., `64`, `128`, `256`, `512`). A different value may yield better performance on your specific hardware. Remember to recompile (`make`) after changing the value.

---

### A Note on Puzzles (Search Pattern)

The kernel validates a successful decryption by searching for a hardcoded text pattern. This pattern is defined in `combined_main.cu` on line 119:

```c++
const char* pattern = "\"kty\":\"RSA\"";
```

This pattern works for puzzles where the decrypted content is an RSA key file (like Puzzles 3 and 5). For other puzzles with different content (like Puzzle 7, which contains `"version":3,"id"`), you will need to change this pattern and recompile the program accordingly.