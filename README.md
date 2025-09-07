# Arweave Puzzle CUDA Brute-Forcer

This repository contains a high-performance CUDA-based brute-force tool specifically designed to solve the Arweave Puzzles. It is engineered to overcome a unique challenge presented by the puzzles' non-standard AES encryption implementation.

## The Challenge: A Non-Standard 128-Byte AES Key

The Arweave puzzles utilize a JavaScript-based encryption setup where `CryptoJS.algo.AES.keySize` is set to `32`. Instead of interpreting this as a standard 32-byte (256-bit) key, CryptoJS erroneously creates a **128-byte (1024-bit)** key.

This "illegal" key size is not supported by most standard cryptographic libraries, which typically either throw an error or truncate the key to 32 bytes. Consequently, decryption can only be performed using the original, slow JavaScript implementation or a custom-built tool that replicates this specific behavior. This peculiarity, while likely unintentional, serves as an additional layer of protection against conventional brute-force attacks.

This project successfully replicates the entire cryptographic pipeline in CUDA, providing a significant speed-up over any JavaScript-based solution.

## Cryptographic Pipeline

The tool automates every step of the process, from a plaintext password to the final decryption check:

1.  **Iterative Hashing:** The plaintext password is first hashed **11,513 times** using **SHA-512**.
2.  **Key & IV Derivation (KDF):** The resulting SHA-512 hash is then fed into an OpenSSL-style `EVP_BytesToKey` function. This Key Derivation Function uses **10,000 iterations of MD5** along with the salt (extracted from bytes 8-16 of the message) to generate a **128-byte encryption key** and a **16-byte IV**.
3.  **AES Decryption:** The derived key and IV are used to decrypt the puzzle's message payload using AES-256-CBC. Since the key is 128 bytes, a custom implementation based on a modified Tiny AES library is used to handle the non-standard key size.
4.  **Validation:** The decryption is considered successful if the decrypted text contains a specific pattern (e.g., `"kty":"RSA"`), which is hardcoded into the kernel.

With this CUDA implementation, an NVIDIA RTX 2080 achieves approximately **12,000 passwords per second**. The primary bottleneck is not the AES decryption but the ~21,000 hashing iterations (SHA-512 + MD5) required for each password candidate.

> **Disclaimer:** This tool is highly specialized for the Arweave puzzles. Its logic is tailored to the unique 128-byte key bug and is unlikely to work for other AES-encrypted messages. However, you are welcome to adapt any part of the source code for your own projects.

---

## Prerequisites

*   An NVIDIA GPU with compatible CUDA drivers installed.
*   The CUDA Toolkit (for compiling the source code).
*   **Python 3**: Required for the recommended `combo_brute_manager.py` script.
*   **Node.js**: (Optional) Required only if you want to use the legacy `run_batch.js` script.
*   **Windows Users**: You can run the Linux version using WSL 2 (WSL 1 is not supported) or use the provided `.exe` executable in Command Prompt.

---

## Repository Contents

The project is structured to be easy to use. Pre-compiled binaries for both Linux (`ar-brute`) and Windows (`ar-brute.exe`) are included in the respective puzzle folders.

*   **`combo_brute_manager.py`**: The new, recommended, all-in-one tool for "on-the-fly" key generation and batch processing. It combines the functionality of the old key generator and batch runner.
*   **`config.ini`**: The configuration file for the Python manager.
*   **`keys-input.txt`**: The input file for the Python combination generator.
*   **`combined_main.cu`**: The main source file containing the host code and the primary CUDA kernel.
*   **`crypto_kernels.cuh`**: Header file with all device-side cryptographic functions (SHA-512, MD5, AES).
*   **`Makefile`**: Used to compile the project on Linux/WSL.
*   **`message.b64`**: The encrypted message file for the puzzle.
*   **`run_batch.js`**: (Legacy) A Node.js helper script for processing very large, pre-generated password lists.
*   **Other Folders (e.g., `Key-Generator`, `Debugging-Tools`)**: Contain useful scripts and information for generating inputs and understanding the JS bug. Please see the README files in those directories for more details.

---

## Usage Guide

You can compile the program from source using the `make` command:
```bash
# Clean previous builds and compile a fresh version
make clean && make
```

### Option 1: Using the Python Manager (Recommended)

The `combo_brute_manager.py` script is the most powerful and flexible way to use this tool. It generates password combinations on-the-fly based on your input rules and feeds them directly to the CUDA executable in manageable batches. This eliminates the need for massive, pre-generated password files.

**How it works:**
1.  **Configure:** Edit `config.ini` to set your desired `batch_size`, `cuda_executable` path, and other settings.
2.  **Define Structure:** Edit `keys-input.txt` to define the parts that will be combined to create passwords. This file also supports a powerful `Full-Combo-X` syntax to generate all possible character combinations of a certain length.
3.  **Run:** Execute the Python script. It will calculate the total number of possible combinations, ask for confirmation, and then start the process.

**Example Workflow:**

1.  **Set up your password structure in `keys-input.txt`:**
    ```
    head,body,tail
    -v1,-v2
    "Full-Combo-4" 
    ```
    This would generate combinations like `head-v1abcd`, `body-v2wxyz`, etc., where the last part is any 4-character string defined in `config.ini`.

2.  **(Optional) Adjust `config.ini`:**
    *   Change `batch_size` for your GPU's memory.
    *   Modify the `characters` under `[full_combo]` to include special characters or numbers.

3.  **Run the manager:**
    ```bash
    python3 combo_brute_manager.py
    ```

**Example Output:**
```
Key generator and batch manager started - 2025-09-07 12:00:00
Number of elements per line:
Line 1: 3 elements
Line 2: 2 elements
Line 3: 1,679,616 elements (contains Full-Combo pattern)

Total number of combinations to be generated: 10,077,696
Do you want to start the process? (y/n): y

--- Starting Process ---

[2025-09-07T12:00:10.500Z] Starting Batch #1 (Combinations 1 - 1,000,000)

Starting CUDA process for input_batch.txt...
...
>>> MATCH FOUND! <<<
The correct password is: *48GCEErisUmberCastlePicasso
--------------------------------------------------

>>> MATCH FOUND! Stopping automation. <<<
Password: *48GCEErisUmberCastlePicasso
The found password has been saved to 'FOUND-PW.txt'
```

### Option 2: Standalone CUDA Program (for pre-generated lists)

The `ar-brute` executable can be run directly. It will automatically search for and load all password files matching the pattern `input_*.txt` in the same directory.

1.  Place your password list in a file named `input_1.txt`, `input_abc.txt`, etc.
2.  Ensure the `message.b64` file is in the same directory.
3.  Execute the program:
    *   **Linux / WSL:** `./ar-brute`
    *   **Windows:** `ar-brute.exe`

> **Warning**
> The program is designed to load **every** file that matches the `input_*.txt` pattern. To prevent it from processing multiple lists at once, make sure that **only one** input file with this naming convention exists in the directory when you run the program.



### Option 3: Using `run_batch.js` (Legacy)

The `run_batch.js` script is a legacy option for processing a single, massive, pre-generated password file by splitting it into chunks.

1.  Name your master password file `keys.txt`.
2.  Run the script using Node.js:
    ```bash
    node run_batch.js
    ```
3.  To resume processing from a specific line, pass the line number as an argument.
    ```bash
    # Skip the first 10,000,000 lines and start from there
    node run_batch.js 10000000
    ```

---

## Performance Tuning

You can potentially gain more performance by tuning the following parameters for your specific GPU:

#### 1. Thread Count
The number of threads per block can be adjusted in `combined_main.cu` on line 271:
```c++
int threads_per_block = 64;
```
Experiment with values like `32`, `64`, `128`, `256`, `512`, or `1024`. After changing the value, recompile the program.

#### 2. Compilation Flags
The provided `Makefile` does not specify a GPU architecture by default. For optimal performance, you can manually add the `-arch=sm_XX` flag to the `NVCC_FLAGS` in the `Makefile` to target your specific GPU architecture (e.g., `-arch=sm_75` for Turing, `-arch=sm_86` for Ampere).

---

### A Note on Puzzles (Search Pattern)

The kernel validates a successful decryption by searching for a hardcoded text pattern. This pattern is defined in `combined_main.cu` on line 119:

```c++
const char* pattern = "\"kty\":\"RSA\"";
```

This pattern works for puzzles where the decrypted content is an RSA key file (like Puzzles 3 and 5). For other puzzles with different content (like Puzzle 7, which contains `"version":3,"id"`), you will need to change this pattern and recompile the program accordingly.