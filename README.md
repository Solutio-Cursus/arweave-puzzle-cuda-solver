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
*   **Node.js**: Required for the `run_batch.js` script to manage large password lists.
*   **Windows Users**: You can run the Linux version using WSL 2 (WSL 1 is not supported) or use the provided `.exe` executable in Command Prompt.

---

## Repository Contents

The project is structured to be easy to use. Pre-compiled binaries for both Linux (`ar-brute`) and Windows (`ar-brute.exe`) are included in the respective puzzle folders.

*   **`combined_main.cu`**: The main source file containing the host code and the primary CUDA kernel.
*   **`crypto_kernels.cuh`**: Header file with all device-side cryptographic functions (SHA-512, MD5, AES).
*   **`Makefile`**: Used to compile the project on Linux/WSL.
*   **`message.b64`**: The encrypted message file for the puzzle.
*   **`run_batch.js`**: A Node.js helper script for processing very large password lists.
*   **Other Folders (e.g., `Key-Generator`, `Debugging-Tools`)**: Contain useful scripts and information for generating inputs and understanding the JS bug. Please see the README files in those directories for more details.

---

## Performance Tuning

You can potentially gain more performance by tuning the following parameters for your specific GPU:

#### 1. Thread Count
The number of threads per block can be adjusted in `combined_main.cu` on line 271:
```c++
int threads_per_block = 64;
```
Lower thread counts (32, 64) have shown better performance on some architectures (like the RTX 2080) due to the nature of the workload. Experiment with values like `32`, `64`, `128`, `256`, `512`, or `1024`. After changing the value, recompile the program.

**Note:** If you see an unrealistically high throughput (e.g., billions of Pw/s), the thread count is likely too high, and the kernel is failing silently. Always test with a known password to ensure your settings are valid.

#### 2. Compilation Flags
The provided `Makefile` does not specify a GPU architecture by default, allowing `nvcc` to select a suitable one automatically for broad compatibility.

For optimal performance, you can manually add the `-arch=sm_XX` flag to the `NVCC_FLAGS` in the `Makefile` to target your specific GPU architecture (e.g., `-arch=sm_61` for Pascal, `-arch=sm_75` for Turing, `-arch=sm_86` for Ampere). This can result in a significant speed-up.

**Note:** Using an incorrect architecture flag can lead to silent failures and prevent the correct password from being found. Always test with a known password after making changes.

---

## Usage Guide

You can compile the program from source using the `make` command:
```bash
# Clean previous builds and compile a fresh version
make clean && make
```

#### Option 1: Standalone CUDA Program

The `ar-brute` executable can be run directly. It will automatically search for password files matching the pattern `input_*.txt` in the same directory.

1.  Place your password list in a file named `input_1.txt`, `input_abc.txt`, etc.
2.  Ensure the `message.b64` file is in the same directory.
3.  Execute the program:
    *   **Linux / WSL:** `./ar-brute`
    *   **Windows:** `ar-brute.exe`

> **Warning**
> The program is designed to load **every** file that matches the `input_*.txt` pattern. To prevent it from processing multiple lists at once, make sure that **only one** input file with this naming convention exists in the directory when you run the program.

For very large lists (e.g., >1GB or millions of lines), it is highly recommended to use the batch script to avoid memory issues.

**Example Output:**
```
Loading passwords from:
 - input_1.txt
100000 passwords loaded.

Message 'message.b64' loaded.
 - Salt (HEX): 84dbff786c983a3f
 - Ciphertext length: 3168 Bytes

Starting...

--------------------------------------------------
Processing finished in 8.7661 seconds.
Throughput: 11409.63 Pw/s.

>>> MATCH FOUND! <<<
The correct password is: *48GCEErisUmberCastlePicasso
--------------------------------------------------
```

#### Option 2: Using `run_batch.js` for Large Lists

The `run_batch.js` script is designed to process massive password files by splitting them into manageable chunks.

1.  Name your master password file `keys.txt`.
2.  Run the script using Node.js:
    ```bash
    node run_batch.js
    ```
3.  To resume processing from a specific line, pass the line number as an argument. The script will skip all lines before it.
    ```bash
    # Skip the first 10,000,000 lines and start from there
    node run_batch.js 10000000
    ```

The script works by creating a temporary `input_batch.txt` file, calling the CUDA executable, and then deleting the temporary file. It logs its progress to `batch_runner.log`. You can configure the batch size on line 13 of `run_batch.js`.

**Important:** When using `run_batch.js`, ensure there are no other `input_*.txt` files in the directory, as the CUDA program would otherwise load them in addition to the batch file.

**Example Output:**
```
Starting batch processing for 'keys.txt'
Batch size: 1,000 lines
---

[2025-09-05T16:00:04.487Z] Starting Batch #10 (Lines 9,001 - 10,000)

Starting CUDA process for input_batch.txt...
Loading passwords from:
 - input_batch.txt
1000 passwords loaded.
...
>>> MATCH FOUND! <<<
The correct password is: *48GCEErisUmberCastlePicasso
--------------------------------------------------
Temporary file 'input_batch.txt' has been cleaned up.

>>> MATCH FOUND! Stopping automation. <<<
```

---

### A Note on Puzzles (Search Pattern)

The kernel validates a successful decryption by searching for a hardcoded text pattern. This pattern is defined in `combined_main.cu` on line 119:

```c++
const char* pattern = "\"kty\":\"RSA\"";
```

This pattern works for puzzles where the decrypted content is an RSA key file (like Puzzles 3 and 5). For other puzzles with different content (like Puzzle 7, which contains `"version":3,"id"`), you will need to change this pattern and recompile the program accordingly.

You can find test files with the correct password at the end of the list in the folders for Puzzle 5 and Puzzle 7 to verify your setup.