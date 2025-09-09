# Password Combination Generator

This Python script (`key-gen-v2.py`) is a powerful tool for generating large password lists based on a combinator-style attack. It creates every possible combination of password parts defined in an input file and writes them to an output file.

## Purpose

When solving puzzles or attempting to crack passwords, it's common to know certain parts of the password but not the exact combination. This script automates the process of generating every single permutation from a list of known parts, creating a comprehensive wordlist that can then be used with the CUDA brute-force tool.

## How it Works

The script reads an input file named `keys-input.txt`. Each line in this file represents a "slot" or segment of the final password. The elements on each line, separated by commas, are the possible options for that specific slot.

The script then generates all possible strings by taking one element from each line and concatenating them together.

### Input File Format (`keys-input.txt`)

*   Each line represents a part of the password.
*   On each line, list all possible words or characters for that part, separated by commas.
*   The script will combine one element from line 1, with one from line 2, with one from line 3, and so on.

**Example `keys-input.txt`:**
```
partA1,partA2
partB1,partB2,partB3
partC1
```

This input would generate the following combinations:
```
partA1partB1partC1
partA1partB2partC1
partA1partB3partC1
partA2partB1partC1
partA2partB2partC1
partA2partB3partC1
```

### Usage

1.  **Prerequisites:** You need Python 3 installed. No external libraries are required.
2.  **Create the Input File:** Create a file named `keys-input.txt` in the same directory and define your password parts as described above.
3.  **Run the Script:** Execute the script from your terminal:
    ```bash
    python key-gen-v2.py
    ```
4.  **Get the Output:** The script will generate a file named `keys-output.txt` containing all the combinations. This file can be renamed to `keys.txt` to be used with the `run_batch.js` helper script.

### Features

*   **Total Combination Calculation:** Before starting, the script calculates and displays the total number of passwords that will be generated.
*   **File Size Estimation:** It estimates the approximate size of the output file and asks for confirmation if it exceeds 1 GB, preventing you from accidentally filling up your storage.
*   **Live Progress Bar:** For large lists, it shows a detailed progress bar with the percentage complete, the number of generated keys, and an estimated time remaining.
*   **Performance Metrics:** After completion, it reports the total time taken and the generation speed (combinations per second).