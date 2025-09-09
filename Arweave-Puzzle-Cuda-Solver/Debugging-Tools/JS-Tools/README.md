# JavaScript Debugging Tools

This folder contains a set of Node.js scripts that were used to analyze, reverse-engineer, and replicate the unique cryptographic behavior of the Arweave puzzles. They provide deep insight into how CryptoJS handles the non-standard 128-byte (1024-bit) AES key.

### Prerequisites

To use these scripts, you need to have **Node.js** installed. You will also need the `crypto-js` library:
```bash
npm install crypto-js
```

---

### Script Descriptions

#### 1. `encrypt-with-sha512hash.js`
This script is used to create your own encrypted messages for testing.

*   **Function:** It takes a plaintext string (e.g., "test") and a password (which should be a 128-character hex string, like the output of the 11,513 SHA-512 iterations) and encrypts it using the same non-standard configuration as the Arweave puzzles.
*   **Utility:** This is essential for generating new test vectors. You can create a new encrypted message and use its output (ciphertext, salt, key, IV) to verify that your CUDA implementation can decrypt it correctly.
*   **Usage:**
    ```bash
    node encrypt-with-sha512hash.js
    ```
    The script will print a new Base64 encrypted message and a full breakdown of the cryptographic values used to create it.

#### 2. `decrypt-msg-with-sha512hash.js`
This is the primary script for understanding the full decryption pipeline from the post-SHA hash onwards.

*   **Function:** It takes the puzzle's encrypted message and a password (the post-SHA512 hash) and performs the full KDF (Key Derivation Function) and decryption, just like CryptoJS does it internally.
*   **Utility:** This script reveals all the critical intermediate values needed to build the CUDA solver:
    *   The 8-byte **Salt**.
    *   The 10,000-iteration MD5 process.
    *   The final **128-byte Key** and **16-byte IV**.
    *   The full **AES Key Schedule** (all round keys).
*   **Usage:**
    ```bash
    # Provide the post-SHA512 hash of a password candidate
    node decrypt-msg-with-sha512hash.js <post-sha512-hash-as-hex-string>
    ```

#### 3. `decrypt-msg-with-128key-iv.js`
This script isolates and tests only the final AES decryption step.

*   **Function:** It bypasses the KDF entirely. Instead, you provide it with a hardcoded 128-byte key and 16-byte IV (which you can get from `decrypt-msg-with-sha512hash.js`). It then attempts to decrypt the message using only these values.
*   **Utility:** Invaluable for debugging the CUDA AES implementation. If you have the correct key and IV but your CUDA code fails, this script helps confirm that the issue lies within your AES key expansion or decryption logic, not the KDF part. It also prints the full key schedule for direct comparison.
*   **Usage:**
    ```bash
    # Modify the hardcoded key and IV inside the script first
    node decrypt-msg-with-128key-iv.js
    ```