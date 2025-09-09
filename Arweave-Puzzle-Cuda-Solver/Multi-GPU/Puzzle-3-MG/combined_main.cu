#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <iomanip>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "crypto_kernels.cuh"

#define CUDA_CHECK(x) do { cudaError_t e = x; if(e != cudaSuccess) { \
    fprintf(stderr, "CUDA ERROR: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);}} while(0)

// =============================================================================
// KERNEL
// =============================================================================
__global__ void generation_and_validation_kernel(
    const char* passwords_data, const int* password_lengths, int password_maxlen,
    const uint8_t* salt, const uint8_t* ciphertext, size_t ciphertext_len,
    int sha_iterations, int kdf_iterations, int key_len_bytes, int iv_len_bytes,
    int* found_flag, char* found_password, int num_passwords,
    int debug_idx
) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_passwords || (*found_flag && debug_idx == -1)) return;

    // --- PART 1: Key & IV Generation ---
    const char* pw = passwords_data + idx * password_maxlen;
    int pw_len = password_lengths[idx];
    char local_pw[128];
    for(int i=0; i<pw_len; ++i) local_pw[i] = pw[i];

    char sha512_hex_output[129] = {0};
    iterative_sha512(local_pw, pw_len, sha512_hex_output, sha_iterations);
    
    uint8_t kdf_key_iv[144];
    int kdf_key_iv_len = 0;
    uint8_t md5_prev[16] = {0};
    bool has_prev = false;
    int total_bytes_needed = key_len_bytes + iv_len_bytes;

    while (kdf_key_iv_len < total_bytes_needed) {
        uint8_t block[256];
        int block_len = 0;
        if (has_prev) for (int i = 0; i < 16; ++i) block[block_len++] = md5_prev[i];
        for (int i = 0; i < 128; ++i) block[block_len++] = (uint8_t)sha512_hex_output[i];
        for (int i = 0; i < 8; ++i) block[block_len++] = salt[i];

        md5_cuda(block, block_len, md5_prev);
        for (int i = 1; i < kdf_iterations; ++i) md5_cuda(md5_prev, 16, md5_prev);

        int copy_len = 16;
        if (kdf_key_iv_len + copy_len > total_bytes_needed) copy_len = total_bytes_needed - kdf_key_iv_len;
        for (int i = 0; i < copy_len; ++i) kdf_key_iv[kdf_key_iv_len + i] = md5_prev[i];
        
        kdf_key_iv_len += copy_len;
        has_prev = true;
    }

    uint8_t* generated_key = kdf_key_iv;
    uint8_t* generated_iv = kdf_key_iv + key_len_bytes;
    
    // --- PART 2: AES Decryption & Validation ---
    uint8_t round_key[240 * 4];
    uint8_t local_iv[AES_BLOCKLEN];
    uint8_t decrypted_buffer[4096]; 
    uint8_t aes_rounds = 38, aes_Nk = 32;

    KeyExpansion(round_key, generated_key, aes_rounds, aes_Nk);
    for(int i=0; i<AES_BLOCKLEN; ++i) local_iv[i] = generated_iv[i];
    for(size_t i=0; i<ciphertext_len; ++i) decrypted_buffer[i] = ciphertext[i];

    uint8_t storeNextIv[AES_BLOCKLEN];
    for (size_t i = 0; i < ciphertext_len; i += AES_BLOCKLEN) {
        for(int j=0; j<AES_BLOCKLEN; ++j) storeNextIv[j] = decrypted_buffer[i+j];
        InvCipher((state_t*)(decrypted_buffer + i), round_key, aes_rounds);
        XorWithIv(decrypted_buffer + i, local_iv);
        for(int j=0; j<AES_BLOCKLEN; ++j) local_iv[j] = storeNextIv[j];
    }
    
    const char* pattern = "\"kty\":\"RSA\"";
    for (size_t i = 0; i + 10 <= ciphertext_len; ++i) {
        bool match = true;
        for(int j=0; j<10; ++j) if (decrypted_buffer[i+j] != pattern[j]) { match = false; break; }
        if (match) {
            if (atomicExch(found_flag, 1) == 0) {
                for(int k=0; k < pw_len; ++k) found_password[k] = local_pw[k];
                found_password[pw_len] = '\0';
            }
            return;
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================
bool load_and_decode_message(const std::string& filename, std::vector<uint8_t>& salt, std::vector<uint8_t>& ciphertext) {
    std::ifstream infile(filename);
    if (!infile) return false;
    std::string b64_data((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
    infile.close();
    b64_data.erase(std::remove(b64_data.begin(), b64_data.end(), '\n'), b64_data.end());
    b64_data.erase(std::remove(b64_data.begin(), b64_data.end(), '\r'), b64_data.end());

    std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uint8_t> decoded;
    int val = 0, bits = 0;
    for (char c : b64_data) {
        if (c == '=') break;
        auto pos = chars.find(c);
        if (pos == std::string::npos) continue;
        val = (val << 6) | pos;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            decoded.push_back(val >> bits);
        }
    }
    if (decoded.size() < 16 || std::string(decoded.begin(), decoded.begin() + 8) != "Salted__") return false;
    salt.assign(decoded.begin() + 8, decoded.begin() + 16);
    ciphertext.assign(decoded.begin() + 16, decoded.end());
    return true;
}

int main(int argc, char* argv[]) {
    // *** NEU: Überprüfe, ob ein Dateiname als Argument übergeben wurde ***
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_filename.txt>\n";
        return 1;
    }
    std::string input_filename = argv[1];
    std::cout << "Loading passwords from: " << input_filename << "\n";

    std::vector<std::string> passwords;
    std::ifstream infile(input_filename);
    if (!infile) {
        std::cerr << "Error: Could not open password file '" << input_filename << "'.\n";
        return 1;
    }
    std::string line;
    while (std::getline(infile, line)) {
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (!line.empty()) passwords.push_back(line);
    }
    
    if (passwords.empty()) { std::cerr << "Error: No passwords found in '" << input_filename << "'.\n"; return 1; }
    std::cout << passwords.size() << " passwords loaded.\n\n";

    std::vector<uint8_t> salt, ciphertext;
    if (!load_and_decode_message("message.b64", salt, ciphertext)) {
        std::cerr << "Error: 'message.b64' could not be read or decoded.\n"; return 1;
    }
    std::cout << "Message 'message.b64' loaded.\n";
    std::cout << " - Salt (HEX): ";
    for(uint8_t b : salt) std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)b;
    std::cout << "\n - Ciphertext length: " << ciphertext.size() << " Bytes\n\n";

    const int PW_MAXLEN = 128;
    std::vector<char> h_pw_data(passwords.size() * PW_MAXLEN, 0);
    std::vector<int>  h_pw_lengths(passwords.size());
    for(size_t i=0; i<passwords.size(); ++i) {
        int len = std::min((int)passwords[i].size(), PW_MAXLEN);
        memcpy(&h_pw_data[i * PW_MAXLEN], passwords[i].c_str(), len);
        h_pw_lengths[i] = len;
    }

    char* d_pw_data; int* d_pw_lengths; uint8_t* d_salt; uint8_t* d_ciphertext;
    int* d_found_flag; char* d_found_password;
    CUDA_CHECK(cudaMalloc(&d_pw_data, h_pw_data.size()));
    CUDA_CHECK(cudaMalloc(&d_pw_lengths, h_pw_lengths.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_salt, salt.size()));
    CUDA_CHECK(cudaMalloc(&d_ciphertext, ciphertext.size()));
    CUDA_CHECK(cudaMalloc(&d_found_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_found_password, PW_MAXLEN + 1));
    CUDA_CHECK(cudaMemcpy(d_pw_data, h_pw_data.data(), h_pw_data.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pw_lengths, h_pw_lengths.data(), h_pw_lengths.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_salt, salt.data(), salt.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ciphertext, ciphertext.data(), ciphertext.size(), cudaMemcpyHostToDevice));
    int h_found_flag = 0;
    CUDA_CHECK(cudaMemcpy(d_found_flag, &h_found_flag, sizeof(int), cudaMemcpyHostToDevice));

    std::cout << "Starting...\n";
    auto start = std::chrono::high_resolution_clock::now();

    int threads_per_block = 64;
    int blocks = (passwords.size() + threads_per_block - 1) / threads_per_block;

    generation_and_validation_kernel<<<blocks, threads_per_block>>>(
        d_pw_data, d_pw_lengths, PW_MAXLEN, d_salt, d_ciphertext, ciphertext.size(),
        11513, 10000, 128, 16,
        d_found_flag, d_found_password, passwords.size(),
        -1
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();
    // *** KORRIGIERTE ZEILE 222 ***
    double elapsed_s = std::chrono::duration<double>(stop - start).count();

    CUDA_CHECK(cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "\n--------------------------------------------------\n";
    std::cout << "Processing finished in " << std::fixed << std::setprecision(4) << elapsed_s << " seconds.\n";
    if (elapsed_s > 0) std::cout << "Throughput: " << std::fixed << std::setprecision(2) << (passwords.size() / elapsed_s) << " Pw/s.\n";
    
    if (h_found_flag) {
        std::vector<char> h_found_password(PW_MAXLEN + 1, 0);
        CUDA_CHECK(cudaMemcpy(h_found_password.data(), d_found_password, PW_MAXLEN + 1, cudaMemcpyDeviceToHost));
        std::cout << "\n>>> MATCH FOUND! <<<\n";
        std::cout << "The correct password is: " << std::string(h_found_password.data()) << "\n";
    } else {
        std::cout << "\nNo match found in the list.\n";
    }
    std::cout << "--------------------------------------------------\n";

    cudaFree(d_pw_data); cudaFree(d_pw_lengths); cudaFree(d_salt); cudaFree(d_ciphertext);
    cudaFree(d_found_flag); cudaFree(d_found_password);

    return 0;
}