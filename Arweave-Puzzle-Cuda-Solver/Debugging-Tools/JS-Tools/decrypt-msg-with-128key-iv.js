const CryptoJS = require('crypto-js');

// --- Values directly from your debug output ---
const ciphertext_b64 = "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";
const salt_hex = "84dbff786c983a3f";
const key_hex = "bdcf243c89cc1a867b60374509bff90507efc80e93d75bcbc2a8017250e7fee2d632ffc3c1918e7c0b1faf18a9560f37639761475c55d858aff6f91bebe81dfb2a90f6763806e9e9c3c65bd0f1b420053031db5831cf964e5629fa2659011722df9ec816e4c4aba06b26c686a061dbcaa85ed617a8cf56c49467183d8593461a";
const iv_hex  = "794943e977489b5ee9f303bce4b1b412";

// --- Helper functions ---
function hexToWordArray(hexStr) {
    return CryptoJS.enc.Hex.parse(hexStr);
}

function wordArrayToHex(wa) {
    return wa.toString(CryptoJS.enc.Hex);
}

function wordToHex(word) {
    // word: 32-bit int
    return ("00000000" + (word >>> 0).toString(16)).slice(-8);
}

function printKeySchedule(aes, rounds, Nk) {
    const schedule = aes._keySchedule; // Array of 32-bit words
    const Nb = 4;
    console.log("=== AES KeySchedule DEBUG ===");
    console.log("Rounds (Nr):", rounds);
    console.log("Nk (Key-Words):", Nk, "(Key-Bytes:", Nk*4, ")");
    console.log("KeySchedule-Length (Words):", schedule.length);
    for (let r = 0; r <= rounds; ++r) {
        let roundKey = [];
        for (let c = 0; c < Nb; ++c) {
            roundKey.push(wordToHex(schedule[r*Nb + c]));
        }
        console.log(`RoundKey[${r}]: ${roundKey.join(" ")}`);
    }
    console.log("============================");
}

// --- Extract raw ciphertext (without Salted__ and Salt) ---
function extractRawCiphertext(b64) {
    const raw = Buffer.from(b64, 'base64');
    if (raw.slice(0, 8).toString('utf8') !== "Salted__") throw new Error("No Salted__ header!");
    return raw.slice(16); // after "Salted__" (8) + salt (8)
}

const ciphertext_bytes = extractRawCiphertext(ciphertext_b64);
const ciphertextWA = CryptoJS.lib.WordArray.create(ciphertext_bytes);

const keyWA = hexToWordArray(key_hex);
const ivWA = hexToWordArray(iv_hex);

// HACK: Internal access to AES object and KeySchedule!
const AES = CryptoJS.algo.AES.createDecryptor(keyWA, { iv: ivWA, mode: CryptoJS.mode.CBC, padding: CryptoJS.pad.Pkcs7 });

// Extract parameters:
const Nk = keyWA.words.length;
const rounds = Nk + 6;

// Debug KeySchedule:
printKeySchedule(AES, rounds, Nk);

// Decryption:
const decrypted = AES.process(ciphertextWA);
const final = AES.finalize();

const full = decrypted.clone().concat(final);

console.log("Decrypted (hex):", wordArrayToHex(full));
console.log("Decrypted (utf8):", full.toString(CryptoJS.enc.Utf8));