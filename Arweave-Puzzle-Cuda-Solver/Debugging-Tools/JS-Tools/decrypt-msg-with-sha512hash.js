const CryptoJS = require('crypto-js');
const util = require('util');

CryptoJS.algo.AES.keySize = 32;
CryptoJS.algo.EvpKDF.cfg.iterations = 10000;
CryptoJS.algo.EvpKDF.cfg.keySize = 32;

// Encrypted message
const encryptedMessage = "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";

// Utility for Hex output
function wordsToHex(words, len) {
    let out = "";
    for (let i = 0; i < len; ++i) {
        const w = words[i];
        out += ("00000000" + ((w >>> 0).toString(16))).slice(-8);
    }
    return out;
}

// Utility for Expanded Key Hex output
function expandedKeyToHex(expandedKey) {
    return expandedKey.map(w => ("00000000" + ((w >>> 0).toString(16))).slice(-8)).join('');
}

function bufToHex(buf) {
    return Array.from(buf).map(b => b.toString(16).padStart(2, '0')).join('');
}
function wordArrayToBuffer(wordArray) {
    const words = wordArray.words;
    const sigBytes = wordArray.sigBytes;
    const buffer = Buffer.alloc(sigBytes);
    for (let i = 0; i < sigBytes; i++) {
        buffer[i] = (words[i >>> 2] >>> (24 - (i % 4) * 8)) & 0xff;
    }
    return buffer;
}

// ---- Extended Debug Function ----
function testKey(password) {
    try {
        // --- Decrypt ---
        const decrypted = CryptoJS.AES.decrypt(encryptedMessage, password);

        // --- Debug: Salt extraction (if available) ---
        const rawBuffer = Buffer.from(encryptedMessage, 'base64');
        const isSalted = rawBuffer.toString('utf8', 0, 8) === "Salted__";
        let salt = null, ctBuffer = null;
        if (isSalted) {
            salt = rawBuffer.slice(8, 16);
            ctBuffer = rawBuffer.slice(16);
        } else {
            ctBuffer = rawBuffer;
        }

        // --- Debug: Key/IV calculation (CryptoJS internal) ---
        const keySizeWords = CryptoJS.algo.AES.keySize;
        const ivSizeWords = 4;
        const iterations = CryptoJS.algo.EvpKDF.cfg.iterations || 1;
        const EvpKDF = CryptoJS.algo.EvpKDF.create({
            keySize: keySizeWords + ivSizeWords,
            ivSize: ivSizeWords,
            iterations: iterations,
            salt: salt ? CryptoJS.lib.WordArray.create(salt) : undefined,
            hasher: CryptoJS.algo.MD5
        });
        const derived = EvpKDF.compute(password, salt ? CryptoJS.lib.WordArray.create(salt) : undefined);
        const keyWords = derived.words.slice(0, keySizeWords);
        const ivWords = derived.words.slice(keySizeWords, keySizeWords + ivSizeWords);

        // --- Number of rounds (as CryptoJS actually uses it) ---
        const rounds = keySizeWords + 6;

        // --- Expanded Key (all subkeys/round keys) ---
        const keyWA = CryptoJS.lib.WordArray.create(keyWords, keySizeWords * 4);
        const ivWA = CryptoJS.lib.WordArray.create(ivWords, ivSizeWords * 4);
        const aes = CryptoJS.algo.AES.createEncryptor(keyWA, { iv: ivWA });
        const expandedKeyWords = aes._keySchedule;

        // --- KDF stream as bytes (Key+IV) ---
        const kdfBuffer = wordArrayToBuffer(derived);

        // --- Padding block (can be reconstructed from length during decryption) ---
        const blockSize = CryptoJS.algo.AES.blockSize * 4; // 16 Bytes

        // --- FULL DEBUG OUTPUT ---
        console.log("----- DEBUG INFOS -----");
        console.log("Password (ASCII):", password);
        console.log("Password (hex):", Buffer.from(password, 'utf8').toString('hex'));
        console.log("Used keySize (words):", keySizeWords, "(bytes):", keySizeWords * 4);
        console.log("Used ivSize (words):", ivSizeWords, "(bytes):", ivSizeWords * 4);
        console.log("KDF Iterations:", iterations);
        console.log("AES Block size (bytes):", blockSize);
        console.log("AES Number of rounds:", rounds);

        console.log("Salt (hex):", salt ? bufToHex(salt) : "<none>");
        console.log("Raw ciphertext (hex, first 64 bytes):", bufToHex(ctBuffer).slice(0, 128));
        console.log("Raw ciphertext length:", ctBuffer.length);

        console.log("KDF key (hex, full):", wordsToHex(keyWords, keySizeWords), "  (", (keySizeWords*4), "bytes )");
        console.log("KDF IV (hex, full):", wordsToHex(ivWords, ivSizeWords), "  (", (ivSizeWords*4), "bytes )");
        console.log("KDF key+IV full (hex):", wordsToHex(derived.words, keySizeWords + ivSizeWords));
        console.log("KDF-Stream (all bytes, key+iv):", bufToHex(kdfBuffer), "(length:", kdfBuffer.length, ")");

        console.log("Expanded Key (all subkeys, hex, " + expandedKeyWords.length + " words):");
        console.log(expandedKeyToHex(expandedKeyWords));

        // --- Output decrypted as hex (first 64 bytes), plus as utf8 ---
        const decryptedHex = decrypted.toString(CryptoJS.enc.Hex);
        const decryptedUtf8 = CryptoJS.enc.Utf8.stringify(decrypted);

        console.log("Decrypted (hex, first 64 bytes):", decryptedHex.slice(0, 128));
        console.log("Decrypted (utf8, first 64):", decryptedUtf8.slice(0, 64));
        console.log("Decrypted (full utf8):", decryptedUtf8);
        console.log("Decrypted Raw WordArray:", decrypted);

        // --- Optional: Reconstruct padding info if plaintext is readable ---
        if (decrypted.sigBytes > 0) {
            const decBytes = Buffer.from(decryptedHex, 'hex');
            const padByte = decBytes[decBytes.length - 1];
            const paddingOK = decBytes.slice(-padByte).every(b => b === padByte);
            if (paddingOK) {
                console.log("Detected PKCS#7 padding: length", padByte, "padding byte:", padByte.toString(16));
            }
        }

        // --- Success/Fail check ---
        if (decryptedUtf8 && decryptedUtf8.includes('"kty":"RSA"')) {
            console.log("✅ Success! Key found:", password);
            return true;
        } else if (decryptedUtf8 === "test") {
            console.log("✅ Success! Plaintext matches.");
            return true;
        } else {
            console.log("❌ No match with this key.");
            return false;
        }
        console.log("-----------------------");
    } catch (error) {
        console.log("❌ Error:", error);
        return false;
    }
}

// Process command line argument
if (require.main === module) {
    if (process.argv.length < 3) {
        console.log("Usage: node debug.js PASSWORD");
        process.exit(1);
    }
    const password = process.argv[2];
    testKey(password);
}

module.exports = { testKey };