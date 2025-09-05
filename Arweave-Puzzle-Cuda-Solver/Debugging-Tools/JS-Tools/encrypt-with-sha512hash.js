const CryptoJS = require('crypto-js');
const util = require('util');

// Configuration (as in the bug case!)
CryptoJS.algo.AES.keySize = 32;
CryptoJS.algo.EvpKDF.cfg.iterations = 10000;
CryptoJS.algo.EvpKDF.cfg.keySize = 32;

const password = "c80e33e57328a3f4524d78e337c13e800fffbf9629215d8987fe57d384f80dbd559091cd5d1a0e2ba40c403c68086c0f134bba71bdec7dc7d90738f6f945ff10";
const plaintext = "test";

// --- ENCRYPT ---
const encryptedObj = CryptoJS.AES.encrypt(plaintext, password);
const encryptedString = encryptedObj.toString();
console.log("encrypted:", encryptedString);

const rawBuffer = Buffer.from(encryptedString, 'base64');
const isSalted = rawBuffer.slice(0, 8).toString('utf8') === "Salted__";
let salt = null, ctBuffer = null;
if (isSalted) {
    salt = rawBuffer.slice(8, 16);
    ctBuffer = rawBuffer.slice(16);
} else {
    ctBuffer = rawBuffer;
}

// --- KDF Key/IV calculation (as CryptoJS does internally) ---
const keySizeWords = CryptoJS.algo.AES.keySize; // 32
const ivSizeWords = 4; // 16 bytes
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

function wordsToHex(words, len) {
    let out = "";
    for (let i = 0; i < len; ++i) {
        const w = words[i];
        out += ("00000000" + ((w >>> 0).toString(16))).slice(-8);
    }
    return out;
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

// --- Number of rounds (CryptoJS: keyWords + 6) ---
const rounds = keySizeWords + 6;

// --- DECRYPT ---
const decrypted = CryptoJS.AES.decrypt(encryptedString, password);
const decryptedHex = decrypted.toString(CryptoJS.enc.Hex);
const decryptedUtf8 = CryptoJS.enc.Utf8.stringify(decrypted);

// --- Padding block on encryption: ---
const blockSize = CryptoJS.algo.AES.blockSize * 4; // 16 Bytes
const messageBytes = Buffer.from(plaintext, "utf8");
const padLen = blockSize - (messageBytes.length % blockSize);
const padByte = padLen;
const padBlock = Buffer.alloc(padLen, padByte);

// --- Expanded Key (all subkeys/round keys) ---
const keyWA = CryptoJS.lib.WordArray.create(keyWords, keySizeWords * 4);
const ivWA = CryptoJS.lib.WordArray.create(ivWords, ivSizeWords * 4);
const aes = CryptoJS.algo.AES.createEncryptor(keyWA, { iv: ivWA });
const expandedKeyWords = aes._keySchedule; // Array of Words

// --- KDF stream as bytes (Key+IV) ---
const kdfBuffer = wordArrayToBuffer(derived);

// --- FULL DEBUG OUTPUT ---
console.log("----- DEBUG INFOS -----");
console.log("Password (ASCII):", password);
console.log("Password (hex):", Buffer.from(password, 'utf8').toString('hex'));
console.log("Plaintext:", plaintext);
console.log("Plaintext (utf8 bytes):", bufToHex(messageBytes));
console.log("Used keySize (words):", keySizeWords, "(bytes):", keySizeWords * 4);
console.log("Used ivSize (words):", ivSizeWords, "(bytes):", ivSizeWords * 4);
console.log("KDF Iterations:", iterations);
console.log("AES Block size (bytes):", blockSize);
console.log("AES Number of rounds:", rounds);
console.log("Padding block (hex):", bufToHex(padBlock));
console.log("Padding len:", padLen, "Padding byte:", padByte);

console.log("Salt (hex):", salt ? bufToHex(salt) : "<none>");
console.log("Raw ciphertext (hex):", bufToHex(ctBuffer));
console.log("Raw ciphertext length:", ctBuffer.length);

console.log("KDF key (hex, full):", wordsToHex(keyWords, keySizeWords), "  (", (keySizeWords*4), "bytes )");
console.log("KDF IV (hex, full):", wordsToHex(ivWords, ivSizeWords), "  (", (ivSizeWords*4), "bytes )");
console.log("KDF key+IV full (hex):", wordsToHex(derived.words, keySizeWords + ivSizeWords));
console.log("KDF-Stream (all bytes, key+iv):", bufToHex(kdfBuffer), "(length:", kdfBuffer.length, ")");

// --- Expanded Key Schedule ---
console.log("Expanded Key (all subkeys, hex, " + expandedKeyWords.length + " words):");
console.log(wordsToHex(expandedKeyWords, expandedKeyWords.length));

// --- Internal structure of the EncryptedObj ---
console.log("EncryptedObj (full):", util.inspect(encryptedObj, { depth: 5, colors: false, showHidden: false }));

console.log("Decrypted (hex):", decryptedHex);
console.log("Decrypted (utf8):", decryptedUtf8);
console.log("Decrypted Raw WordArray:", decrypted);

if (decryptedUtf8 === plaintext) {
    console.log("✅ Success! Plaintext matches.");
} else {
    console.log("❌ No match. Plaintext does not match.");
}
console.log("-----------------------");