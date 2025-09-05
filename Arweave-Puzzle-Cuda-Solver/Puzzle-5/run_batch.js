const fs = require('fs');
const readline = require('readline');
const { spawn } = require('child_process');

// ===================================================================
// ========================= CONFIGURATION ===========================
// ===================================================================

// The large password list to be processed.
const MASTER_KEY_FILE = 'keys.txt';

// Number of passwords to send to the GPU per run.
const BATCH_SIZE = 1_000_000;

// The name of the temporary file created for each batch.
const TEMP_BATCH_FILE = 'input_batch.txt';


const CUDA_EXECUTABLE = './ar-brute'; // The CUDA program to call.
const CUDA_ARGS = [];                 // There are no additional arguments.
// =================================================================================

// Log file to track progress and know where you were in case of a crash.
const LOG_FILE = 'batch_runner.log';

// ===================================================================
// ======================= AUTOMATION ================================
// ===================================================================

/**
 * Executes the CUDA program for a single batch file.
 * @returns {Promise<boolean>} True if a match was found, otherwise false.
 */
function runCudaProcess() {
    return new Promise((resolve) => {
        console.log(`Starting CUDA process for ${TEMP_BATCH_FILE}...`);

        const cudaProcess = spawn(CUDA_EXECUTABLE, CUDA_ARGS, { stdio: 'pipe' });

        let foundHit = false;

        cudaProcess.stdout.on('data', (data) => {
            const output = data.toString();
            process.stdout.write(output);
            if (output.includes('MATCH FOUND')) {
                foundHit = true;
            }
        });

        cudaProcess.stderr.on('data', (data) => {
            process.stderr.write(data.toString());
        });

        cudaProcess.on('close', (code) => {
            if (code !== 0 && !foundHit) {
                console.error(`\nCUDA process exited with error code ${code}.`);
            }
            try {
                fs.unlinkSync(TEMP_BATCH_FILE);
                console.log(`\nTemporary file '${TEMP_BATCH_FILE}' has been cleaned up.`);
            } catch (err) {}
            resolve(foundHit);
        });

        cudaProcess.on('error', (err) => {
            console.error('\nError starting the CUDA process. Is the file executable?');
            console.error(err);
            resolve(false);
        });
    });
}

/**
 * Main function to control the batch process.
 */
async function main() {
    const startLine = process.argv[2] ? parseInt(process.argv[2], 10) : 0;

    if (!fs.existsSync(MASTER_KEY_FILE)) {
        console.error(`Error: The master file '${MASTER_KEY_FILE}' was not found.`);
        return;
    }

    console.log(`Starting batch processing for '${MASTER_KEY_FILE}'`);
    if (startLine > 0) {
        console.log(`Resuming from line: ${startLine.toLocaleString()}`);
    }
    console.log(`Batch size: ${BATCH_SIZE.toLocaleString()} lines`);
    console.log('---');

    const fileStream = fs.createReadStream(MASTER_KEY_FILE);
    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity,
    });

    let currentLineNumber = 0;
    let batchNumber = Math.floor(startLine / BATCH_SIZE) + 1;
    let batch = [];

    for await (const line of rl) {
        currentLineNumber++;

        if (currentLineNumber <= startLine) {
            if (currentLineNumber % 1_000_000 === 0) {
                process.stdout.write(`\rSkipping lines... ${currentLineNumber.toLocaleString()}`);
            }
            continue;
        }

        batch.push(line);

        if (batch.length >= BATCH_SIZE) {
            const batchStartLine = currentLineNumber - batch.length + 1;
            const logMessage = `[${new Date().toISOString()}] Starting Batch #${batchNumber} (Lines ${batchStartLine.toLocaleString()} - ${currentLineNumber.toLocaleString()})\n`;
            
            console.log(`\n${logMessage.trim()}`);
            fs.appendFileSync(LOG_FILE, logMessage);
            
            fs.writeFileSync(TEMP_BATCH_FILE, batch.join('\n'));
            
            const hitFound = await runCudaProcess();
            
            if (hitFound) {
                console.log('\n\n>>> MATCH FOUND! Stopping automation. <<<');
                return;
            }

            batch = [];
            batchNumber++;
        }
    }
    
    if (batch.length > 0) {
        const batchStartLine = currentLineNumber - batch.length + 1;
        const logMessage = `[${new Date().toISOString()}] Starting final Batch #${batchNumber} (Lines ${batchStartLine.toLocaleString()} - ${currentLineNumber.toLocaleString()})\n`;

        console.log(`\n${logMessage.trim()}`);
        fs.appendFileSync(LOG_FILE, logMessage);
        
        fs.writeFileSync(TEMP_BATCH_FILE, batch.join('\n'));
        const hitFound = await runCudaProcess();
        
        if (hitFound) {
            console.log('\n\n>>> MATCH FOUND! Stopping automation. <<<');
            return;
        }
    }

    console.log('\n\nAll batches have been processed. No match found in the entire file.');
}

main();