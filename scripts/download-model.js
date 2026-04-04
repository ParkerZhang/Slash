#!/usr/bin/env node
/**
 * Download embedding model to modelFiles folder
 * Usage: npm run download-model
 */

import * as path from 'path';
import * as fs from 'fs';
import { env } from '@huggingface/transformers';

const MODEL_DIR = path.join(process.cwd(), 'modelFiles');

// Set cache location
env.cacheDir = MODEL_DIR;

async function downloadModel() {
    if (!fs.existsSync(MODEL_DIR)) {
        fs.mkdirSync(MODEL_DIR, { recursive: true });
    }

    console.log('Downloading Xenova/all-MiniLM-L6-v2 to modelFiles/...');
    console.log('(This is a ~23MB quantized model)');
    console.log(`Cache directory: ${MODEL_DIR}\n`);

    try {
        const { pipeline } = await import('@huggingface/transformers');
        const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
            quantized: true,
        });

        console.log('\n✓ Model downloaded successfully!');
        console.log(`Model files saved to: ${MODEL_DIR}`);
        
        // List downloaded files
        const files = fs.readdirSync(MODEL_DIR, { recursive: true });
        if (files.length > 0) {
            console.log('\nDownloaded files:');
            files.forEach(f => console.log(`  - ${f}`));
        }
        
        console.log('\nYou can now use /cluster without needing Ollama.');
    } catch (error) {
        console.error('❌ Failed to download model:', error.message);
        process.exit(1);
    }
}

downloadModel();
