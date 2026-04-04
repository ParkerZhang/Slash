import { pipeline, env } from '@huggingface/transformers';
import * as path from 'path';

// Use local modelFiles folder in project root
const MODEL_DIR = path.join(process.cwd(), 'modelFiles');

// Set cache location
env.cacheDir = MODEL_DIR;

let extractor: any = null;
let isInitialized = false;

async function getExtractor() {
    if (!extractor) {
        extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
            quantized: true,
        });
        isInitialized = true;
    }
    return extractor;
}

export async function getLocalEmbeddings(
    records: Record<string, unknown>[],
    onProgress?: (current: number, total: number) => void,
): Promise<{ embeddings: number[][]; model: string }> {
    const ext = await getExtractor();
    const embeddings: number[][] = [];

    for (let i = 0; i < records.length; i++) {
        const rec = records[i];
        const text = Object.entries(rec)
            .map(([k, v]) => `${k}=${v}`)
            .join(' | ');

        const output = await ext(text, { pooling: 'mean', normalize: true });
        const embedding = Array.from(output.data);

        embeddings.push(embedding as number[]);
        onProgress?.(i + 1, records.length);
    }

    return { embeddings, model: 'Xenova/all-MiniLM-L6-v2' };
}

export function isLocalEmbeddingAvailable(): boolean {
    return isInitialized;
}
