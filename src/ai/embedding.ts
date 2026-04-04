import { execFileSync } from 'node:child_process';

export interface EmbeddingResult {
    embeddings: number[][];
    model: string;
}

function recordToText(record: Record<string, unknown>): string {
    const parts: string[] = [];
    const keys = Object.keys(record).sort();

    for (const key of keys) {
        const value = record[key];
        if (value === null || value === undefined) continue;

        const strVal = String(value);
        if (strVal.length > 0) {
            parts.push(`${key}=${strVal}`);
        }
    }

    return parts.join(' | ');
}

export function getEmbeddings(
    model: string,
    records: Record<string, unknown>[],
): EmbeddingResult {
    const texts = records.map(recordToText);
    const embeddings: number[][] = [];

    for (const text of texts) {
        const prompt = `Generate embedding for: ${text}`;
        const output = execFileSync('ollama', ['embed', model, prompt], {
            encoding: 'utf-8',
            stdio: ['ignore', 'pipe', 'pipe'],
            maxBuffer: 1024 * 1024 * 10,
        });

        try {
            const parsed = JSON.parse(output);
            if (Array.isArray(parsed.embedding)) {
                embeddings.push(parsed.embedding);
            } else if (Array.isArray(parsed.embeddings) && parsed.embeddings.length > 0) {
                embeddings.push(parsed.embeddings[0]);
            } else {
                throw new Error('Invalid embedding response format');
            }
        } catch (error) {
            throw new Error(`Failed to parse embedding response: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    return { embeddings, model };
}

export async function getEmbeddingsStreaming(
    model: string,
    records: Record<string, unknown>[],
    onProgress?: (current: number, total: number) => void,
): Promise<EmbeddingResult> {
    const texts = records.map(recordToText);
    const embeddings: number[][] = [];
    const total = texts.length;

    for (let i = 0; i < texts.length; i++) {
        onProgress?.(i + 1, total);

        const text = texts[i];
        const output = execFileSync('ollama', ['embed', model, text], {
            encoding: 'utf-8',
            stdio: ['ignore', 'pipe', 'pipe'],
            maxBuffer: 1024 * 1024 * 10,
        });

        try {
            const parsed = JSON.parse(output);
            if (Array.isArray(parsed.embedding)) {
                embeddings.push(parsed.embedding);
            } else if (Array.isArray(parsed.embeddings) && parsed.embeddings.length > 0) {
                embeddings.push(parsed.embeddings[0]);
            } else {
                throw new Error('Invalid embedding response format');
            }
        } catch (error) {
            throw new Error(`Failed to parse embedding for record ${i}: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    onProgress?.(total, total);
    return { embeddings, model };
}

export function validateEmbeddingModel(model: string): boolean {
    try {
        const output = execFileSync('ollama', ['show', model, '--modelfile'], {
            encoding: 'utf-8',
            stdio: ['ignore', 'pipe', 'pipe'],
            timeout: 10000,
        });
        return output.includes('embed') || output.includes('architecture');
    } catch {
        return false;
    }
}
