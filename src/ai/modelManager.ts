import { execFileSync, spawn } from 'node:child_process';
import { ElasticSearchResult, LoadedFile } from '../core/types.js';

export function getAvailableModels(): string[] {
    try {
        const output = execFileSync('ollama', ['list'], {
            encoding: 'utf-8',
            stdio: ['ignore', 'pipe', 'pipe'],
        });

        const lines = output
            .split('\n')
            .map((line) => line.trimEnd())
            .filter((line) => line.trim().length > 0);

        if (lines.length <= 1) {
            return [];
        }

        return lines
            .slice(1)
            .map((line) => line.trim().split(/\s{2,}/)[0])
            .filter((name) => name.length > 0);
    } catch {
        return [];
    }
}

function buildSchemaPrompt(
    fileId: string,
    data: ElasticSearchResult,
    fileFormat: LoadedFile['fileFormat'],
    extraInstructions?: string,
): string {
    const sampleHits = data.hits.hits.slice(0, 10).map((hit) => hit._source);
    const payload = {
        fileId,
        fileFormat: fileFormat || 'generated',
        totalHits: data.hits.total.value,
        sampleSize: sampleHits.length,
        sample: sampleHits,
    };

    const effectiveFormat = fileFormat || 'generated';
    const sourceDescription =
        fileFormat === 'csv'
            ? `You are given sample rows from loaded file '${fileId}' (format: csv).`
            : fileFormat === 'json'
                ? `You are given sample records from loaded file '${fileId}' (format: json).`
                : `You are given sample records from loaded file '${fileId}' (format: ${effectiveFormat}).`;
    const schemaInstruction =
        fileFormat === 'csv'
            ? `Infer the row schema for file '${fileId}'. Return it as a JSON Schema object.`
            : `Infer the record schema for file '${fileId}'. Return it as a JSON Schema object.`;

    return [
        sourceDescription,
        schemaInstruction,
        'Return only valid JSON.',
        'Do not wrap the response in markdown fences.',
        'Use JSON Schema draft-07 style keys when appropriate.',
        extraInstructions?.trim() ? `Additional instructions: ${extraInstructions.trim()}` : '',
        '',
        JSON.stringify(payload, null, 2),
    ].filter((line) => line.length > 0).join('\n');
}

function extractJsonObject(raw: string): string {
    const trimmed = raw.trim();
    if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
        return trimmed;
    }

    const fencedMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
    if (fencedMatch?.[1]) {
        return fencedMatch[1].trim();
    }

    const firstBrace = trimmed.indexOf('{');
    const lastBrace = trimmed.lastIndexOf('}');
    if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
        return trimmed.slice(firstBrace, lastBrace + 1);
    }

    return trimmed;
}

export function inferSchemaForData(
    model: string,
    fileId: string,
    data: ElasticSearchResult,
    fileFormat: LoadedFile['fileFormat'],
    extraInstructions?: string,
): Record<string, unknown> {
    const prompt = buildSchemaPrompt(fileId, data, fileFormat, extraInstructions);

    const output = execFileSync('ollama', ['run', model, prompt], {
        encoding: 'utf-8',
        stdio: ['ignore', 'pipe', 'pipe'],
        maxBuffer: 1024 * 1024 * 10,
    });

    const jsonText = extractJsonObject(output);
    const parsed = JSON.parse(jsonText) as Record<string, unknown>;

    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        throw new Error('Model did not return a JSON object schema');
    }

    return parsed;
}

export async function inferSchemaForDataStreaming(
    model: string,
    fileId: string,
    data: ElasticSearchResult,
    fileFormat: LoadedFile['fileFormat'],
    extraInstructions?: string,
    onChunk?: (chunk: string) => void,
): Promise<Record<string, unknown>> {
    const prompt = buildSchemaPrompt(fileId, data, fileFormat, extraInstructions);

    return await new Promise<Record<string, unknown>>((resolve, reject) => {
        const child = spawn('ollama', ['run', model, prompt], {
            stdio: ['ignore', 'pipe', 'pipe'],
        });

        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (chunk: Buffer | string) => {
            const text = chunk.toString();
            stdout += text;
            onChunk?.(text);
        });

        child.stderr.on('data', (chunk: Buffer | string) => {
            stderr += chunk.toString();
        });

        child.on('error', (error) => {
            reject(error);
        });

        child.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(stderr.trim() || `ollama exited with code ${code}`));
                return;
            }

            try {
                const jsonText = extractJsonObject(stdout);
                const parsed = JSON.parse(jsonText) as Record<string, unknown>;
                if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
                    reject(new Error('Model did not return a JSON object schema'));
                    return;
                }
                resolve(parsed);
            } catch (error) {
                reject(error instanceof Error ? error : new Error(String(error)));
            }
        });
    });
}

export async function runPromptStreaming(
    model: string,
    prompt: string,
    onChunk?: (chunk: string) => void,
): Promise<string> {
    return await new Promise<string>((resolve, reject) => {
        const child = spawn('ollama', ['run', model, prompt], {
            stdio: ['ignore', 'pipe', 'pipe'],
        });

        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (chunk: Buffer | string) => {
            const text = chunk.toString();
            stdout += text;
            onChunk?.(text);
        });

        child.stderr.on('data', (chunk: Buffer | string) => {
            stderr += chunk.toString();
        });

        child.on('error', (error) => {
            reject(error);
        });

        child.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(stderr.trim() || `ollama exited with code ${code}`));
                return;
            }
            resolve(stdout.trim());
        });
    });
}
