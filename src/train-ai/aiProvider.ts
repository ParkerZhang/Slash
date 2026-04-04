import * as fs from 'fs';
import * as path from 'path';

export type AiProviderType = 'ollama' | 'llama-cpp' | 'none';

export interface AiProviderConfig {
    provider: AiProviderType;
    modelName: string;
    modelPath?: string; // For llama-cpp
    host?: string;       // For ollama (default: localhost:11434)
    temperature: number;
    maxTokens: number;
}

const CONFIG_FILE = path.join(process.cwd(), '.ai-provider.json');

const defaultConfig: AiProviderConfig = {
    provider: 'ollama',
    modelName: 'qwen2.5:1.5b',
    host: 'http://localhost:11434',
    temperature: 0.1,
    maxTokens: 500,
};

export function loadAiProviderConfig(): AiProviderConfig {
    try {
        if (fs.existsSync(CONFIG_FILE)) {
            const content = fs.readFileSync(CONFIG_FILE, 'utf-8');
            return { ...defaultConfig, ...JSON.parse(content) };
        }
    } catch {
        // ignore
    }
    return { ...defaultConfig };
}

export function saveAiProviderConfig(config: Partial<AiProviderConfig>): void {
    const current = loadAiProviderConfig();
    const merged = { ...current, ...config };
    fs.writeFileSync(CONFIG_FILE, JSON.stringify(merged, null, 2), 'utf-8');
}

export function resetAiProviderConfig(): void {
    if (fs.existsSync(CONFIG_FILE)) {
        fs.unlinkSync(CONFIG_FILE);
    }
}

export const PROVIDER_INFO: Record<string, { description: string; status: 'stable' | 'experimental' | 'planned' }> = {
    'ollama': {
        description: 'Ollama local LLM runner (requires `ollama` CLI)',
        status: 'stable',
    },
    'llama-cpp': {
        description: 'Direct llama.cpp via node-llama-cpp (loads GGUF files)',
        status: 'experimental',
    },
    'none': {
        description: 'No AI provider (clustering only, no AI analysis)',
        status: 'stable',
    },
};
