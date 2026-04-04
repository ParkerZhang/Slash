import { AiPromptState } from '../core/types.js';

interface PromptHistoryEntry {
    timestamp: string;
    model: string;
    prompt: string;
    result: string;
    purpose: string;
}

let lastPrompt: AiPromptState | null = null;
let lastPromptRaw: string | null = null;
let lastResult: string | null = null;
let promptHistory: PromptHistoryEntry[] = [];

const MAX_HISTORY = 10;

export function setLastPrompt(prompt: AiPromptState | null, rawPrompt: string | null): void {
    lastPrompt = prompt;
    lastPromptRaw = rawPrompt;
}

export function setLastResult(result: string): void {
    lastResult = result;

    if (lastPromptRaw) {
        promptHistory.push({
            timestamp: new Date().toISOString(),
            model: lastPrompt?.title || 'unknown',
            prompt: lastPromptRaw,
            result,
            purpose: lastPrompt?.title || 'AI Analysis',
        });

        if (promptHistory.length > MAX_HISTORY) {
            promptHistory = promptHistory.slice(-MAX_HISTORY);
        }
    }
}

export function getLastPrompt(): AiPromptState | null {
    return lastPrompt;
}

export function getLastPromptRaw(): string | null {
    return lastPromptRaw;
}

export function getLastResult(): string | null {
    return lastResult;
}

export function getPromptHistory(): PromptHistoryEntry[] {
    return promptHistory;
}

export function getPromptHistoryEntry(index: number): PromptHistoryEntry | undefined {
    if (index < 0 || index >= promptHistory.length) return undefined;
    return promptHistory[index];
}

export function clearPromptHistory(): void {
    promptHistory = [];
    lastPrompt = null;
    lastPromptRaw = null;
    lastResult = null;
}
