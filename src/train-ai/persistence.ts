import * as fs from 'fs';
import * as path from 'path';
import { WorkflowState } from './types.js';

const TRAIN_AI_DIR = path.join(process.cwd(), '.train-ai');
const CACHE_DIR = path.join(TRAIN_AI_DIR, 'cache');
const OUTPUT_DIR = path.join(TRAIN_AI_DIR, 'output');

function ensureDirectories(): void {
    if (!fs.existsSync(TRAIN_AI_DIR)) fs.mkdirSync(TRAIN_AI_DIR, { recursive: true });
    if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR, { recursive: true });
    if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

export function generateWorkflowId(): string {
    const now = new Date();
    const dateStr = now.toISOString().replace(/[:.]/g, '-').slice(0, 19);
    return `train-ai-${dateStr}`;
}

export async function saveState(state: WorkflowState): Promise<void> {
    ensureDirectories();
    state.lastUpdated = new Date().toISOString();
    const stateFile = path.join(TRAIN_AI_DIR, `${state.workflowId}.json`);
    await fs.promises.writeFile(stateFile, JSON.stringify(state, null, 2));
}

export async function loadState(workflowId: string): Promise<WorkflowState | null> {
    ensureDirectories();
    const stateFile = path.join(TRAIN_AI_DIR, `${workflowId}.json`);
    if (!fs.existsSync(stateFile)) return null;
    const content = await fs.promises.readFile(stateFile, 'utf-8');
    return JSON.parse(content);
}

export async function listWorkflows(): Promise<WorkflowState[]> {
    ensureDirectories();
    const files = await fs.promises.readdir(TRAIN_AI_DIR);
    const workflows: WorkflowState[] = [];
    for (const file of files) {
        if (file.endsWith('.json')) {
            const content = await fs.promises.readFile(path.join(TRAIN_AI_DIR, file), 'utf-8');
            workflows.push(JSON.parse(content));
        }
    }
    return workflows.sort((a, b) => b.lastUpdated.localeCompare(a.lastUpdated));
}

export async function cacheStepData(workflowId: string, step: number, key: string, data: any): Promise<void> {
    ensureDirectories();
    const cacheFile = path.join(CACHE_DIR, `${workflowId}-step${step}-${key}.json`);
    await fs.promises.writeFile(cacheFile, JSON.stringify(data, null, 2));
}

export async function loadCachedStepData(workflowId: string, step: number, key: string): Promise<any | null> {
    const cacheFile = path.join(CACHE_DIR, `${workflowId}-step${step}-${key}.json`);
    if (!fs.existsSync(cacheFile)) return null;
    const content = await fs.promises.readFile(cacheFile, 'utf-8');
    return JSON.parse(content);
}

export async function saveOutput(workflowId: string, filename: string, data: any): Promise<void> {
    ensureDirectories();
    const outputFile = path.join(OUTPUT_DIR, `${workflowId}-${filename}`);
    await fs.promises.writeFile(outputFile, JSON.stringify(data, null, 2));
}

export async function deleteWorkflow(workflowId: string): Promise<void> {
    ensureDirectories();
    const stateFile = path.join(TRAIN_AI_DIR, `${workflowId}.json`);
    if (fs.existsSync(stateFile)) await fs.promises.unlink(stateFile);

    // Clean up cache files
    const cacheFiles = await fs.promises.readdir(CACHE_DIR);
    for (const file of cacheFiles) {
        if (file.startsWith(workflowId)) {
            await fs.promises.unlink(path.join(CACHE_DIR, file));
        }
    }

    // Clean up output files
    const outputFiles = await fs.promises.readdir(OUTPUT_DIR);
    for (const file of outputFiles) {
        if (file.startsWith(workflowId)) {
            await fs.promises.unlink(path.join(OUTPUT_DIR, file));
        }
    }
}
