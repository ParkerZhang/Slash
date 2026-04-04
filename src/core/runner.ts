import { createRegisteredCommandRegistry } from '../commands/index.js';
import { Workspace, LoadedFile, ViewMode, ElasticSearchResult, AiPromptState } from './types.js';

// Simple in-memory workspace for CLI/Test usage
export class MemoryWorkspace implements Workspace {
    loadedFiles: Map<string, LoadedFile> = new Map();
    history: string[] = [];
    aiChatHistory: string[] = [];
    aiPromptState: AiPromptState | null = null;
    viewMode: ViewMode = 'main';
    model = 'default';
    exited = false;

    getLoadedFiles() { return this.loadedFiles; }
    addFile(id: string, name: string, data: ElasticSearchResult) {
        this.loadedFiles.set(id, { id, name, data, fileFormat: 'generated' });
    }
    updateFile(id: string, updates: Partial<LoadedFile>) {
        const file = this.loadedFiles.get(id);
        if (file) Object.assign(file, updates);
    }
    removeFile(id: string) { this.loadedFiles.delete(id); }
    getModel() { return this.model; }
    setModel(model: string) { this.model = model; }
    setViewMode(mode: ViewMode) { this.viewMode = mode; }
    getHistory() { return this.history; }
    addHistory(cmd: string) { this.history.push(cmd); }
    clearHistory() { this.history = []; }
    getAiChatHistory() { return this.aiChatHistory; }
    setAiChatHistory(history: string[]) { this.aiChatHistory = history; }
    getAiPromptState() { return this.aiPromptState; }
    setAiPromptState(state: AiPromptState | null) { this.aiPromptState = state; }
    setCommandOutput(_lines: string[]) {}
    exit() { this.exited = true; }
}

async function run() {
    const registry = createRegisteredCommandRegistry();
    const workspace = new MemoryWorkspace();
    
    const args = process.argv.slice(2);
    if (args.length === 0) {
        console.log("Usage: tui-runner <command> [args]");
        process.exit(0);
    }

    const input = args.join(' ');
    const result = await registry.execute(input, workspace);
    
    if (result.output) {
        console.log(result.output);
    }
    
    if (result.action === 'EXIT') {
        process.exit(0);
    }
}

// Only run if this is the main module
if (import.meta.url === `file://${process.argv[1]}` || process.argv[1]?.endsWith('runner.ts')) {
    run().catch(console.error);
}
