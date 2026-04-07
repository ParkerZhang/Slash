import { createRegisteredCommandRegistry } from './commands/index.js';
import { Workspace, LoadedFile, ViewMode, ElasticSearchResult, AiPromptState } from './core/types.js';
import * as fs from 'fs';
import * as path from 'path';

export class CliWorkspace implements Workspace {
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
    setCommandOutput(lines: string[]) {
        process.stderr.write(lines.join('\n') + '\n');
    }
    exit() { this.exited = true; }
}

async function runCli() {
    const registry = createRegisteredCommandRegistry();
    const workspace = new CliWorkspace();
    
    const args = process.argv.slice(2);
    
    if (args.length === 0) {
        console.log("Usage: npm run cli -- \"/load f1 data/file1.csv /load f2 data/file2.csv /diff f1 f2\"");
        console.log("Multiple commands can be chained by starting each with /");
        process.exit(0);
    }

    const input = args.join(' ');
    const commandStrings = input.split(/(?=\/)/).map(cmd => cmd.trim()).filter(Boolean);

    for (const cmdStr of commandStrings) {
        console.log(`\n> Executing: ${cmdStr}`);
        const result = await registry.execute(cmdStr, workspace);
        
        if (result.output) {
            console.log(result.output);
        }
        
        if (result.action === 'EXIT' || workspace.exited) {
            break;
        }
    }
}

const isMain = import.meta.url === `file://${process.argv[1]}` || process.argv[1]?.endsWith('cli.ts');

if (isMain) {
    runCli().catch(err => {
        console.error("CLI Error:", err);
        process.exit(1);
    });
}
