import * as fs from 'fs';
import * as path from 'path';

const HISTORY_FILE = path.join(process.cwd(), '.cli_history.json');

export interface Command {
    name: string;
    description: string;
    execute: (args: string) => void | Promise<void>;
}

interface LoadedFileInfo {
    id: string;
    name: string;
    fileFormat?: 'csv' | 'json' | 'generated';
    keyField?: string;
}

interface CommandHistory {
    commands: string[];
    loadedFiles: LoadedFileInfo[];
    selectedModel?: string;
}

export function loadHistory(): CommandHistory {
    try {
        if (fs.existsSync(HISTORY_FILE)) {
            const data = fs.readFileSync(HISTORY_FILE, 'utf-8');
            if (!data.trim()) {
                return { commands: [], loadedFiles: [] };
            }
            const parsed = JSON.parse(data) as CommandHistory;
            return {
                commands: parsed.commands || [],
                loadedFiles: parsed.loadedFiles || [],
                selectedModel: parsed.selectedModel,
            };
        }
        return { commands: [], loadedFiles: [] };
    } catch (error) {
        return { commands: [], loadedFiles: [] };
    }
}

export function saveHistory(history: CommandHistory): void {
    try {
        fs.writeFileSync(HISTORY_FILE, JSON.stringify(history, null, 2), 'utf-8');
    } catch (error) {
        console.error("Error saving history:", error);
    }
}

export function addCommand(command: string): void {
    const history = loadHistory();
    history.commands.push(command);
    saveHistory(history);
}

export function getHistory(): string[] {
    return loadHistory().commands;
}

export function getLoadedFiles(): LoadedFileInfo[] {
    return loadHistory().loadedFiles;
}

export function addLoadedFile(id: string, name: string, keyField?: string, fileFormat?: 'csv' | 'json' | 'generated'): void {
    const history = loadHistory();
    const existing = history.loadedFiles.findIndex(f => f.id === id);
    if (existing >= 0) {
        history.loadedFiles[existing] = {
            id,
            name,
            keyField: keyField || history.loadedFiles[existing].keyField,
            fileFormat: fileFormat || history.loadedFiles[existing].fileFormat,
        };
    } else {
        history.loadedFiles.push({ id, name, keyField, fileFormat });
    }
    saveHistory(history);
}

export function updateKeyField(id: string, keyField: string): void {
    const history = loadHistory();
    const file = history.loadedFiles.find(f => f.id === id);
    if (file) {
        file.keyField = keyField;
        saveHistory(history);
    }
}

export function getSelectedModel(): string | undefined {
    return loadHistory().selectedModel;
}

export function setSelectedModel(model: string): void {
    const history = loadHistory();
    history.selectedModel = model;
    saveHistory(history);
}
