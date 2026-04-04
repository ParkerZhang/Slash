import { JsonSchema } from './schema.js';

export interface ElasticSearchHit {
    _index: string;
    _id: string;
    _score: number;
    _source: Record<string, unknown>;
}

export interface ElasticSearchResult {
    took: number;
    timed_out: boolean;
    hits: {
        total: { value: number; relation: string };
        max_score: number;
        hits: ElasticSearchHit[];
    };
}

export interface LoadedFile {
    id: string;
    name: string;
    data: ElasticSearchResult;
    fileFormat?: 'csv' | 'json' | 'generated';
    schema?: JsonSchema;
    subSchemas?: Record<string, JsonSchema>;
    selectedSubSchemaPaths?: string[];
    keyField?: string;
    sortField?: string;
    sortOrder?: 'asc' | 'desc';
}

export type ViewMode = 'main' | 'preview' | 'detail' | 'nested' | 'compare' | 'match' | 'keyfield' | 'schema';

export interface CommandResult {
    output?: string;
    action?: 'EXIT' | 'CLEAR' | 'VIEW_CHANGE' | 'REFRESH';
    viewMode?: ViewMode;
    viewData?: any;
}

export interface AiPromptState {
    title: string;
    system: string;
    user: string;
    context?: string;
    fullPrompt?: string;
}

export interface Workspace {
    getLoadedFiles: () => Map<string, LoadedFile>;
    addFile: (id: string, name: string, data: ElasticSearchResult) => void;
    updateFile: (id: string, updates: Partial<LoadedFile>) => void;
    removeFile: (id: string) => void;
    getModel: () => string;
    setModel: (model: string) => void;
    setViewMode: (mode: ViewMode, data?: any) => void;
    getHistory: () => string[];
    addHistory: (cmd: string) => void;
    clearHistory: () => void;
    getAiChatHistory?: () => string[];
    setAiChatHistory?: (history: string[]) => void;
    getAiPromptState?: () => AiPromptState | null;
    setAiPromptState?: (state: AiPromptState | null) => void;
    setCommandOutput?: (lines: string[]) => void;
    exit: () => void;
}
