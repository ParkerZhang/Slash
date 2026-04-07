import * as fs from 'fs';
import * as path from 'path';
import { Workspace, CommandResult, ElasticSearchResult, ElasticSearchHit, LoadedFile } from '../core/types.js';
import { SlashCommand, globalRegistry, CommandRegistry } from './commandRegistry.js';
import { diffFiles } from '../core/diff.js';
import { addLoadedFile, setSelectedModel } from '../core/historyManager.js';
import { getAvailableModels, inferSchemaForDataStreaming, runPromptStreaming } from '../ai/modelManager.js';
import { getLocalEmbeddings } from '../core/localEmbeddings.js';
import { kMeans, kMeansCosine, formatClusterSummary, findDominantTermsForCluster, buildMultiDimensionalFeatures, evaluateClusters, formatMentalModelVisualization, formatEnhancedClusterSummary, twoStageClustering, buildClusterProfile } from '../core/clustering.js';
import { AiProviderCommand, ShowPromptCommand, executeAiTrain, executeAiDebug } from '../train-ai/command.js';
import { ClusterCommand } from './clusterCommand.js';
import alasql from 'alasql';
import { buildSelectedSchema, extractSubSchemas, JsonSchema } from '../core/schema.js';

const DATA_DIR = path.join(process.cwd(), 'data');

const resolveDataPath = (fileName: string): string => {
    if (path.isAbsolute(fileName)) return fileName;
    if (fileName.startsWith('data/') || fileName.startsWith('./data/')) return path.resolve(fileName);
    return path.join(DATA_DIR, fileName);
};

// Auto-detect keyfield from record fields
function autoDetectKeyField(fields: string[]): string | undefined {
    return fields.find(f => ['isin', 'id', 'code', 'symbol'].includes(f.toLowerCase()));
}

function renderFileInOriginalFormat(file: LoadedFile, limit: number = 5): string {
    const hits = file.data.hits.hits.slice(0, limit).map((hit) => hit._source);

    if (file.fileFormat === 'csv') {
        const headers = hits.length > 0 ? Object.keys(hits[0]) : [];
        const lines = [
            headers.join('|'),
            ...hits.map((row) => headers.map((header) => String(row[header] ?? '')).join('|')),
        ].filter((line) => line.length > 0);
        return lines.join('\n');
    }

    return JSON.stringify(
        {
            took: file.data.took,
            timed_out: file.data.timed_out,
            hits: {
                total: file.data.hits.total,
                hits: file.data.hits.hits.slice(0, limit),
            },
        },
        null,
        2,
    );
}

function buildFilePromptContext(file: LoadedFile, limit: number = 5, label?: string): string {
    return [
        label ? `${label}` : 'File Context',
        `File id: ${file.id}`,
        `File name: ${file.name}`,
        `File format: ${file.fileFormat || 'generated'}`,
        `Key field: ${file.keyField || '(none)'}`,
        file.selectedSubSchemaPaths?.length ? `Selected sub-schemas: ${file.selectedSubSchemaPaths.join(', ')}` : 'Selected sub-schemas: (none)',
        'Raw file view:',
        renderFileInOriginalFormat(file, limit),
    ].join('\n');
}

function getComparisonKeyFields(file: LoadedFile): string[] {
    if (!file || file.data.hits.hits.length === 0) {
        return file.keyField ? [file.keyField] : [];
    }

    const headers = Object.keys(file.data.hits.hits[0]._source);
    const autoKeys = headers.filter((h) => ['isin', 'currency', 'exchange_code'].includes(h.toLowerCase()));
    return autoKeys.length > 0 ? autoKeys : (file.keyField ? [file.keyField] : []);
}

// --- Cat Command ---
export const CatCommand: SlashCommand = {
    name: '/cat',
    description: 'Print file contents to terminal (Usage: /cat <id> [limit])',
    execute: (args, workspace) => {
        const parts = args.trim().split(/\s+/).filter(Boolean);
        if (parts.length < 1) return { output: 'Usage: /cat <id> [limit]' };
        
        const fileId = parts[0];
        const limit = parts[1] ? parseInt(parts[1], 10) : 10;
        
        const loadedFiles = workspace.getLoadedFiles();
        const file = loadedFiles.get(fileId);
        if (!file) return { output: `File '${fileId}' not loaded` };
        
        const hits = file.data.hits.hits.slice(0, limit).map(h => h._source);
        if (hits.length === 0) return { output: 'File is empty' };
        
        const headers = Object.keys(hits[0]);
        const lines = [
            headers.join('|'),
            ...hits.map(row => headers.map(h => String(row[h] ?? '')).join('|'))
        ];
        
        return { 
            output: [
                `--- ${fileId} (${hits.length}/${file.data.hits.total.value} records) ---`,
                ...lines,
                file.data.hits.total.value > limit ? `... (${file.data.hits.total.value - limit} more records)` : ''
            ].filter(Boolean).join('\n')
        };
    }
};

// --- Help Command ---
export const HelpCommand: SlashCommand = {
    name: '/help',
    description: 'Show available commands',
    execute: (_args, _workspace, registry) => {
        const helpText = registry.getAllCommands()
            .map(cmd => `${cmd.name}: ${cmd.description}`)
            .join('\n');
        return { output: `Available commands:\n${helpText}` };
    }
};

// --- Clear Command ---
export const ClearCommand: SlashCommand = {
    name: '/clear',
    description: 'Clear command history',
    execute: (_args, workspace) => {
        workspace.clearHistory();
        return { output: 'History cleared', action: 'CLEAR' };
    }
};

export const RefreshCommand: SlashCommand = {
    name: '/refresh',
    description: 'Reload command modules without restarting the TUI',
    execute: () => {
        return { output: 'Refreshing command registry...', action: 'REFRESH' };
    }
};

// --- Model Command ---
export const ModelCommand: SlashCommand = {
    name: '/model',
    description: 'Show or set the current model (Usage: /model [name])',
    execute: (args, workspace) => {
        const nextModel = args.trim();
        if (!nextModel) {
            const availableModels = getAvailableModels();
            const currentModel = workspace.getModel();
            const lines = [`Current model: ${currentModel}`];

            if (availableModels.length > 0) {
                lines.push('Available models: ' + availableModels.join(', '));
            } else {
                lines.push('Available models: unavailable');
            }

            return { output: lines.join('\n') };
        }

        workspace.setModel(nextModel);
        setSelectedModel(nextModel);
        return { output: `Model set to: ${nextModel}` };
    },
    suggestArgs: (input) => {
        const query = input.trim().toLowerCase();
        const models = getAvailableModels();
        if (!query) {
            return models;
        }

        const exactMatch = models.find((model) => model.toLowerCase() === query);
        if (exactMatch) {
            return models;
        }

        return models.filter((model) => model.toLowerCase().startsWith(query));
    },
};

// --- Load Command (auto-detects CSV/JSON, sets keyfield) ---
export const LoadCommand: SlashCommand = {
    name: '/load',
    description: 'Load CSV or JSON file, auto-detects format and sets keyfield (Usage: /load <id> <file>)',
    execute: (args, workspace) => {
        const parts = args.split(' ');
        if (parts.length < 2) return { output: 'Usage: /load <id> <filepath>' };
        const fileId = parts[0];
        const fileName = parts.slice(1).join(' ');

        try {
            const content = fs.readFileSync(resolveDataPath(fileName), 'utf-8');
            const ext = fileName.toLowerCase().split('.').pop();

            if (ext === 'csv') {
                // Load CSV - auto-detect separator
                const lines = content.trim().split('\n');
                if (lines.length < 2) return { output: 'CSV must have header and at least one row' };
                
                // Auto-detect separator: comma or pipe
                const firstLine = lines[0];
                const separator = firstLine.includes('|') ? '|' : ',';
                const headers = lines[0].split(separator).map((h: string) => h.trim());
                const hits: ElasticSearchHit[] = lines.slice(1).map((line: string, idx: number) => {
                    const values = line.split(separator);
                    const source: Record<string, unknown> = {};
                    headers.forEach((header: string, i: number) => {
                        let rawVal = values[i]?.trim() || '';
                        let parsedVal: unknown = rawVal;
                        if (rawVal === 'true') parsedVal = true;
                        else if (rawVal === 'false') parsedVal = false;
                        else if (!isNaN(Number(rawVal)) && rawVal !== '') parsedVal = Number(rawVal);
                        source[header] = parsedVal;
                    });
                    return { _index: 'csv', _id: String(idx + 1), _score: 1.0, _source: source };
                });

                const data: ElasticSearchResult = {
                    took: 0, timed_out: false,
                    hits: { total: { value: hits.length, relation: 'eq' }, max_score: 1.0, hits }
                };

                workspace.addFile(fileId, fileName, data);
                workspace.updateFile(fileId, { fileFormat: 'csv' });
                addLoadedFile(fileId, fileName, undefined, 'csv');

                // Auto-detect and set keyfield
                const keyField = autoDetectKeyField(headers);
                if (keyField) {
                    workspace.updateFile(fileId, { keyField });
                    addLoadedFile(fileId, fileName, keyField, 'csv');
                    return { output: `Loaded ${fileName} as '${fileId}' (${hits.length} rows) | Key field: ${keyField}` };
                }
                return { output: `Loaded ${fileName} as '${fileId}' (${hits.length} rows)` };

            } else {
                // Load JSON
                const data = JSON.parse(content) as ElasticSearchResult;
                workspace.addFile(fileId, fileName, data);
                workspace.updateFile(fileId, { fileFormat: 'json' });
                addLoadedFile(fileId, fileName, undefined, 'json');

                // Auto-detect and set keyfield
                if (data.hits.hits.length > 0) {
                    const fields = Object.keys(data.hits.hits[0]._source);
                    const keyField = autoDetectKeyField(fields);
                    if (keyField) {
                        workspace.updateFile(fileId, { keyField });
                        addLoadedFile(fileId, fileName, keyField, 'json');
                        return { output: `Loaded ${fileName} as '${fileId}' (${data.hits.hits.length} hits) | Key field: ${keyField}` };
                    }
                }
                return { output: `Loaded ${fileName} as '${fileId}' (${data.hits.hits.length} hits)` };
            }
        } catch (error) {
            return { output: `Error: ${error instanceof Error ? error.message : String(error)}` };
        }
    }
};

// --- LoadJson Command ---
export const LoadJsonCommand: SlashCommand = {
    name: '/loadJson',
    description: 'Load JSON file (Usage: /loadJson <id> <file>)',
    execute: (args, workspace) => {
        const parts = args.split(' ');
        if (parts.length < 2) return { output: 'Usage: /loadJson <id> <filepath>' };
        const fileId = parts[0];
        const fileName = parts.slice(1).join(' ');

        try {
            const content = fs.readFileSync(resolveDataPath(fileName), 'utf-8');
            const data = JSON.parse(content) as ElasticSearchResult;
            workspace.addFile(fileId, fileName, data);
            workspace.updateFile(fileId, { fileFormat: 'json' });
            addLoadedFile(fileId, fileName, undefined, 'json');
            return { output: `Loaded ${fileName} as '${fileId}' (${data.hits.hits.length} hits)` };
        } catch (error) {
            return { output: `Error: ${error instanceof Error ? error.message : String(error)}` };
        }
    }
};

// --- LoadCsv Command ---
export const LoadCsvCommand: SlashCommand = {
    name: '/loadCsv',
    description: 'Load CSV (Usage: /loadCsv <id> <file> [sep])',
    execute: (args, workspace) => {
        const parts = args.trim().split(' ');
        if (parts.length < 2) return { output: 'Usage: /loadCsv <id> <filepath> [sep]' };
        const fileId = parts[0];
        const fileName = parts.slice(1, -1).join(' ') || parts[1];
        const separator = parts.length > 2 ? parts[parts.length - 1] : '|';
        
        if (separator.length !== 1) return { output: 'Separator must be a single character' };
        
        try {
            const content = fs.readFileSync(resolveDataPath(fileName), 'utf-8');
            const lines = content.trim().split('\n');
            if (lines.length < 2) return { output: 'CSV must have header and at least one row' };
            
            const headers = lines[0].split(separator).map((h: string) => h.trim());
            const hits: ElasticSearchHit[] = lines.slice(1).map((line: string, idx: number) => {
                const values = line.split(separator);
                const source: Record<string, unknown> = {};
                headers.forEach((header: string, i: number) => {
                    let rawVal = values[i]?.trim() || '';
                    let parsedVal: unknown = rawVal;
                    if (rawVal === 'true') parsedVal = true;
                    else if (rawVal === 'false') parsedVal = false;
                    else if (!isNaN(Number(rawVal)) && rawVal !== '') parsedVal = Number(rawVal);
                    source[header] = parsedVal;
                });
                return {
                    _index: 'csv',
                    _id: String(idx + 1),
                    _score: 1.0,
                    _source: source
                };
            });
            
            const data: ElasticSearchResult = {
                took: 0,
                timed_out: false,
                hits: {
                    total: { value: hits.length, relation: 'eq' },
                    max_score: 1.0,
                    hits: hits
                }
            };
            
            workspace.addFile(fileId, fileName, data);
            workspace.updateFile(fileId, { fileFormat: 'csv' });
            addLoadedFile(fileId, fileName, undefined, 'csv');
            return { output: `Loaded ${fileName} as '${fileId}' (${hits.length} rows)` };
        } catch (error) {
            return { output: `Error: ${error instanceof Error ? error.message : String(error)}` };
        }
    }
};

export const AnalyzeCommand: SlashCommand = {
    name: '/analyze',
    description: 'Load request/response CSVs and diff them (Usage: /analyze <request.csv> <response.csv> [key])',
    execute: (args, workspace) => {
        const parts = args.trim().split(/\s+/).filter(Boolean);
        if (parts.length < 2) {
            return { output: 'Usage: /analyze <request.csv> <response.csv> [key]' };
        }

        const requestFileName = parts[0];
        const responseFileName = parts[1];
        const keyField = parts[2];
        const requestId = 'request';
        const responseId = 'response';

        const loadRequest = LoadCsvCommand.execute(`${requestId} ${requestFileName}`, workspace, {} as CommandRegistry);
        if (loadRequest.output?.startsWith('Error:') || loadRequest.output?.startsWith('Usage:')) {
            return loadRequest;
        }

        const loadResponse = LoadCsvCommand.execute(`${responseId} ${responseFileName}`, workspace, {} as CommandRegistry);
        if (loadResponse.output?.startsWith('Error:') || loadResponse.output?.startsWith('Usage:')) {
            return loadResponse;
        }

        if (keyField) {
            workspace.updateFile(requestId, { keyField });
            workspace.updateFile(responseId, { keyField });
        }

        const diffResult = DiffCommand.execute(`${requestId} ${responseId}`, workspace, {} as CommandRegistry);
        if (diffResult.output?.startsWith('Both files need keyfield') || diffResult.output?.startsWith("File '")) {
            return diffResult;
        }

        const diffId = `${requestId}-diff-${responseId}`;
        if (workspace.getLoadedFiles().has(diffId)) {
            workspace.setViewMode('preview', { selectedFileId: diffId });
        }

        return {
            output: [
                loadRequest.output,
                loadResponse.output,
                keyField ? `Set key field '${keyField}' on '${requestId}' and '${responseId}'` : '',
                diffResult.output,
                `Opened analysis result '${diffId}'`,
            ].filter(Boolean).join('\n'),
            action: 'VIEW_CHANGE',
        };
    },
};

// --- Preview Command ---
export const PreviewCommand: SlashCommand = {
    name: '/preview',
    description: 'Preview loaded file (Usage: /preview <id>)',
    execute: (args, workspace) => {
        const loadedFiles = workspace.getLoadedFiles();
        if (!args.trim()) {
            const ids = Array.from(loadedFiles.keys());
            if (ids.length === 0) return { output: 'No files loaded' };
            return { output: `Loaded: ${ids.join(', ')}\nUsage: /preview <id>` };
        }
        const fileId = args.trim();
        if (!loadedFiles.has(fileId)) return { output: `File '${fileId}' not loaded` };
        
        workspace.setViewMode('preview', { selectedFileId: fileId });
        return { output: '', action: 'VIEW_CHANGE' };
    }
};

// --- Files Command ---
export const FilesCommand: SlashCommand = {
    name: '/files',
    description: 'List loaded files',
    execute: (_args, workspace) => {
        const loadedFiles = workspace.getLoadedFiles();
        const ids = Array.from(loadedFiles.keys());
        if (ids.length === 0) return { output: 'No files loaded' };
        return { output: "Loaded: " + ids.map((id: string) => id + ": " + loadedFiles.get(id)?.name).join(', ') };
    }
};

// --- Compare Command ---
export const CompareCommand: SlashCommand = {
    name: '/compare',
    description: 'Compare two files (Usage: /compare <id1> <id2>)',
    execute: (args, workspace) => {
        const parts = args.trim().split(' ');
        if (parts.length < 2) return { output: 'Usage: /compare <id1> <id2>' };
        const id1 = parts[0];
        const id2 = parts[1];
        
        const loadedFiles = workspace.getLoadedFiles();
        const fileA = loadedFiles.get(id1);
        const fileB = loadedFiles.get(id2);
        
        if (!fileA) return { output: `File '${id1}' not loaded` };
        if (!fileB) return { output: `File '${id2}' not loaded` };
        
        workspace.setViewMode('compare', { fileA, fileB, selectedIndex: 0 });
        return { output: '', action: 'VIEW_CHANGE' };
    }
};

// --- Match Command ---
export const MatchCommand: SlashCommand = {
    name: '/match',
    description: 'Match request vs response (Usage: /match <reqId> <respId>)',
    execute: (args, workspace) => {
        const parts = args.trim().split(' ');
        if (parts.length < 2) return { output: 'Usage: /match <requestId> <responseId>' };
        const reqId = parts[0];
        const respId = parts[1];
        
        const loadedFiles = workspace.getLoadedFiles();
        const reqFile = loadedFiles.get(reqId);
        const respFile = loadedFiles.get(respId);
        
        if (!reqFile) return { output: `File '${reqId}' not loaded` };
        if (!respFile) return { output: `File '${respId}' not loaded` };
        
        const reqHits = reqFile.data.hits.hits;
        const respHits = respFile.data.hits.hits;
        
        const reqHeaders = reqHits.length > 0 ? Object.keys(reqHits[0]._source).sort() : [];
        const keyFields = reqHeaders.filter((h: string) => ['isin', 'currency', 'exchange_code'].includes(h.toLowerCase()));
        
        if (keyFields.length === 0) return { output: 'Request file must have ISIN, Currency, or ExchangeCode fields' };
        
        const getKey = (source: Record<string, unknown>): string => {
            return keyFields.map((f: string) => String(source[f] || '')).join('~');
        };
        
        const respMap = new Map<string, Record<string, unknown>>();
        respHits.forEach((hit: ElasticSearchHit) => {
            const key = getKey(hit._source);
            respMap.set(key, hit._source);
        });
        
        const matches = reqHits.map((reqHit: ElasticSearchHit) => {
            const key = getKey(reqHit._source);
            const respData = respMap.get(key);
            return {
                request: reqHit._source,
                response: respData || null,
                key: key,
                matched: !!respData
            };
        });
        
        workspace.setViewMode('match', {
            fileA: reqFile,
            fileB: respFile,
            selectedIndex: 0,
            isMatchResult: true,
            matchData: matches
        });
        return { output: '', action: 'VIEW_CHANGE' };
    }
};

// --- KeyField Command ---
export const KeyFieldCommand: SlashCommand = {
    name: '/keyfield',
    description: 'Set key field (Usage: /keyfield <id>)',
    execute: (args, workspace) => {
        const fileId = args.trim();
        if (!fileId) return { output: 'Usage: /keyfield <id>' };
        const loadedFiles = workspace.getLoadedFiles();
        const file = loadedFiles.get(fileId);
        if (!file) return { output: `File '${fileId}' not loaded` };
        
        workspace.setViewMode('keyfield', { selectedFileId: fileId });
        return { output: '', action: 'VIEW_CHANGE' };
    }
};

// --- Sort Command ---
export const SortCommand: SlashCommand = {
    name: '/sort',
    description: 'Sort file (Usage: /sort <id> [field] [asc|desc])',
    execute: (args, workspace) => {
        const parts = args.trim().split(' ');
        if (parts.length < 1) return { output: 'Usage: /sort <id> [field] [asc|desc]' };
        const fileId = parts[0];
        
        const loadedFiles = workspace.getLoadedFiles();
        const file = loadedFiles.get(fileId);
        if (!file) return { output: `File '${fileId}' not loaded` };
        
        const field = parts[1] || file.keyField;
        if (!field) return { output: 'No keyfield set. Usage: /sort <id> [field] [asc|desc]' };
        
        let order: 'asc' | 'desc' = 'asc';
        const lastPart = parts[parts.length - 1];
        if (lastPart === 'asc' || lastPart === 'desc') {
            order = lastPart;
        } else if (lastPart !== field && parts.length > 2) {
            return { output: 'Invalid order. Use asc or desc' };
        }
        
        const hit = file.data.hits.hits[0];
        if (!hit) return { output: 'File is empty' };
        const fields = Object.keys(hit._source).sort();
        if (!fields.includes(field)) return { output: `Field '${field}' not found. Available: ${fields.join(', ')}` };
        
        const sortedHits = [...file.data.hits.hits].sort((a, b) => {
            const valA = a._source[field];
            const valB = b._source[field];
            
            if (valA === undefined || valA === null) return order === 'asc' ? 1 : -1;
            if (valB === undefined || valB === null) return order === 'asc' ? -1 : 1;
            
            if (typeof valA === 'number' && typeof valB === 'number') {
                return order === 'asc' ? valA - valB : valB - valA;
            }
            
            const strA = String(valA);
            const strB = String(valB);
            return order === 'asc' ? strA.localeCompare(strB) : strB.localeCompare(strA);
        });
        
        const newData: ElasticSearchResult = {
            ...file.data,
            hits: {
                ...file.data.hits,
                hits: sortedHits
            }
        };
        
        workspace.updateFile(fileId, { data: newData, sortField: field, sortOrder: order });
        return { output: `Sorted ${fileId} by ${field} (${order})` };
    }
};

// --- Minus Command ---
export const MinusCommand: SlashCommand = {
    name: '/minus',
    description: 'Minus files (Usage: /minus <id1> <id2>)',
    execute: (args, workspace) => {
        const parts = args.trim().split(' ');
        if (parts.length < 2) return { output: 'Usage: /minus <id1> <id2>' };
        const id1 = parts[0];
        const id2 = parts[1];
        
        const loadedFiles = workspace.getLoadedFiles();
        const file1 = loadedFiles.get(id1);
        const file2 = loadedFiles.get(id2);
        if (!file1) return { output: `File '${id1}' not loaded` };
        if (!file2) return { output: `File '${id2}' not loaded` };
        
        const getKeyFields = (file: LoadedFile): string[] => {
            if (!file || file.data.hits.hits.length === 0) return [];
            const headers = Object.keys(file.data.hits.hits[0]._source);
            return headers.filter((h: string) => ['isin', 'currency', 'exchange_code'].includes(h.toLowerCase()));
        };
        
        const keyFields1 = getKeyFields(file1);
        const keyFields2 = getKeyFields(file2);
        
        const preferredOrder = ['isin', 'currency', 'exchange_code'];
        const sortByPreferred = (arr: string[]): string[] => {
            return arr.sort((a, b) => {
                const ia = preferredOrder.indexOf(a.toLowerCase());
                const ib = preferredOrder.indexOf(b.toLowerCase());
                return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
            });
        };
        
        let useKeyFields1 = keyFields1.length > 0 ? sortByPreferred(keyFields1) : (file1.keyField ? [file1.keyField] : []);
        let useKeyFields2 = keyFields2.length > 0 ? sortByPreferred(keyFields2) : (file2.keyField ? [file2.keyField] : []);
        
        if (useKeyFields1.length === 0 || useKeyFields2.length === 0) {
            return { output: 'Both files need keyfield set. Use /keyfield <id>' };
        }
        
        const getKey = (source: Record<string, unknown>, keyFields: string[]): string => {
            return keyFields.map((f: string) => String(source[f] || '')).join('~');
        };
        
        const keys2 = new Set(file2.data.hits.hits.map(h => getKey(h._source, useKeyFields2)));
        const minusHits = file1.data.hits.hits.filter(h => !keys2.has(getKey(h._source, useKeyFields1)));
        
        const newData: ElasticSearchResult = {
            took: 0,
            timed_out: false,
            hits: {
                total: { value: minusHits.length, relation: 'eq' },
                max_score: 1.0,
                hits: minusHits
            }
        };
        
        const newId = id1 + '-minus-' + id2;
        workspace.addFile(newId, file1.name + ' - ' + file2.name, newData);
        return { output: `Created '${newId}' with ${minusHits.length} records` };
    }
};

// --- SQL Command ---
export const SqlCommand: SlashCommand = {
    name: '/sql',
    description: 'SQL query (Usage: /sql SELECT * FROM f1)',
    execute: (args, workspace) => {
        const query = args.trim().replace(/^["']|["']$/g, '');
        if (!query) return { output: 'Usage: /sql SELECT * FROM f1' };
        
        const loadedFiles = workspace.getLoadedFiles();
        try {
            alasql('DROP TABLE IF EXISTS _tables');
            alasql('CREATE TABLE _tables (id VARCHAR(255), name VARCHAR(255))');
            
            for (const [id, file] of loadedFiles) {
                const tableData = file.data.hits.hits.map(h => h._source);
                alasql('DROP TABLE IF EXISTS ' + id);
                if (tableData.length > 0) {
                    alasql('CREATE TABLE ' + id + ' (' + Object.keys(tableData[0] || {}).map(c => c + ' VARCHAR(255)').join(', ') + ')');
                    alasql('INSERT INTO ' + id + ' SELECT * FROM ?', [tableData]);
                } else {
                    alasql('CREATE TABLE ' + id + ' (id INT)');
                }
            }
            
            const results = alasql(query);
            
            if (!Array.isArray(results) || results.length === 0) {
                return { output: 'No results found' };
            }
            
            const resultHits: ElasticSearchHit[] = results.map((row: Record<string, unknown>, idx: number) => ({
                _index: 'sql',
                _id: String(idx + 1),
                _score: 1.0,
                _source: row
            }));
            
            const newData: ElasticSearchResult = {
                took: 0,
                timed_out: false,
                hits: {
                    total: { value: resultHits.length, relation: 'eq' },
                    max_score: 1.0,
                    hits: resultHits
                }
            };
            
            const newId = 'sql_result_' + Date.now();
            workspace.addFile(newId, 'SQL Result', newData);
            workspace.setViewMode('preview', { selectedFileId: newId });
            return { output: `Created '${newId}' with ${results.length} records`, action: 'VIEW_CHANGE' };
        } catch (error) {
            return { output: `SQL Error: ${error instanceof Error ? error.message : String(error)}` };
        }
    }
};

export const SchemaCommand: SlashCommand = {
    name: '/schema',
    description: 'Show schema for a loaded file and select sub-schemas (Usage: /schema <id>)',
    execute: (args, workspace) => {
        const fileId = args.trim();
        if (!fileId) return { output: 'Usage: /schema <id>' };

        const loadedFiles = workspace.getLoadedFiles();
        const file = loadedFiles.get(fileId);
        if (!file) return { output: `File '${fileId}' not loaded` };
        if (!file.schema) return { output: `File '${fileId}' has no schema. Run /ai ${fileId} first` };

        workspace.setViewMode('schema', { selectedFileId: fileId });
        return { output: '', action: 'VIEW_CHANGE' };
    },
    suggestArgs: (_input, workspace) => Array.from(workspace.getLoadedFiles().keys()),
};

export const SaveSchemaCommand: SlashCommand = {
    name: '/saveSchema',
    description: 'Save schema to disk (Usage: /saveSchema <id> <filepath> [--selected-only])',
    execute: (args, workspace) => {
        const parts = args.trim().split(/\s+/).filter(Boolean);
        const selectedOnly = parts.includes('--selected-only');
        const cleanedParts = parts.filter((part) => part !== '--selected-only');

        if (cleanedParts.length < 2) {
            return { output: 'Usage: /saveSchema <id> <filepath> [--selected-only]' };
        }

        const fileId = cleanedParts[0];
        const fileName = cleanedParts.slice(1).join(' ');
        const loadedFiles = workspace.getLoadedFiles();
        const file = loadedFiles.get(fileId);

        if (!file) return { output: `File '${fileId}' not loaded` };
        if (!file.schema) return { output: `File '${fileId}' has no schema. Run /ai ${fileId} first` };

        const schemaToSave = selectedOnly
            ? buildSelectedSchema(file.schema, file.subSchemas || {}, file.selectedSubSchemaPaths || [])
            : file.schema;

        try {
            fs.writeFileSync(fileName, JSON.stringify(schemaToSave, null, 2), 'utf-8');
            return {
                output: `Saved ${selectedOnly ? 'selected schema' : 'schema'} for ${fileId} to ${fileName}`,
            };
        } catch (error) {
            return { output: `Error: ${error instanceof Error ? error.message : String(error)}` };
        }
    },
    suggestArgs: (_input, workspace) => Array.from(workspace.getLoadedFiles().keys()),
};

// --- Save Command ---
export const SaveCommand: SlashCommand = {
    name: '/save',
    description: 'Save file (Usage: /save <id> <filepath>) - defaults to ./data/',
    execute: (args, workspace) => {
        const parts = args.trim().split(' ');
        if (parts.length < 2) return { output: 'Usage: /save <id> <filepath>' };
        const fileId = parts[0];
        const fileName = parts.slice(1).join(' ');

        const loadedFiles = workspace.getLoadedFiles();
        const file = loadedFiles.get(fileId);
        if (!file) return { output: `File '${fileId}' not loaded` };

        // Default to ./data/ if no directory specified
        const resolvedPath = !path.isAbsolute(fileName) && !fileName.includes('/') && !fileName.includes('\\')
            ? path.join(DATA_DIR, fileName)
            : fileName;

        try {
            const data = file.data.hits.hits.map(h => h._source);

            // Detect format from file extension
            const ext = resolvedPath.toLowerCase().split('.').pop();

            if (ext === 'csv') {
                // Export as CSV
                if (data.length === 0) {
                    fs.writeFileSync(resolvedPath, '', 'utf-8');
                    return { output: `Saved ${fileId} to ${resolvedPath} (empty)` };
                }
                const headers = Object.keys(data[0]);
                const csvLines = [
                    headers.join(','),
                    ...data.map(row => headers.map(h => {
                        const val = row[h] ?? '';
                        const strVal = String(val);
                        if (strVal.includes(',') || strVal.includes('"')) {
                            return `"${strVal.replace(/"/g, '""')}"`;
                        }
                        return strVal;
                    }).join(','))
                ];
                fs.writeFileSync(resolvedPath, csvLines.join('\n'), 'utf-8');
            } else {
                // Default: JSON
                fs.writeFileSync(resolvedPath, JSON.stringify(data, null, 2), 'utf-8');
            }

            return { output: `Saved ${fileId} (${data.length} records) to ${resolvedPath}` };
        } catch (error) {
            return { output: `Error: ${error instanceof Error ? error.message : String(error)}` };
        }
    }
};

// --- Diff Command (using core logic from diffCommand.ts) ---
export const DiffCommand: SlashCommand = {
    name: '/diff',
    description: 'Diff files - records in f1 missing from f2 (Usage: /diff <id1> <id2>)',
    execute: (args, workspace) => {
        const parts = args.trim().split(' ');
        if (parts.length < 2) return { output: 'Usage: /diff <id1> <id2>' };
        const id1 = parts[0];
        const id2 = parts[1];

        const files = workspace.getLoadedFiles();
        const file1 = files.get(id1);
        const file2 = files.get(id2);

        if (!file1) return { output: `File '${id1}' not loaded` };
        if (!file2) return { output: `File '${id2}' not loaded` };

        const result = diffFiles(file1, file2, id1 + '-diff-' + id2);
        if (typeof result === 'string') {
            return { output: result };
        }

        workspace.addFile(result.newId, file1.name + ' diff ' + file2.name, result.result);
        return { output: `Created '${result.newId}' with ${result.recordCount} records` };
    }
};

// --- Exit Command ---
export const ExitCommand: SlashCommand = {
    name: '/exit',
    description: 'Exit application',
    execute: (_args, workspace) => {
        workspace.exit();
        return { action: 'EXIT' };
    }
};

// --- AI Router Command ---
export const AiCommand: SlashCommand = {
    name: '/ai',
    description: 'AI commands: /ai <file> | /ai analyze <file> "<q>" | /ai train <model> <req> <resp> | /ai debug ...',
    execute: async (args, workspace, registry) => {
        const trimmedArgs = args.trim();

        if (trimmedArgs.toLowerCase().startsWith('train')) {
            return await executeAiTrain(trimmedArgs, workspace);
        }
        if (trimmedArgs.toLowerCase().startsWith('debug')) {
            return await executeAiDebug(trimmedArgs, workspace, registry);
        }
        if (trimmedArgs.toLowerCase().startsWith('analyze')) {
            return await executeAiAnalyze(trimmedArgs, workspace);
        }
        if (trimmedArgs.toLowerCase().startsWith('compare')) {
            return await executeAiCompare(trimmedArgs, workspace);
        }
        return await executeAiSchema(trimmedArgs, workspace);
    },
    suggestArgs: (_input, workspace) => ['analyze', 'train', 'debug', 'compare', ...Array.from(workspace.getLoadedFiles().keys())],
};

async function executeAiAnalyze(
    args: string,
    workspace: Workspace,
): Promise<CommandResult> {
    const analyzeParts = args.split(/\s+/).filter(Boolean);

    if (analyzeParts.length < 3) {
        return { output: 'Usage: /ai analyze <fileId> "<question>"\nExample: /ai analyze data "what are the dominant groups, top 4 clusters"' };
    }

    const fileId = analyzeParts[1];
    const question = analyzeParts.slice(2).join(' ').replace(/^["']|["']$/g, '');
    const loadedFiles = workspace.getLoadedFiles();
    const file = loadedFiles.get(fileId);

    if (!file) return { output: `File '${fileId}' not loaded` };

    const model = workspace.getModel().trim();
    if (!model || model === 'default') {
        return { output: 'Set a model first with /model <name>' };
    }

    const records = file.data.hits.hits.map((hit) => hit._source);
    const kMatch = question.match(/(\d+)\s*(cluster|group|dominance)/i);
    const k = kMatch ? parseInt(kMatch[1], 10) : 4;

    if (records.length < k) {
        return { output: `Not enough records for ${k} clusters. File has ${records.length} records.` };
    }

    try {
        workspace.setCommandOutput?.([
            `Running /ai analyze for '${fileId}'...`,
            `Loading embedding model (local)...`,
        ]);

        // Use local embeddings
        const { embeddings: nameEmbeddings } = await getLocalEmbeddings(records, (current, total) => {
            workspace.setCommandOutput?.([
                `Running /ai analyze for '${fileId}'...`,
                `Generating embeddings: ${current}/${total} records`,
            ]);
        });

        workspace.setCommandOutput?.([
            `Running /ai analyze for '${fileId}'...`,
            `Running multi-dimensional clustering (${k} clusters)...`,
        ]);

        // Use advanced multi-dimensional clustering
        const finalFeatures = buildMultiDimensionalFeatures(records, nameEmbeddings);
        const clusteringResult = kMeans(finalFeatures, k);
        const evaluation = evaluateClusters(clusteringResult.clusters, records);

        // Build clustering summary
        const clusteringSummary = formatEnhancedClusterSummary(clusteringResult.clusters, records, evaluation);
        const visualization = formatMentalModelVisualization(clusteringResult.clusters, records);

        // Save clustering result
        const clusteringFileId = `${fileId}-clustering-${k}`;
        workspace.addFile(clusteringFileId, `${file.name} (${k} clusters)`, {
            took: 0,
            timed_out: false,
            hits: {
                total: { value: records.length, relation: 'eq' },
                max_score: 1.0,
                hits: records.map((record, idx) => ({
                    _index: 'clustering',
                    _id: String(idx + 1),
                    _score: 1.0,
                    _source: {
                        ...record,
                        _cluster: clusteringResult.assignments[idx],
                    },
                })),
            },
        } as any);

        // Build detailed cluster info for AI
        const clusterDetails: string[] = [];
        for (let c = 0; c < clusteringResult.clusters.length; c++) {
            const cluster = clusteringResult.clusters[c];
            const members = cluster.members;
            const profile = buildClusterProfile(cluster, records);
            const sampleRecords = members.slice(0, 5).map(idx => {
                const rec = records[idx];
                const name = rec.name || rec.Name || 'Unknown';
                const country = (rec.country || rec.Country || 'N/A').toString().toUpperCase();
                return `${name} (${country})`;
            });

            clusterDetails.push(`Cluster ${c + 1} (${members.length} records):`);
            clusterDetails.push(`  Top sectors: ${Object.entries(profile.sectorClusters).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([s,n])=>`${s}(${n})`).join(', ')}`);
            clusterDetails.push(`  Top countries: ${Object.entries(profile.geoClusters).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([co,n])=>`${co.toUpperCase()}(${n})`).join(', ')}`);
            clusterDetails.push(`  Sample records: ${sampleRecords.join(', ')}`);
        }

        // Run AI analysis
        let aiAnalysis = '';
        workspace.setCommandOutput?.([
            `Running /ai analyze for '${fileId}'...`,
            `Generating AI analysis with ${model}...`,
        ]);

        const prompt = [
            'You are analyzing multi-dimensional clustering results from a dataset.',
            `User question: "${question}"`,
            '',
            'CLUSTERING SUMMARY:',
            clusteringSummary,
            '',
            'VISUALIZATION:',
            visualization,
            '',
            'DETAILED CLUSTER DATA:',
            clusterDetails.join('\n'),
            '',
            'Based on the clustering data above, answer the user\'s question.',
            'Be specific about what each cluster contains.',
            'Mention the dominant sectors, countries, and record names.',
        ].join('\n');

        try {
            aiAnalysis = await runPromptStreaming(model, prompt, (chunk) => {
                workspace.setCommandOutput?.([
                    `Running AI analysis...`,
                    chunk.split('\n').slice(-3).join('\n'),
                ]);
            });
        } catch (error) {
            aiAnalysis = `AI analysis error: ${error instanceof Error ? error.message : String(error)}`;
        }

        return {
            output: clusteringSummary + '\n\n' + visualization + (aiAnalysis ? `\n\n🤖 AI Analysis:\n${aiAnalysis}` : ''),
        };
    } catch (error) {
        return { output: `AI Analyze Error: ${error instanceof Error ? error.message : String(error)}` };
    }
}

async function executeAiCompare(
    args: string,
    workspace: Workspace,
): Promise<CommandResult> {
    const compareParts = args.split(/\s+/).filter(Boolean);
    if (compareParts.length < 4) {
        return { output: 'Usage: /ai compare <id1> <id2> <question>' };
    }

    const id1 = compareParts[1];
    const id2 = compareParts[2];
    const question = compareParts.slice(3).join(' ').trim();
    const loadedFiles = workspace.getLoadedFiles();
    const file1 = loadedFiles.get(id1);
    const file2 = loadedFiles.get(id2);

    if (!file1) return { output: `File '${id1}' not loaded` };
    if (!file2) return { output: `File '${id2}' not loaded` };

    const model = workspace.getModel().trim();
    if (!model || model === 'default') {
        return { output: 'Set a model first with /model <name>' };
    }

    const diffResult = diffFiles(file1, file2, `${id1}-diff-${id2}`);
    const missingRecords =
        typeof diffResult === 'string'
            ? null
            : diffResult.result.hits.hits.map((hit) => hit._source).slice(0, 10);
    const keyFields1 = getComparisonKeyFields(file1);
    const keyFields2 = getComparisonKeyFields(file2);
    const comparisonRule =
        keyFields1.length > 0 && keyFields2.length > 0
            ? `Compare records using these key fields. File 1 keys: ${keyFields1.join(', ')}. File 2 keys: ${keyFields2.join(', ')}. A record is "missing in file 2" when a file 1 key tuple does not exist in file 2.`
            : 'No explicit key field was found. Do not guess exact missing records unless the prompt context clearly states them.';

    const prompt = [
        'You are analyzing two loaded data files.',
        'Answer the user question using only the provided prompt data.',
        'Be concise and practical.',
        'Do not assume Elasticsearch or any wrapper format unless the raw payload explicitly shows it.',
        'If both payloads are CSV, reason from CSV rows and headers only.',
        comparisonRule,
        'Definition: "missing" means a file 1 record whose comparison key does not appear in file 2.',
        'Definition: "pattern" means the majority/common trend across the compared rows, such as common missing exchanges, common added fields, common status mapping, or common date lag.',
        'If you mention a pattern, describe it as a repeated/common trend, not a one-off example.',
        'Output format:',
        '1. Missing records summary',
        '2. Majority pattern summary',
        '3. Structural differences',
        '4. Short conclusion',
        '',
        `Question: ${question}`,
        '',
        buildFilePromptContext(file1, 5, 'Raw Request Payload'),
        '',
        buildFilePromptContext(file2, 5, 'Raw Response Payload'),
        '',
        missingRecords
            ? `Records in ${id1} missing from ${id2}: ${JSON.stringify(missingRecords, null, 2)}`
            : `Diff status: ${diffResult}`,
        '',
        'Focus on what is missing in file 1 vs file 2, field-level patterns, structural differences, and likely transformation trends.',
    ].join('\n');

    try {
        const streamedChunks: string[] = [];
        workspace.setAiPromptState?.({
            title: `AI Compare: ${id1} vs ${id2}`,
            system: [
                'You are analyzing two loaded data files.',
                'Answer the user question using only the provided prompt data.',
                'Be concise and practical.',
                'Do not assume Elasticsearch or any wrapper format unless the raw payload explicitly shows it.',
                'If both payloads are CSV, reason from CSV rows and headers only.',
                comparisonRule,
                'Definition: "missing" means a file 1 record whose comparison key does not appear in file 2.',
                'Definition: "pattern" means the majority/common trend across the compared rows.',
                'Focus on what is missing in file 1 vs file 2, field-level patterns, structural differences, and likely transformation trends.',
            ].join('\n'),
            user: question,
            context: [
                buildFilePromptContext(file1, 3, 'Raw Request Payload'),
                '',
                buildFilePromptContext(file2, 3, 'Raw Response Payload'),
                '',
                missingRecords
                    ? `Records in ${id1} missing from ${id2}: ${JSON.stringify(missingRecords, null, 2)}`
                    : `Diff status: ${diffResult}`,
            ].join('\n'),
            fullPrompt: prompt,
        });
        workspace.setCommandOutput?.([
            `Running /ai compare for '${id1}' vs '${id2}' with ${model}...`,
            'Streaming model analysis...',
        ]);

        const analysis = await runPromptStreaming(model, prompt, (chunk) => {
            streamedChunks.push(...chunk.split('\n'));
            const visible = streamedChunks
                .join('\n')
                .split('\n')
                .map((line) => line.trimEnd())
                .filter((line) => line.length > 0)
                .slice(-8);

            workspace.setCommandOutput?.([
                `Running /ai compare for '${id1}' vs '${id2}' with ${model}...`,
                ...visible,
            ]);
        });

        return {
            output: `AI compare analysis for '${id1}' vs '${id2}':\n${analysis}`,
        };
    } catch (error) {
        return { output: `AI Error: ${error instanceof Error ? error.message : String(error)}` };
    }
}

async function executeAiSchema(
    args: string,
    workspace: Workspace,
): Promise<CommandResult> {
    const rawParts = args.split(/\s+/).filter(Boolean);
    const selectedOnly = rawParts.includes('--selected-only');
    const cleanedParts = rawParts.filter((part) => part !== '--selected-only');
    const fileId = cleanedParts[0] || '';
    const extraInstructions = cleanedParts.slice(1).join(' ').trim();

    if (!fileId) return { output: 'Usage: /ai <id>' };

    const loadedFiles = workspace.getLoadedFiles();
    const file = loadedFiles.get(fileId);

    if (!file) {
        const model = workspace.getModel().trim();
        if (!model || model === 'default') {
            return { output: 'Set a model first with /model <name>' };
        }

        const userPrompt = args;
        const previousTurns = workspace.getAiChatHistory?.() || [];
        const loadedFileSummary = Array.from(loadedFiles.values()).map((loadedFile) => ({
            id: loadedFile.id,
            name: loadedFile.name,
            format: loadedFile.fileFormat || 'generated',
            keyField: loadedFile.keyField || null,
            count: loadedFile.data.hits.total.value,
            schema: !!loadedFile.schema,
        }));

        const prompt = [
            'You are the AI assistant inside a terminal data workspace.',
            'Continue the conversation naturally and use prior turns for context.',
            'Be concise and practical.',
            'Use the raw payload/context shown below as ground truth.',
            'Do not invent Elasticsearch wrapper structure unless it is explicitly present in the payload.',
            '',
            `Loaded files summary: ${JSON.stringify(loadedFileSummary, null, 2)}`,
            '',
            previousTurns.length > 0 ? `Conversation so far:\n${previousTurns.join('\n')}` : 'Conversation so far:\n(none)',
            '',
            `User: ${userPrompt}`,
            'Assistant:',
        ].join('\n');

        try {
            const streamedChunks: string[] = [];
            workspace.setAiPromptState?.({
                title: 'AI Chat',
                system: [
                    'You are the AI assistant inside a terminal data workspace.',
                    'Continue the conversation naturally and use prior turns for context.',
                    'Be concise and practical.',
                    'Use the raw payload/context shown below as ground truth.',
                    'Do not invent Elasticsearch wrapper structure unless it is explicitly present in the payload.',
                ].join('\n'),
                user: userPrompt,
                context: [
                    `Loaded files summary: ${JSON.stringify(loadedFileSummary, null, 2)}`,
                    '',
                    ...Array.from(loadedFiles.values()).slice(0, 2).map((loadedFile, index) =>
                        buildFilePromptContext(loadedFile, 2, `Raw Workspace Payload ${index + 1}`),
                    ),
                    '',
                    previousTurns.length > 0 ? `Conversation so far:\n${previousTurns.join('\n')}` : 'Conversation so far:\n(none)',
                ].join('\n'),
                fullPrompt: prompt,
            });
            workspace.setCommandOutput?.([
                `Running /ai chat with ${model}...`,
                'Streaming assistant response...',
            ]);

            const reply = await runPromptStreaming(model, prompt, (chunk) => {
                streamedChunks.push(...chunk.split('\n'));
                const visible = streamedChunks
                    .join('\n')
                    .split('\n')
                    .map((line) => line.trimEnd())
                    .filter((line) => line.length > 0)
                    .slice(-8);

                workspace.setCommandOutput?.([
                    `Running /ai chat with ${model}...`,
                    ...visible,
                ]);
            });

            const nextHistory = [
                ...previousTurns,
                `User: ${userPrompt}`,
                `Assistant: ${reply}`,
            ].slice(-12);
            workspace.setAiChatHistory?.(nextHistory);

            return {
                output: `AI chat:\n${reply}`,
            };
        } catch (error) {
            return { output: `AI Error: ${error instanceof Error ? error.message : String(error)}` };
        }
    }

    try {
        if (selectedOnly) {
            if (!file.schema || !file.subSchemas) {
                return { output: `File '${fileId}' has no schema. Run /ai ${fileId} first` };
            }

            const selectedSchema = buildSelectedSchema(
                file.schema,
                file.subSchemas,
                file.selectedSubSchemaPaths || [],
            );
            const selectedSchemaFileId = `${fileId}-schema-selected`;
            const selectedSchemaData: ElasticSearchResult = {
                took: 0,
                timed_out: false,
                hits: {
                    total: { value: 1, relation: 'eq' },
                    max_score: 1.0,
                    hits: [{
                        _index: 'ai-schema-selected',
                        _id: '1',
                        _score: 1.0,
                        _source: selectedSchema,
                    }],
                },
            };

            workspace.addFile(selectedSchemaFileId, `${file.name} selected schema`, selectedSchemaData);
            workspace.setViewMode('preview', { selectedFileId: selectedSchemaFileId });
            return {
                output: `Selected schema for '${fileId}' saved as '${selectedSchemaFileId}':\n${JSON.stringify(selectedSchema, null, 2)}`,
                action: 'VIEW_CHANGE',
            };
        }

        const model = workspace.getModel().trim();
        if (!model || model === 'default') {
            return { output: 'Set a model first with /model <name>' };
        }

        const streamedChunks: string[] = [];
        const schemaSystemLines = [
            file.fileFormat === 'csv'
                ? `You are given sample rows from loaded file '${fileId}' (format: csv).`
                : file.fileFormat === 'json'
                    ? `You are given sample records from loaded file '${fileId}' (format: json).`
                    : `You are given sample records from loaded file '${fileId}' (format: ${file.fileFormat || 'generated'}).`,
            file.fileFormat === 'csv'
                ? `Infer the row schema for file '${fileId}'. Return it as a JSON Schema object.`
                : `Infer the record schema for file '${fileId}'. Return it as a JSON Schema object.`,
            'Return only valid JSON.',
            'Do not wrap the response in markdown fences.',
            'Use JSON Schema draft-07 style keys when appropriate.',
            'Use the raw file payload below as ground truth.',
            'Do not describe Elasticsearch wrapper structure unless it is explicitly present in the payload.',
        ];
        workspace.setAiPromptState?.({
            title: `AI Schema: ${fileId}`,
            system: schemaSystemLines.join('\n'),
            user: extraInstructions || `Infer schema for loaded file '${fileId}'`,
            context: [
                buildFilePromptContext(file, 5, 'Raw File Payload'),
                '',
                file.subSchemas
                    ? `Schema View:\n${JSON.stringify({
                        selectedSubSchemas: file.selectedSubSchemaPaths || [],
                        availableSubSchemas: Object.keys(file.subSchemas),
                    }, null, 2)}`
                    : 'Schema View:\n(none)',
            ].join('\n'),
            fullPrompt: [
                ...schemaSystemLines,
                extraInstructions?.trim() ? `Additional instructions: ${extraInstructions.trim()}` : '',
                '',
                buildFilePromptContext(file, 5, 'Raw File Payload'),
                '',
                file.subSchemas
                    ? `Schema View:\n${JSON.stringify({
                        selectedSubSchemas: file.selectedSubSchemaPaths || [],
                        availableSubSchemas: Object.keys(file.subSchemas),
                    }, null, 2)}`
                    : 'Schema View:\n(none)',
            ].filter((line) => line.length > 0).join('\n'),
        });
        workspace.setCommandOutput?.([
            `Running /ai for '${fileId}' with ${model}...`,
            'Streaming model output...',
        ]);

        const schema = await inferSchemaForDataStreaming(
            model,
            fileId,
            file.data,
            file.fileFormat,
            extraInstructions,
            (chunk) => {
                streamedChunks.push(...chunk.split('\n'));
                const visible = streamedChunks
                    .join('\n')
                    .split('\n')
                    .map((line) => line.trimEnd())
                    .filter((line) => line.length > 0)
                    .slice(-6);

                workspace.setCommandOutput?.([
                    `Running /ai for '${fileId}' with ${model}...`,
                    ...visible,
                ]);
            },
        );
        const typedSchema = schema as JsonSchema;
        const subSchemas = extractSubSchemas(typedSchema);
        const schemaFileId = `${fileId}-schema`;
        const schemaData: ElasticSearchResult = {
            took: 0,
            timed_out: false,
            hits: {
                total: { value: 1, relation: 'eq' },
                max_score: 1.0,
                hits: [{
                    _index: 'ai-schema',
                    _id: '1',
                    _score: 1.0,
                    _source: schema,
                }],
            },
        };

        workspace.updateFile(fileId, {
            schema: typedSchema,
            subSchemas,
            selectedSubSchemaPaths: Object.keys(subSchemas),
        });
        workspace.addFile(schemaFileId, `${file.name} schema`, schemaData);
        workspace.setViewMode('preview', { selectedFileId: schemaFileId });
        return {
            output: `Schema for '${fileId}' via ${model} saved as '${schemaFileId}':\n${JSON.stringify(schema, null, 2)}`,
            action: 'VIEW_CHANGE',
        };
    } catch (error) {
        return { output: `AI Error: ${error instanceof Error ? error.message : String(error)}` };
    }
}

// Register all commands
export function registerAllCommands(registry: CommandRegistry = globalRegistry) {
    registry.clear();
    registry.register(HelpCommand);
    registry.register(ClearCommand);
    registry.register(RefreshCommand);
    registry.register(ModelCommand);
    registry.register(LoadCommand);
    registry.register(LoadJsonCommand);
    registry.register(LoadCsvCommand);
    registry.register(AnalyzeCommand);
    registry.register(PreviewCommand);
    registry.register(CatCommand);
    registry.register(FilesCommand);
    registry.register(CompareCommand);
    registry.register(MatchCommand);
    registry.register(KeyFieldCommand);
    registry.register(SortCommand);
    registry.register(MinusCommand);
    registry.register(DiffCommand);
    registry.register(SqlCommand);
    registry.register(AiCommand);
    registry.register(AiProviderCommand);
    registry.register(ShowPromptCommand);
    registry.register(SchemaCommand);
    registry.register(SaveSchemaCommand);
    registry.register(SaveCommand);
    registry.register(ClusterCommand);
    registry.register(ExitCommand);
}

export function createRegisteredCommandRegistry(): CommandRegistry {
    const registry = new CommandRegistry();
    registerAllCommands(registry);
    return registry;
}
