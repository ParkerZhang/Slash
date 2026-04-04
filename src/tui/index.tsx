import React, { useState, useEffect, useMemo, useRef } from 'react';
import { render, Text, Box, useInput } from 'ink';
import TextInput from 'ink-text-input';
import { addCommand, getHistory, getLoadedFiles, getSelectedModel, setSelectedModel, updateKeyField } from '../core/historyManager.js';
import * as fs from 'fs';
import * as path from 'path';
import { AiPromptState, ElasticSearchHit, ElasticSearchResult, LoadedFile, ViewMode, Workspace } from '../core/types.js';
import { CommandRegistry } from '../commands/commandRegistry.js';
import { createRegisteredCommandRegistry } from '../commands/index.js';
import { JsonSchema } from '../core/schema.js';
import { BANNERS } from './banners.js';
import { BannerComponent } from './Banner.js';

interface CompareData {
    fileA: LoadedFile;
    fileB: LoadedFile;
    selectedIndex: number;
    isMatchResult?: boolean;
    matchData?: Array<{
        request: Record<string, unknown>;
        response: Record<string, unknown> | null;
        key: string;
        matched: boolean;
    }>;
}

interface NestedPath {
    rowIndex: number;
    fieldName: string;
    data: Record<string, unknown>;
}

let globalUnmount: (() => void) | null = null;

const DATA_DIR = path.join(process.cwd(), 'data');

async function loadFreshRegistry(): Promise<CommandRegistry> {
    const mod = await import(`../commands/index.js?refresh=${Date.now()}`);
    if (typeof mod.createRegisteredCommandRegistry === 'function') {
        return mod.createRegisteredCommandRegistry();
    }
    return createRegisteredCommandRegistry();
}

const resolveDataPath = (fileName: string): string => {
    if (path.isAbsolute(fileName)) return fileName;
    if (fileName.startsWith('data/') || fileName.startsWith('./data/')) return path.resolve(fileName);
    return path.join(DATA_DIR, fileName);
};

const getDataFiles = (): string[] => {
    try {
        return fs.readdirSync(DATA_DIR).filter(f => /\.(csv|json)$/i.test(f));
    } catch {
        return [];
    }
};

const getCsvHeaders = (fileName: string): string[] => {
    try {
        const content = fs.readFileSync(resolveDataPath(fileName), 'utf-8');
        const firstLine = content.split('\n')[0]?.trim();
        if (!firstLine) {
            return [];
        }
        return firstLine.split('|').map((header) => header.trim()).filter(Boolean);
    } catch {
        return [];
    }
};

type SuggestionKind = 'command' | 'model' | 'data-file' | 'loaded-file' | 'option';

interface SuggestionState {
    items: string[];
    kind: SuggestionKind;
    appendAsNewCommand?: boolean;
    parameterMode?: boolean;
}

const FIXED_ARG_COUNTS: Record<string, number> = {
    '/refresh': 0,
    '/help': 0,
    '/files': 0,
    '/clear': 0,
    '/exit': 0,
    '/model': 1,
    '/preview': 1,
    '/schema': 1,
    '/keyfield': 1,
    '/compare': 2,
    '/match': 2,
    '/minus': 2,
    '/diff': 2,
};

const getCurrentCommandSegment = (input: string): { prefix: string; segment: string } => {
    const match = input.match(/^(.*?)([^&]*)$/s);
    if (!match) {
        return { prefix: '', segment: input };
    }

    const rawPrefix = match[1] || '';
    const normalizedPrefix = rawPrefix.includes('&&')
        ? rawPrefix.slice(0, rawPrefix.lastIndexOf('&&') + 2)
        : '';

    return {
        prefix: normalizedPrefix,
        segment: normalizedPrefix ? input.slice(normalizedPrefix.length) : input,
    };
};

const getSuggestionStateForInput = (
    input: string,
    registry: CommandRegistry,
    workspace: Workspace,
    loadedFiles: Map<string, LoadedFile>,
    fileCommandNames: Set<string>,
): SuggestionState => {
    const { segment } = getCurrentCommandSegment(input);
    const activeInput = segment.trimStart();

    if (!activeInput.startsWith('/')) {
        return { items: [], kind: 'command' };
    }

    const parts = activeInput.trim().split(/\s+/);
    const cmd = parts[0].toLowerCase();
    const hasTrailingSpace = activeInput.endsWith(' ');
    const argIndex = Math.max(0, parts.length - 1 + (hasTrailingSpace ? 1 : 0));
    const currentToken = hasTrailingSpace ? '' : (parts[parts.length - 1] || '');
    const currentTokenLower = currentToken.toLowerCase();
    const loadedIds = Array.from(loadedFiles.keys());
    const fixedArgCount = FIXED_ARG_COUNTS[cmd];
    const currentArgCount = parts.length - 1;

    if (fixedArgCount !== undefined && currentArgCount >= fixedArgCount && hasTrailingSpace) {
        return {
            items: registry.getAllCommands().map((c) => c.name),
            kind: 'command',
            appendAsNewCommand: true,
            parameterMode: false,
        };
    }

    if ((cmd === '/load' || cmd === '/loadcsv') && (parts.length >= 2 || hasTrailingSpace)) {
        const dataFiles = getDataFiles();
        return {
            items: dataFiles.filter((f) => f.toLowerCase().startsWith(currentTokenLower)),
            kind: 'data-file',
            parameterMode: true,
        };
    }

    if (cmd === '/analyze') {
        if (argIndex === 1 || argIndex === 2) {
            const dataFiles = getDataFiles();
            return {
                items: dataFiles.filter((f) => f.toLowerCase().startsWith(currentTokenLower)),
                kind: 'data-file',
                parameterMode: true,
            };
        }

        if (argIndex >= 3) {
            const requestFile = parts[1];
            const responseFile = parts[2];
            const requestHeaders = getCsvHeaders(requestFile);
            const responseHeaders = getCsvHeaders(responseFile);
            const commonHeaders = requestHeaders.filter((header) => responseHeaders.includes(header));

            return {
                items: commonHeaders.filter((header) => header.toLowerCase().startsWith(currentTokenLower)),
                kind: 'option',
                parameterMode: true,
            };
        }
    }

    if (cmd === '/model' && (parts.length >= 2 || hasTrailingSpace)) {
        const modelCommand = registry.getCommand('/model');
        return {
            items: modelCommand?.suggestArgs?.(currentToken, workspace) || [],
            kind: 'model',
            parameterMode: true,
        };
    }

    if (cmd === '/ai' && argIndex === 1) {
        const baseOptions = ['compare', '--selected-only'];
        const fileMatches = loadedIds.filter((id) => id.toLowerCase().startsWith(currentTokenLower));
        const optionMatches = baseOptions.filter((option) => option.startsWith(currentTokenLower));
        return {
            items: [...optionMatches, ...fileMatches],
            kind: optionMatches.length > 0 && fileMatches.length === 0 ? 'option' : 'loaded-file',
            parameterMode: true,
        };
    }

    if (cmd === '/ai' && parts[1]?.toLowerCase() === 'compare') {
        if (argIndex === 2 || argIndex === 3) {
            return {
                items: loadedIds.filter((id) => id.toLowerCase().startsWith(currentTokenLower)),
                kind: 'loaded-file',
                parameterMode: true,
            };
        }
        return { items: [], kind: 'option', parameterMode: true };
    }

    if (cmd === '/saveschema' && argIndex === 1) {
        return {
            items: loadedIds.filter((id) => id.toLowerCase().startsWith(currentTokenLower)),
            kind: 'loaded-file',
            parameterMode: true,
        };
    }

    if (cmd === '/saveschema' && currentToken.startsWith('--')) {
        return {
            items: ['--selected-only'].filter((option) => option.startsWith(currentToken)),
            kind: 'option',
            parameterMode: true,
        };
    }

    if (cmd === '/sort') {
        if (argIndex === 1 || argIndex === 2) {
            return {
                items: loadedIds.filter((id) => id.toLowerCase().startsWith(currentTokenLower)),
                kind: 'loaded-file',
                parameterMode: true,
            };
        }
        if (argIndex >= 3) {
            return {
                items: ['asc', 'desc'].filter((option) => option.startsWith(currentTokenLower)),
                kind: 'option',
                parameterMode: true,
            };
        }
    }

    if (cmd === '/compare' || cmd === '/match' || cmd === '/minus' || cmd === '/diff') {
        if (argIndex === 1 || argIndex === 2) {
            const usedIds = new Set(parts.slice(1, argIndex).map((part) => part.toLowerCase()));
            return {
                items: loadedIds
                    .filter((id) => !usedIds.has(id.toLowerCase()))
                    .filter((id) => id.toLowerCase().startsWith(currentTokenLower)),
                kind: 'loaded-file',
                parameterMode: true,
            };
        }
    }

    if (parts.length >= 2 && fileCommandNames.has(cmd)) {
        return {
            items: loadedIds.filter((id) => id.toLowerCase().startsWith(currentTokenLower)),
            kind: 'loaded-file',
            parameterMode: true,
        };
    }

    return {
        items: registry.getAllCommands()
            .filter((c) => c.name.startsWith(activeInput.toLowerCase()))
            .map((c) => c.name),
        kind: 'command',
        parameterMode: false,
    };
};

const Banner: React.FC = () => {
    const bannerPoem = [
        '皇帝立国，维初在昔，嗣世称王',
        '讨伐乱逆，威动四极，武义直方',
        '戎臣奉诏，经时不久，灭六暴强',
        '廿有六年，上荐高号，孝道显明',
    ];
    const bannerSuffix = [
        '1. 山岳名录 - 搜罗天下名山，编纂《山岳志》',
        '2. 篆文编纂 - 以李斯笔法，为每山撰写铭文',
        '3. 碑文镌刻 - 程序生成篆文碑文，支持 Unicode 小篆',
    ];
    const churchillLines = [
        'We shall fight on the beaches,',
        'We shall fight on the landing grounds,',
        'We shall fight in the fields and in the streets,',
        'We shall fight in the hills;',
        'We shall never surrender.',
    ];

    return (
        <Box flexDirection="row" borderStyle="round" borderDimColor paddingX={1} paddingY={0}>
            <Box flexDirection="column">
                <Text bold color="#00FF00">/slash 峄石铭 · 秦篆</Text>
                {bannerPoem.map((line: string, i: number) => (
                    <Text key={'poem-' + i} bold color="#00FF00">
                        {'  '}{line}
                    </Text>
                ))}
                <Text color="#00FF00">{'  ' + '─'.repeat(30)}</Text>
                <Text bold color="#00FF00">{'  '} 一、山岳刻石工程</Text>
                {bannerSuffix.map((line: string, i: number) => (
                    <Text key={'suffix-' + i} color="#00FF00">
                        {'  '}{line}
                    </Text>
                ))}
            </Box>
            <Box flexDirection="column">
                <Text color="#00FF00">{'  ' + '─'.repeat(30)}</Text>
                <Text bold color="#00FF00">{'  '}We Shall Fight</Text>
                {churchillLines.map((line: string, i: number) => (
                    <Text key={'eng-' + i} bold color="#00FF00">
                        {'  '}{line}
                    </Text>
                ))}
                <Text color="#00FF00">{'  '}— Winston Churchill, 1940</Text>
                <Text color="#00FF00">{'  ' + '─'.repeat(30)}</Text>
            </Box>
        </Box>
    );
};

const SUGGESTION_PANEL_HEIGHT = 6;

const StatusLine: React.FC<{ model: string; historyCount: number; fileCount: number }> = ({
    model,
    historyCount,
    fileCount,
}) => {
    return (
        <Box marginTop={1} marginBottom={1} borderStyle="round" borderColor="blue" paddingX={1}>
            <Text>
                <Text bold color="blue">Model:</Text> <Text color="cyan">{model}</Text>
                <Text dimColor> | History: {historyCount} | Files: {fileCount}</Text>
            </Text>
        </Box>
    );
};

const App = () => {
    const [history, setHistory] = useState<string[]>([]);
    const [input, setInput] = useState('');
    const [output, setOutput] = useState<string[]>([]);
    const [outputScroll, setOutputScroll] = useState<number>(0);
    const maxVisibleLines = 16;
    const [aiChatHistory, setAiChatHistory] = useState<string[]>([]);
    const [aiPromptState, setAiPromptState] = useState<AiPromptState | null>(null);
    const [spinnerFrame, setSpinnerFrame] = useState(0);
    const [historyIndex, setHistoryIndex] = useState(-1);
    const [loadedFiles, setLoadedFiles] = useState<Map<string, LoadedFile>>(new Map());
    const [model, setModel] = useState('default');
    const [registry, setRegistry] = useState<CommandRegistry>(() => createRegisteredCommandRegistry());
    const [bannerIndex, setBannerIndex] = useState<number>(() => {
        const saved = process.env.TUI_BANNER;
        const idx = BANNERS.findIndex(b => b.name === saved);
        return idx >= 0 ? idx : 0; // haiku-1 is first
    });
    const bannerStyle = useMemo(() => BANNERS[bannerIndex], [bannerIndex]);
    const ignoreNextInput = useRef(false);
    const cycleBanner = () => {
        ignoreNextInput.current = true;
        setBannerIndex(prev => (prev + 1) % BANNERS.length);
    };
    const [viewMode, setViewMode] = useState<ViewMode>('main');
    const [selectedFileId, setSelectedFileId] = useState<string>('');
    const [selectedRow, setSelectedRow] = useState(0);
    const [detailFieldIndex, setDetailFieldIndex] = useState(0);
    const [expandedFields, setExpandedFields] = useState<Record<number, Set<string>>>({});
    const [nestedPath, setNestedPath] = useState<NestedPath | null>(null);
    const [compareData, setCompareData] = useState<CompareData | null>(null);
    const [keyFieldFileId, setKeyFieldFileId] = useState<string>('');
    const [schemaSelectionIndex, setSchemaSelectionIndex] = useState(0);
    const [suggestionIndex, setSuggestionIndex] = useState(-1);
    const [historySearchPrefix, setHistorySearchPrefix] = useState<string | null>(null);
    const [suggestionInputStack, setSuggestionInputStack] = useState<string[]>([]);
    const [inputKey, setInputKey] = useState(0);
    const isAiRunning = output[0]?.startsWith('Running /ai for ') ?? false;

    const handleInputChange = (value: string) => {
        if (ignoreNextInput.current) {
            ignoreNextInput.current = false;
            return;
        }
        setInput(value);
        setHistoryIndex(-1);
        setHistorySearchPrefix(null);
        setSuggestionIndex(-1);
    };

    const refreshRegistry = async () => {
        const nextRegistry = await loadFreshRegistry();
        setRegistry(nextRegistry);
        setOutput(['Command registry refreshed']);
    };

    // Implementation of the Workspace interface for TUI
    const workspace = useMemo<Workspace>(() => ({
        getLoadedFiles: () => loadedFiles,
        addFile: (id, name, data) => {
            setLoadedFiles(prev => {
                const newMap = new Map(prev);
                newMap.set(id, { id, name, data, fileFormat: 'generated' });
                return newMap;
            });
        },
        updateFile: (id, updates) => {
            setLoadedFiles(prev => {
                const newMap = new Map(prev);
                const file = newMap.get(id);
                if (file) {
                    newMap.set(id, { ...file, ...updates });
                }
                return newMap;
            });
            if (updates.keyField) {
                updateKeyField(id, updates.keyField);
            }
        },
        removeFile: (id) => {
            setLoadedFiles(prev => {
                const newMap = new Map(prev);
                newMap.delete(id);
                return newMap;
            });
        },
        getModel: () => model,
        setModel: (nextModel) => {
            setModel(nextModel);
            setSelectedModel(nextModel);
        },
        setViewMode: (mode, data) => {
            setViewMode(mode);
            if (mode === 'preview' || mode === 'keyfield' || mode === 'schema') {
                setSelectedFileId(data.selectedFileId);
                setSelectedRow(0);
                setDetailFieldIndex(0);
                setSchemaSelectionIndex(0);
                if (mode === 'keyfield') {
                    setKeyFieldFileId(data.selectedFileId);
                }
            } else if (mode === 'compare' || mode === 'match') {
                setCompareData(data);
            }
        },
        getHistory: () => history,
        addHistory: (cmd) => {
            setHistory(prev => [...prev, cmd]);
            addCommand(cmd);
        },
        clearHistory: () => {
            setHistory([]);
        },
        getAiChatHistory: () => aiChatHistory,
        setAiChatHistory: (nextHistory) => {
            setAiChatHistory(nextHistory);
        },
        getAiPromptState: () => aiPromptState,
        setAiPromptState: (state) => {
            setAiPromptState(state);
        },
        setCommandOutput: (lines) => {
            setOutput(lines);
        },
        exit: () => {
            if (globalUnmount) globalUnmount();
            process.exit(0);
        }
    }), [loadedFiles, history, model, aiChatHistory, aiPromptState]);

    useEffect(() => {
        const loadedHistory = getHistory();
        setHistory(loadedHistory);
        setModel(getSelectedModel() || 'default');

        const savedFiles = getLoadedFiles() || [];
        const newLoadedFiles = new Map<string, LoadedFile>();
        
        for (const fileInfo of savedFiles) {
            if (!fileInfo || !fileInfo.name) continue;
            try {
                const content = fs.readFileSync(resolveDataPath(fileInfo.name), 'utf-8');
                let data: ElasticSearchResult;
                const ext = fileInfo.name.toLowerCase().split('.').pop();
                
                if (ext === 'csv') {
                    const lines = content.trim().split('\n');
                    if (lines.length < 2) continue;
                    const separator = '|';
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
                    data = {
                        took: 0,
                        timed_out: false,
                        hits: {
                            total: { value: hits.length, relation: 'eq' },
                            max_score: 1.0,
                            hits: hits
                        }
                    };
                } else {
                    try {
                        data = JSON.parse(content) as ElasticSearchResult;
                    } catch {
                        continue;
                    }
                }
                
                newLoadedFiles.set(fileInfo.id, { 
                    id: fileInfo.id, 
                    name: fileInfo.name, 
                    data, 
                    fileFormat: fileInfo.fileFormat || (ext === 'csv' ? 'csv' : 'json'),
                    keyField: fileInfo.keyField 
                });
            } catch {
                // skip files that can't be loaded
            }
        }
        
        if (newLoadedFiles.size > 0) {
            setLoadedFiles(newLoadedFiles);
            setOutput(['Loaded ' + newLoadedFiles.size + ' saved file(s) from history']);
        }
    }, []);

    useEffect(() => {
        refreshRegistry().catch((error) => {
            setOutput([`Refresh failed: ${error instanceof Error ? error.message : String(error)}`]);
        });
    }, []);

    useEffect(() => {
        if (!isAiRunning) {
            setSpinnerFrame(0);
            return;
        }

        const interval = setInterval(() => {
            setSpinnerFrame((frame) => (frame + 1) % 4);
        }, 120);

        return () => clearInterval(interval);
    }, [isAiRunning]);

    const toggleField = (rowIndex: number, key: string) => {
        setExpandedFields(prev => {
            const current = prev[rowIndex] || new Set();
            const newSet = new Set(current);
            if (newSet.has(key)) {
                newSet.delete(key);
            } else {
                newSet.add(key);
            }
            return { ...prev, [rowIndex]: newSet };
        });
    };

    // Commands that expect file IDs as arguments
    const fileCommandNames = new Set(['/preview', '/compare', '/match', '/keyfield', '/sort', '/minus', '/diff', '/save', '/ai', '/schema', '/saveschema']);

    const applySuggestion = (
        nextIndex: number,
        currentSuggestions: string[],
        options?: { advanceToNextParam?: boolean },
    ) => {
        const { prefix, segment } = getCurrentCommandSegment(input);
        const activeInput = segment.trimStart();
        const trimmedActiveInput = activeInput.trim();
        const parts = trimmedActiveInput.split(/\s+/).filter(Boolean);
        const hasTrailingSpace = activeInput.endsWith(' ');
        const activeParts = activeInput.startsWith('/') ? [...parts] : [];
        const replacement = currentSuggestions[nextIndex];
        const suggestionState = getSuggestionStateForInput(input, registry, workspace, loadedFiles, fileCommandNames);
        const advanceToNextParam = options?.advanceToNextParam ?? false;

        setSuggestionIndex(nextIndex);

        if (advanceToNextParam) {
            setSuggestionInputStack((prev) => [...prev, input]);
        }

        if (suggestionState.appendAsNewCommand) {
            const base = input.trimEnd();
            setInput(`${base} && ${replacement} `);
            setInputKey(k => k + 1);
            return;
        }

        if (!activeInput.startsWith('/')) {
            setInput(replacement + (advanceToNextParam ? ' ' : ''));
            setInputKey(k => k + 1);
            return;
        }

        if (activeParts.length === 0) {
            setInput(prefix + replacement + (advanceToNextParam ? ' ' : ''));
        } else if (hasTrailingSpace) {
            activeParts.push(replacement);
            setInput(prefix + activeParts.join(' ') + (advanceToNextParam ? ' ' : ''));
        } else {
            activeParts[activeParts.length - 1] = replacement;
            setInput(prefix + activeParts.join(' ') + (advanceToNextParam ? ' ' : ''));
        }
        setInputKey(k => k + 1);
    };

    const exitToCommandHistory = () => {
        const { prefix, segment } = getCurrentCommandSegment(input);
        const activeInput = segment.trimStart();
        if (!activeInput.startsWith('/')) {
            setSuggestionInputStack([]);
            setSuggestionIndex(-1);
            return;
        }

        const parts = activeInput.trim().split(/\s+/).filter(Boolean);
        const commandOnly = parts[0] || '';
        setInput(prefix + commandOnly);
        setSuggestionInputStack([]);
        setSuggestionIndex(-1);
        setInputKey(k => k + 1);
    };

    useInput((inputKey, key) => {
        // Ctrl+B to cycle banners
        if (key.ctrl && inputKey === 'b') {
            cycleBanner();
            return;
        }

        if (viewMode === 'preview' && selectedFileId) {
            const file = loadedFiles.get(selectedFileId);
            const rowCount = file?.data.hits.hits.length || 0;
            
            if (key.upArrow && selectedRow > 0) {
                setSelectedRow(r => r - 1);
            } else if (key.downArrow && selectedRow < rowCount - 1) {
                setSelectedRow(r => r + 1);
            } else if (key.return) {
                setDetailFieldIndex(0);
                setExpandedFields({});
                setViewMode('detail');
            } else if (key.escape) {
                setViewMode('main');
                setSelectedRow(0);
            }
        } else if (viewMode === 'detail') {
            const file = loadedFiles.get(selectedFileId);
            if (!file) return;
            const hit = file.data.hits.hits[selectedRow];
            if (!hit) return;
            const allFields = Object.keys(hit._source).sort();
            const fieldCount = allFields.length;
            
            if (key.upArrow && detailFieldIndex > 0) {
                setDetailFieldIndex(i => i - 1);
            } else if (key.downArrow && detailFieldIndex < fieldCount - 1) {
                setDetailFieldIndex(i => i + 1);
            } else if (key.return) {
                const field = allFields[detailFieldIndex];
                const value = hit._source[field];
                if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
                    setDetailFieldIndex(0);
                    setNestedPath({
                        rowIndex: selectedRow,
                        fieldName: field,
                        data: value as Record<string, unknown>
                    });
                    setViewMode('nested');
                } else {
                    toggleField(selectedRow, field);
                }
            } else if (key.escape) {
                setViewMode('preview');
                setDetailFieldIndex(0);
                setExpandedFields({});
            }
        } else if (viewMode === 'nested' && nestedPath) {
            const nestedKeys = Object.keys(nestedPath.data).sort();
            const keyCount = nestedKeys.length;
            
            if (key.upArrow && detailFieldIndex > 0) {
                setDetailFieldIndex(i => i - 1);
            } else if (key.downArrow && detailFieldIndex < keyCount - 1) {
                setDetailFieldIndex(i => i + 1);
            } else if (key.return) {
                const k = nestedKeys[detailFieldIndex];
                const v = nestedPath.data[k];
                if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
                    setNestedPath({
                        rowIndex: nestedPath.rowIndex,
                        fieldName: nestedPath.fieldName + '.' + k,
                        data: v as Record<string, unknown>
                    });
                } else {
                    toggleField(nestedPath.rowIndex, nestedPath.fieldName + '.' + k);
                }
            } else if (key.escape) {
                setViewMode('detail');
                setNestedPath(null);
            }
        } else if (viewMode === 'compare' && compareData) {
            const maxIndex = Math.max(compareData.fileA.data.hits.hits.length, compareData.fileB.data.hits.hits.length) - 1;
            
            if (key.upArrow && compareData.selectedIndex > 0) {
                setCompareData(d => d ? { ...d, selectedIndex: d.selectedIndex - 1 } : null);
            } else if (key.downArrow && compareData.selectedIndex < maxIndex) {
                setCompareData(d => d ? { ...d, selectedIndex: d.selectedIndex + 1 } : null);
            } else if (key.escape) {
                setViewMode('main');
                setCompareData(null);
            }
        } else if (viewMode === 'match' && compareData?.matchData) {
            const maxIndex = compareData.matchData.length - 1;
            
            if (key.upArrow && compareData.selectedIndex > 0) {
                setCompareData(d => d ? { ...d, selectedIndex: d.selectedIndex - 1 } : null);
            } else if (key.downArrow && compareData.selectedIndex < maxIndex) {
                setCompareData(d => d ? { ...d, selectedIndex: d.selectedIndex + 1 } : null);
            } else if (key.escape) {
                setViewMode('main');
                setCompareData(null);
            }
        } else if (viewMode === 'keyfield' && keyFieldFileId) {
            const file = loadedFiles.get(keyFieldFileId);
            if (!file) return;
            const hit = file.data.hits.hits[0];
            if (!hit) return;
            const fields = Object.keys(hit._source).sort();
            const fieldCount = fields.length;
            
            if (key.upArrow && detailFieldIndex > 0) {
                setDetailFieldIndex(i => i - 1);
            } else if (key.downArrow && detailFieldIndex < fieldCount - 1) {
                setDetailFieldIndex(i => i + 1);
            } else if (key.return) {
                const field = fields[detailFieldIndex];
                workspace.updateFile(keyFieldFileId, { keyField: field });
                setOutput(['Key field set to: ' + field]);
                setViewMode('main');
                setKeyFieldFileId('');
            } else if (key.escape) {
                setViewMode('main');
                setKeyFieldFileId('');
            }
            // Cluster view scrolling handled in main key handler
        } else if (viewMode === 'schema' && selectedFileId) {
            const file = loadedFiles.get(selectedFileId);
            if (!file?.schema) return;
            const paths = Object.keys(file.subSchemas || {}).sort();

            if (key.upArrow && schemaSelectionIndex > 0) {
                setSchemaSelectionIndex(i => i - 1);
            } else if (key.downArrow && schemaSelectionIndex < paths.length - 1) {
                setSchemaSelectionIndex(i => i + 1);
            } else if ((inputKey === ' ' || key.return) && paths.length > 0) {
                const targetPath = paths[schemaSelectionIndex];
                const selected = new Set(file.selectedSubSchemaPaths || []);
                if (selected.has(targetPath)) {
                    selected.delete(targetPath);
                } else {
                    selected.add(targetPath);
                }
                workspace.updateFile(selectedFileId, {
                    selectedSubSchemaPaths: Array.from(selected).sort(),
                });
            } else if (key.escape) {
                setViewMode('main');
                setSchemaSelectionIndex(0);
            }
        } else {
            const currentSuggestionState = getSuggestionStateForInput(input, registry, workspace, loadedFiles, fileCommandNames);
            const currentSuggestions = currentSuggestionState.items;
            const suggestionsActive = input.startsWith('/') && currentSuggestions.length > 0;
            const suggestionIsHighlighted = suggestionsActive && suggestionIndex >= 0;

            if (suggestionIsHighlighted && key.upArrow) {
                const nextIndex = suggestionIndex > 0 ? suggestionIndex - 1 : currentSuggestions.length - 1;
                setSuggestionIndex(nextIndex);
            } else if (suggestionIsHighlighted && key.downArrow) {
                const nextIndex = suggestionIndex < currentSuggestions.length - 1 ? suggestionIndex + 1 : 0;
                setSuggestionIndex(nextIndex);
            } else if (suggestionIsHighlighted && (key.return || key.tab)) {
                applySuggestion(suggestionIndex, currentSuggestions, { advanceToNextParam: true });
                setSuggestionIndex(-1);
            } else if (suggestionsActive && key.escape) {
                setSuggestionIndex(-1);
                setHistoryIndex(-1);
                setHistorySearchPrefix(null);
            } else if (key.upArrow && history.length > 0) {
                const prefix = historySearchPrefix !== null ? historySearchPrefix : input;
                const matches = history.filter(cmd => cmd.startsWith(prefix));
                if (matches.length > 0) {
                    const currentMatchIndexInFiltered = historySearchPrefix === null ? -1 : matches.lastIndexOf(input);
                    const newMatchIndex = currentMatchIndexInFiltered === -1 ? matches.length - 1 : (currentMatchIndexInFiltered > 0 ? currentMatchIndexInFiltered - 1 : 0);
                    
                    if (historySearchPrefix === null) setHistorySearchPrefix(prefix);
                    setHistoryIndex(history.length - 1 - history.lastIndexOf(matches[newMatchIndex]));
                    setInput(matches[newMatchIndex]);
                }
                setSuggestionIndex(-1);
            } else if (key.downArrow && historySearchPrefix !== null) {
                const matches = history.filter(cmd => cmd.startsWith(historySearchPrefix));
                const currentMatchIndexInFiltered = matches.lastIndexOf(input);
                
                if (currentMatchIndexInFiltered !== -1 && currentMatchIndexInFiltered < matches.length - 1) {
                    const newMatchIndex = currentMatchIndexInFiltered + 1;
                    setHistoryIndex(history.length - 1 - history.lastIndexOf(matches[newMatchIndex]));
                    setInput(matches[newMatchIndex]);
                } else {
                    setHistoryIndex(-1);
                    setHistorySearchPrefix(null);
                    setInput(historySearchPrefix);
                }
                setSuggestionIndex(-1);
            } else if (key.tab) {
                if (suggestionsActive) {
                    if (suggestionIndex === -1 && currentSuggestions.length > 0) {
                        setSuggestionIndex(0);
                    } else if (suggestionIndex >= 0) {
                        const nextIndex = suggestionIndex < currentSuggestions.length - 1 ? suggestionIndex + 1 : 0;
                        setSuggestionIndex(nextIndex);
                    }
                }
            } else if (key.escape) {
                setHistoryIndex(-1);
                setHistorySearchPrefix(null);
                setSuggestionIndex(-1);
                setOutputScroll(0);
                setClusterScroll(0);
                if (key.pageUp) {
                    setClusterScroll(prev => Math.max(0, prev - Math.floor(maxVisibleLines / 2)));
                } else if (key.pageDown) {
                    setClusterScroll(prev => prev + Math.floor(maxVisibleLines / 2));
                } else if (key.upArrow) {
                    setClusterScroll(prev => Math.max(0, prev - 1));
                } else if (key.downArrow) {
                    setClusterScroll(prev => prev + 1);
                } else if (key.escape) {
                    setViewMode('main');
                    setClusterScroll(0);
                }
            } else if (key.pageUp) {
                const maxScroll = Math.max(0, output.length - maxVisibleLines);
                setOutputScroll(prev => Math.max(0, prev - Math.floor(maxVisibleLines / 2)));
            } else if (key.pageDown) {
                const maxScroll = Math.max(0, output.length - maxVisibleLines);
                setOutputScroll(prev => Math.min(maxScroll, prev + Math.floor(maxVisibleLines / 2)));
            } else if (key.upArrow && output.length > maxVisibleLines && !suggestionsActive) {
                setOutputScroll(prev => Math.max(0, prev - 1));
            } else if (key.downArrow && output.length > maxVisibleLines && !suggestionsActive) {
                const maxScroll = Math.max(0, output.length - maxVisibleLines);
                setOutputScroll(prev => Math.min(maxScroll, prev + 1));
            }
        }
    });

    const suggestionState = getSuggestionStateForInput(input, registry, workspace, loadedFiles, fileCommandNames);
    const suggestions = suggestionState.items;

    // Reset suggestion index when suggestions change
    useEffect(() => {
        if (suggestions.length === 0) {
            setSuggestionIndex(-1);
        } else if (suggestionIndex >= suggestions.length) {
            setSuggestionIndex(0);
        }
    }, [input, suggestions.length]);

    const handleSubmit = async (value: string) => {
        if (!value.trim()) return;

        setAiPromptState(null);
        const commandsList = value.split(/\s*&&\s*/).map(s => s.trim()).filter(s => s);
        
        for (const cmdValue of commandsList) {
            workspace.addHistory(cmdValue);
            setHistoryIndex(-1);
            setSuggestionInputStack([]);

            if (cmdValue.startsWith('/')) {
                const result = await registry.execute(cmdValue, workspace);
                if (result.action === 'CLEAR') {
                    setHistory([]);
                    setOutputScroll(0);
                    setOutput(['History cleared']);
                } else if (result.action === 'REFRESH') {
                    await refreshRegistry();
                } else if (result.output) {
                    setOutputScroll(0);
                    setOutput(result.output.split('\n'));
                }
            } else {
                setOutputScroll(0);
                setOutput(["Echo: " + cmdValue]);
            }
        }
        setInput('');
        setSuggestionInputStack([]);
    };

    const renderDetailPanel = () => {
        const file = loadedFiles.get(selectedFileId);
        if (!file) return null;

        const hit = file.data.hits.hits[selectedRow];
        if (!hit) return null;

        const allFields = Object.keys(hit._source).sort();
        const expanded = expandedFields[selectedRow] || new Set();

        const renderValue = (value: unknown, indent: number = 2): React.ReactNode => {
            if (Array.isArray(value)) {
                return (
                    <Box flexDirection="column">
                        {value.map((item: unknown, i: number) => (
                            <Text key={i}>{" ".repeat(indent)}[{i}] {JSON.stringify(item)}</Text>
                        ))}
                    </Box>
                );
            } else if (typeof value === 'object' && value !== null) {
                return (
                    <Box flexDirection="column">
                        {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
                            <Text key={k}>{" ".repeat(indent)}{k}: {JSON.stringify(v)}</Text>
                        ))}
                    </Box>
                );
            }
            return <Text>{JSON.stringify(value)}</Text>;
        };

        return (
            <Box flexDirection="column" borderStyle="round" borderColor="yellow" padding={1}>
                <Text bold color="yellow">Detail View - Record {selectedRow + 1}</Text>
                <Text dimColor>↑↓ navigate fields, Enter expand/collapse, Esc back</Text>
                <Text>──────────────────────────────────────</Text>
                <Text bold>ID: {hit._id}</Text>
                <Text dimColor>Index: {hit._index}</Text>
                <Text dimColor>Score: {hit._score}</Text>
                <Text>──────────────────────────────────────</Text>
                
                {allFields.map((field: string, i: number) => {
                    const value = hit._source[field];
                    const isExpanded = expanded.has(field);
                    const isObject = typeof value === 'object' && value !== null;
                    const isSelected = i === detailFieldIndex;

                    return (
                        <Box key={field} flexDirection="column" marginY={0}>
                            <Text bold color={isSelected ? 'green' : 'cyan'}>
                                {isSelected ? '▶ ' : '  '}{field}:
                            </Text>
                            {isObject && !isExpanded ? (
                                <Text dimColor>  (press Enter to expand)</Text>
                            ) : null}
                            {(isExpanded || !isObject) && (
                                <Box paddingLeft={2}>
                                    {renderValue(value)}
                                </Box>
                            )}
                            {isObject && isExpanded && (
                                <Text dimColor>  (press Enter to collapse)</Text>
                            )}
                        </Box>
                    );
                })}
            </Box>
        );
    };

    const renderPreviewPanel = () => {
        const file = loadedFiles.get(selectedFileId);
        if (!file) return null;

        const fields = Array.from(new Set<string>());
        file.data.hits.hits.forEach((hit: ElasticSearchHit) => {
            Object.keys(hit._source).forEach((key: string) => fields.push(key));
        });
        const uniqueFields = [...new Set(fields)].sort();

        return (
            <Box flexDirection="column" borderStyle="round" borderColor="cyan" padding={1}>
                <Text bold color="cyan">Preview: {file.name} (ID: {selectedFileId})</Text>
                {file.sortField && (
                    <Text dimColor>Sorted: {file.sortField} {file.sortOrder}</Text>
                )}
                <Text dimColor>↑↓ navigate, Enter details, Esc exit</Text>
                <Text>──────────────────────────────────────</Text>
                
                {file.data.hits.hits.map((hit: ElasticSearchHit, i: number) => {
                    const name = String(hit._source.name || hit._source.isin || hit._id);
                    return (
                    <Box key={i} flexDirection="column" marginY={0}>
                        <Text bold color={i === selectedRow ? 'green' : 'white'}>
                            {i === selectedRow ? '▶ ' : '  '}[{i + 1}] {name}
                        </Text>
                        {i === selectedRow && (
                            <Box paddingLeft={2} flexDirection="column">
                                {uniqueFields.map((field: string) => {
                                    const value = JSON.stringify(hit._source[field] ?? '');
                                    const isKeyField = file.keyField === field;
                                    return (
                                    <Text key={field} bold={isKeyField} color={isKeyField ? 'yellow' : 'dimColor'}>
                                        {isKeyField ? '[KEY] ' : ''}{field}: <Text>{value}</Text>
                                    </Text>
                                    );
                                })}
                            </Box>
                        )}
                    </Box>
                    );
                })}
            </Box>
        );
    };

    const renderComparePanel = () => {
        if (!compareData) return null;
        
        if (compareData.isMatchResult && compareData.matchData) {
            const { matchData, selectedIndex } = compareData;
            const match = matchData[selectedIndex];
            if (!match) return null;
            
            return (
                <Box flexDirection="column" borderStyle="round" borderColor="magenta" padding={1}>
                    <Text bold color="magenta">Match Result ({selectedIndex + 1}/{matchData.length})</Text>
                    <Text dimColor>Key: {match.key} | {match.matched ? 'MATCHED' : 'NOT FOUND'}</Text>
                    <Text>──────────────────────────────────────</Text>
                    
                    <Box flexDirection="column">
                        <Text bold color="cyan">Request:</Text>
                        {Object.entries(match.request).map(([k, v]) => (
                            <Text key={k}>{k}: {JSON.stringify(v)}</Text>
                        ))}
                    </Box>
                    
                    <Text>──────────────────────────────────────</Text>
                    
                    <Box flexDirection="column">
                        <Text bold color={match.matched ? 'green' : 'red'}>
                            {match.matched ? 'Response:' : 'No Response Found'}
                        </Text>
                        {match.response && Object.entries(match.response).map(([k, v]) => (
                            <Text key={k}>{k}: {JSON.stringify(v)}</Text>
                        ))}
                    </Box>
                </Box>
            );
        }
        
        const { fileA, fileB, selectedIndex } = compareData;
        const hitA = fileA.data.hits.hits[selectedIndex];
        const hitB = fileB.data.hits.hits[selectedIndex];
        
        const fieldsA = hitA ? Object.keys(hitA._source).sort() : [];
        const fieldsB = hitB ? Object.keys(hitB._source).sort() : [];
        const allFields = [...new Set([...fieldsA, ...fieldsB])].sort();
        
        return (
            <Box flexDirection="column" borderStyle="round" borderColor="yellow" padding={1}>
                <Text bold color="yellow">Compare: {fileA.name} vs {fileB.name}</Text>
                <Text dimColor>Record {selectedIndex + 1} | ↑↓ navigate, Esc exit</Text>
                <Text>──────────────────────────────────────</Text>
                
                <Box flexDirection="row">
                    <Box flexDirection="column" width="50%" paddingRight={1}>
                        <Text bold color="cyan">{fileA.id}</Text>
                        {hitA ? (
                            allFields.map((field: string) => {
                                const val = hitA._source[field];
                                const valB = hitB?._source[field];
                                const isDifferent = JSON.stringify(val) !== JSON.stringify(valB);
                                return (
                                    <Text key={field} color={isDifferent ? 'red' : 'white'}>
                                        {field}: {JSON.stringify(val)}
                                    </Text>
                                );
                            })
                        ) : <Text dimColor>No record</Text>}
                    </Box>
                    
                    <Box flexDirection="column" width="50%" paddingLeft={1}>
                        <Text bold color="cyan">{fileB.id}</Text>
                        {hitB ? (
                            allFields.map((field: string) => {
                                const val = hitB._source[field];
                                const valA = hitA?._source[field];
                                const isDifferent = JSON.stringify(val) !== JSON.stringify(valA);
                                return (
                                    <Text key={field} color={isDifferent ? 'red' : 'white'}>
                                        {field}: {JSON.stringify(val)}
                                    </Text>
                                );
                            })
                        ) : <Text dimColor>No record</Text>}
                    </Box>
                </Box>
            </Box>
        );
    };

    const renderMainPanel = () => {
        const spinnerFrames = ['|', '/', '-', '\\'];
        return (
        <Box flexDirection="column">
            <Text dimColor>Type / for commands, ↑↓ for history</Text>
            <Text>────────────────</Text>

            {output.length > 0 && (
                <Box flexDirection="column" marginBottom={1}>
                    {isAiRunning && (
                        <Text color="cyan">
                            {spinnerFrames[spinnerFrame]} Processing model stream...
                        </Text>
                    )}
                    {(() => {
                        const maxScroll = Math.max(0, output.length - maxVisibleLines);
                        const effectiveScroll = Math.min(outputScroll, maxScroll);
                        const visibleLines = output.slice(effectiveScroll, effectiveScroll + maxVisibleLines);
                        const showScrollHint = output.length > maxVisibleLines;

                        return (
                            <>
                                {visibleLines.map((line: string, i: number) => (
                                    <Text key={'out-' + (effectiveScroll + i)}>{line}</Text>
                                ))}
                                {showScrollHint && (
                                    <Text dimColor>
                                        [{effectiveScroll + 1}-{Math.min(effectiveScroll + maxVisibleLines, output.length)} of {output.length}] ↑↓ or PgUp/PgDn to scroll
                                    </Text>
                                )}
                            </>
                        );
                    })()}
                </Box>
            )}

            {aiPromptState && (
                <Box flexDirection="column" borderStyle="round" borderDimColor paddingX={1} marginBottom={1}>
                    <Text bold color="magenta">AI Debug: {aiPromptState.title}</Text>
                    <Text dimColor>System: {aiPromptState.system.split('\n')[0].slice(0, 60)}...</Text>
                    <Text dimColor>User: {aiPromptState.user.slice(0, 60)}...</Text>
                    {aiPromptState.context && (
                        <Text dimColor>Context: {aiPromptState.context.split('\n').length} lines</Text>
                    )}
                    {aiPromptState.fullPrompt && (
                        <Text dimColor>Full Prompt: {aiPromptState.fullPrompt.split('\n').length} lines</Text>
                    )}
                </Box>
            )}

            <Box marginTop={1}>
                <Text bold color="green">{'>'} </Text>
                <TextInput
                    key={inputKey}
                    value={input}
                    onChange={handleInputChange}
                    onSubmit={handleSubmit}
                    placeholder="Type command..."
                />
            </Box>

            <Box flexDirection="column" marginTop={1} minHeight={SUGGESTION_PANEL_HEIGHT}>
                {suggestions.length > 0 ? (
                    <>
                        <Text dimColor>
                            {suggestionState.kind === 'model'
                                ? 'Models: (Tab/Arrows to cycle)'
                                : suggestionState.kind === 'data-file'
                                    ? 'Data files: (Tab/Arrows to cycle)'
                                    : suggestionState.kind === 'loaded-file'
                                        ? 'Files: (Tab/Arrows to cycle)'
                                        : suggestionState.kind === 'option'
                                            ? 'Options: (Tab/Arrows to cycle)'
                                            : 'Commands: (Tab/Arrows to cycle)'}
                        </Text>
                        {suggestions.slice(0, SUGGESTION_PANEL_HEIGHT - 1).map((item: string, i: number) => {
                            const isActive = i === suggestionIndex;
                            return (
                                <Text key={'sug-' + i} bold={isActive} color={isActive ? 'green' : 'yellow'}>
                                    {isActive ? '▸ ' : '  '}{item}
                                </Text>
                            );
                        })}
                    </>
                ) : (
                    Array.from({ length: SUGGESTION_PANEL_HEIGHT }).map((_, i) => (
                        <Text key={`sug-empty-${i}`}> </Text>
                    ))
                )}
            </Box>

        </Box>
        );
    };

    const renderKeyFieldPanel = () => {
        const file = loadedFiles.get(selectedFileId);
        if (!file) return null;
        
        const hit = file.data.hits.hits[0];
        if (!hit) return null;
        const fields = Object.keys(hit._source).sort();
        
        return (
            <Box flexDirection="column" borderStyle="round" borderColor="green" padding={1}>
                <Text bold color="green">Select Key Field: {file.name} (ID: {selectedFileId})</Text>
                <Text dimColor>↑↓ navigate, Enter to select, Esc cancel</Text>
                <Text>──────────────────────────────────────</Text>
                
                {fields.map((field: string, i: number) => {
                    const isSelected = i === detailFieldIndex;
                    const isCurrentKey = file.keyField === field;
                    
                    return (
                        <Box key={field} marginY={0}>
                            <Text bold color={isSelected ? 'green' : (isCurrentKey ? 'yellow' : 'white')}>
                                {isSelected ? '▶ ' : '  '}{isCurrentKey ? '[KEY] ' : ''}{field}
                            </Text>
                        </Box>
                    );
                })}
            </Box>
        );
    };

    const renderSchemaPanel = () => {
        const file = loadedFiles.get(selectedFileId);
        if (!file?.schema) return null;

        const subSchemaEntries = Object.entries(file.subSchemas || {}).sort(([a], [b]) => a.localeCompare(b));
        const selectedPaths = new Set(file.selectedSubSchemaPaths || []);
        const activeEntry = subSchemaEntries[schemaSelectionIndex];
        const activeSchema = activeEntry?.[1] || file.schema;

        const formatSchema = (schema: JsonSchema): string[] =>
            JSON.stringify(schema, null, 2).split('\n').slice(0, 18);

        return (
            <Box flexDirection="column" borderStyle="round" borderColor="blue" padding={1}>
                <Text bold color="blue">Schema: {file.name} (ID: {selectedFileId})</Text>
                <Text dimColor>↑↓ navigate, Space/Enter toggle sub-schema, Esc exit</Text>
                <Text dimColor>Selected sub-schemas: {selectedPaths.size}</Text>
                <Text>──────────────────────────────────────</Text>
                <Text bold color="cyan">Root Schema</Text>
                {formatSchema(file.schema).map((line, index) => (
                    <Text key={`root-${index}`}>{line}</Text>
                ))}
                <Text>──────────────────────────────────────</Text>
                <Text bold color="green">Sub-Schemas</Text>
                {subSchemaEntries.length === 0 && (
                    <Text dimColor>No sub-schemas extracted</Text>
                )}
                {subSchemaEntries.map(([path], index) => (
                    <Text key={path} color={index === schemaSelectionIndex ? 'yellow' : 'white'}>
                        {index === schemaSelectionIndex ? '▶ ' : '  '}
                        {selectedPaths.has(path) ? '[x] ' : '[ ] '}
                        {path}
                    </Text>
                ))}
                <Text>──────────────────────────────────────</Text>
                <Text bold color="magenta">Active Sub-Schema</Text>
                {formatSchema(activeSchema).map((line, index) => (
                    <Text key={`active-${index}`}>{line}</Text>
                ))}
            </Box>
        );
    };

    const renderNestedPanel = () => {
        if (!nestedPath) return null;
        
        const keys = Object.keys(nestedPath.data).sort();
        
        return (
            <Box flexDirection="column" borderStyle="round" borderColor="magenta" padding={1}>
                <Text bold color="magenta">Nested: {nestedPath.fieldName}</Text>
                <Text dimColor>↑↓ navigate, Enter for deeper, Esc back</Text>
                <Text>──────────────────────────────────────</Text>
                
                {keys.map((k: string, i: number) => {
                    const v = nestedPath.data[k];
                    const isSelected = i === detailFieldIndex;
                    const isExpandable = typeof v === 'object' && v !== null && !Array.isArray(v);
                    
                    return (
                        <Box key={k} flexDirection="column" marginY={0}>
                            <Text bold color={isSelected ? 'green' : 'cyan'}>
                                {isSelected ? '▶ ' : '  '}{k}:
                            </Text>
                            {!isExpandable && (
                                <Text>  {JSON.stringify(v)}</Text>
                            )}
                            {isExpandable && (
                                <Text dimColor>  (press Enter to drill down)</Text>
                            )}
                        </Box>
                    );
                })}
            </Box>
        );
    };

    return (
        <Box flexDirection="column">
            <BannerComponent banner={bannerStyle} onCycle={cycleBanner} />
            <StatusLine model={model} historyCount={history.length} fileCount={loadedFiles.size} />
            {viewMode === 'main' && renderMainPanel()}
            {viewMode === 'preview' && renderPreviewPanel()}
            {viewMode === 'detail' && renderDetailPanel()}
            {viewMode === 'nested' && renderNestedPanel()}
            {(viewMode === 'compare' || viewMode === 'match') && renderComparePanel()}
            {viewMode === 'keyfield' && renderKeyFieldPanel()}
            {viewMode === 'schema' && renderSchemaPanel()}
        </Box>
    );
};

const { unmount } = render(<App />);
globalUnmount = unmount;
