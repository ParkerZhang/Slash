import React, { useState, useEffect } from 'react';
import { render, Text, Box, useInput } from 'ink';
import TextInput from 'ink-text-input';
import { addCommand, getHistory, getLoadedFiles, addLoadedFile, updateKeyField } from './historyManager.js';
import alasql from 'alasql';
import * as fs from 'fs';
import * as path from 'path';

interface ElasticSearchHit {
    _index: string;
    _id: string;
    _score: number;
    _source: Record<string, unknown>;
}

interface ElasticSearchResult {
    took: number;
    timed_out: boolean;
    hits: {
        total: { value: number; relation: string };
        max_score: number;
        hits: ElasticSearchHit[];
    };
}

interface LoadedFile {
    id: string;
    name: string;
    data: ElasticSearchResult;
    keyField?: string;
    sortField?: string;
    sortOrder?: 'asc' | 'desc';
}

type ViewMode = 'main' | 'preview' | 'detail' | 'nested' | 'compare' | 'match' | 'keyfield';

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

const App = () => {
    const [history, setHistory] = useState<string[]>([]);
    const [input, setInput] = useState('');
    const [output, setOutput] = useState<string[]>([]);
    const [historyIndex, setHistoryIndex] = useState(-1);
    const [loadedFiles, setLoadedFiles] = useState<Map<string, LoadedFile>>(new Map());
    const [viewMode, setViewMode] = useState<ViewMode>('main');
    const [selectedFileId, setSelectedFileId] = useState<string>('');
    const [selectedRow, setSelectedRow] = useState(0);
    const [detailFieldIndex, setDetailFieldIndex] = useState(0);
    const [expandedFields, setExpandedFields] = useState<Record<number, Set<string>>>({});
    const [nestedPath, setNestedPath] = useState<NestedPath | null>(null);
    const [compareData, setCompareData] = useState<CompareData | null>(null);
    const [keyFieldFileId, setKeyFieldFileId] = useState<string>('');
    const [suggestionIndex, setSuggestionIndex] = useState(-1);
    const [inputKey, setInputKey] = useState(0);

    useEffect(() => {
        const loadedHistory = getHistory();
        setHistory(loadedHistory);

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
    const fileCommandNames = new Set(['/preview', '/compare', '/match', '/keyfield', '/sort', '/minus', '/diff', '/save']);

    useInput((inputKey, key) => {
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
                setLoadedFiles(prev => {
                    const newMap = new Map(prev);
                    const f = newMap.get(keyFieldFileId);
                    if (f) {
                        newMap.set(keyFieldFileId, { ...f, keyField: field });
                    }
                    return newMap;
                });
                updateKeyField(keyFieldFileId, field);
                setOutput(['Key field set to: ' + field]);
                setViewMode('main');
                setKeyFieldFileId('');
            } else if (key.escape) {
                setViewMode('main');
                setKeyFieldFileId('');
            }
        } else {
            if (key.upArrow && history.length > 0) {
                const newIndex = historyIndex < history.length - 1 ? historyIndex + 1 : historyIndex;
                setHistoryIndex(newIndex);
                setInput(history[history.length - 1 - newIndex] || '');
                setSuggestionIndex(-1);
            } else if (key.downArrow) {
                if (historyIndex > 0) {
                    const newIndex = historyIndex - 1;
                    setHistoryIndex(newIndex);
                    setInput(history[history.length - 1 - newIndex] || '');
                    setSuggestionIndex(-1);
                } else if (historyIndex === 0) {
                    setHistoryIndex(-1);
                    setInput('');
                    setSuggestionIndex(-1);
                }
            } else if (key.tab) {
                const parts = input.trim().split(/\s+/);
                const cmd = parts[0].toLowerCase();
                const hasTrailingSpace = input.endsWith(' ');
                const isLoadCommand = cmd === '/load' || cmd === '/loadcsv';
                const isLoadWithArgs = isLoadCommand && (parts.length >= 2 || hasTrailingSpace);
                const isFileSuggestion = parts.length >= 2 && fileCommandNames.has(cmd);

                let currentSuggestions: string[];
                if (isLoadWithArgs) {
                    const dataFiles = getDataFiles();
                    const query = hasTrailingSpace && parts.length < 2 ? '' : parts[parts.length - 1].toLowerCase();
                    currentSuggestions = dataFiles.filter(f => f.toLowerCase().startsWith(query));
                } else if (isFileSuggestion) {
                    const loadedIds = Array.from(loadedFiles.keys());
                    const query = parts[parts.length - 1].toLowerCase();
                    currentSuggestions = loadedIds.filter(id => id.toLowerCase().startsWith(query));
                } else {
                    currentSuggestions = commands
                        .filter(cmd => cmd.name.startsWith(input.toLowerCase()))
                        .map(cmd => cmd.name);
                }

                if (currentSuggestions.length > 0) {
                    const nextIndex = suggestionIndex < currentSuggestions.length - 1 ? suggestionIndex + 1 : 0;
                    setSuggestionIndex(nextIndex);

                    if (isFileSuggestion) {
                        // Replace only the last part (the file ID being typed)
                        const prefix = parts.slice(0, -1).join(' ');
                        setInput(prefix + ' ' + currentSuggestions[nextIndex]);
                    } else {
                        setInput(currentSuggestions[nextIndex]);
                    }
                    setInputKey(k => k + 1);
                }
            }
        }
    });

    const commands = [
        { name: '/help', description: 'Show available commands', execute: () => 'Available: /help, /clear, /load, /loadCsv, /preview, /compare, /match, /keyfield, /sort, /minus, /diff, /sql, /save, /files, /exit' },
        { name: '/clear', description: 'Clear command history', execute: () => '__CLEAR__' },
        { name: '/load', description: 'Load JSON (Usage: /load <id> <file>)', execute: (args: string, setLoadedFiles: React.Dispatch<React.SetStateAction<Map<string, LoadedFile>>>) => {
            const parts = args.split(' ');
            if (parts.length < 2) return 'Usage: /load <id> <filepath>';
            const fileId = parts[0];
            const fileName = parts.slice(1).join(' ');
            
            try {
                const content = fs.readFileSync(resolveDataPath(fileName), 'utf-8');
                const data = JSON.parse(content) as ElasticSearchResult;
                setLoadedFiles(prev => {
                    const newMap = new Map(prev);
                    newMap.set(fileId, { id: fileId, name: fileName, data });
                    return newMap;
                });
                addLoadedFile(fileId, fileName);
                return "Loaded " + fileName + " as '" + fileId + "' (" + data.hits.hits.length + " hits)";
            } catch (error) {
                return "Error: " + (error instanceof Error ? error.message : String(error));
            }
        }},
        { name: '/loadCsv', description: 'Load CSV (Usage: /loadCsv <id> <file> [sep])', execute: (args: string, setLoadedFiles: React.Dispatch<React.SetStateAction<Map<string, LoadedFile>>>) => {
            const parts = args.trim().split(' ');
            if (parts.length < 2) return 'Usage: /loadCsv <id> <filepath> [sep]';
            const fileId = parts[0];
            const fileName = parts.slice(1, -1).join(' ') || parts[1];
            const separator = parts.length > 2 ? parts[parts.length - 1] : '|';
            
            if (separator.length !== 1) return 'Separator must be a single character';
            
            try {
                const content = fs.readFileSync(resolveDataPath(fileName), 'utf-8');
                const lines = content.trim().split('\n');
                if (lines.length < 2) return 'CSV must have header and at least one row';
                
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
                
                setLoadedFiles(prev => {
                    const newMap = new Map(prev);
                    newMap.set(fileId, { id: fileId, name: fileName, data });
                    return newMap;
                });
                addLoadedFile(fileId, fileName);
                return "Loaded " + fileName + " as '" + fileId + "' (" + hits.length + " rows)";
            } catch (error) {
                return "Error: " + (error instanceof Error ? error.message : String(error));
            }
        }},
        { name: '/preview', description: 'Preview loaded file (Usage: /preview <id>)', execute: (args: string, _setLoadedFiles: unknown, loadedFiles: Map<string, LoadedFile>) => {
            if (!args.trim()) {
                const ids = Array.from(loadedFiles.keys());
                if (ids.length === 0) return 'No files loaded';
                return "Loaded: " + ids.join(', ') + "\nUsage: /preview <id>";
            }
            const fileId = args.trim();
            if (!loadedFiles.has(fileId)) return "File '" + fileId + "' not loaded";
            setSelectedFileId(fileId);
            setSelectedRow(0);
            setViewMode('preview');
            return '';
        }},
        { name: '/files', description: 'List loaded files', execute: () => {
            const ids = Array.from(loadedFiles.keys());
            if (ids.length === 0) return 'No files loaded';
            return "Loaded: " + ids.map((id: string) => id + ": " + loadedFiles.get(id)?.name).join(', ');
        }},
        { name: '/compare', description: 'Compare two files (Usage: /compare <id1> <id2>)', execute: (args: string, _setLoadedFiles: unknown, loadedFiles: Map<string, LoadedFile>) => {
            const parts = args.trim().split(' ');
            if (parts.length < 2) return 'Usage: /compare <id1> <id2>';
            const id1 = parts[0];
            const id2 = parts[1];
            const fileA = loadedFiles.get(id1);
            const fileB = loadedFiles.get(id2);
            if (!fileA) return "File '" + id1 + "' not loaded";
            if (!fileB) return "File '" + id2 + "' not loaded";
            setCompareData({ fileA, fileB, selectedIndex: 0 });
            setViewMode('compare');
            return '';
        }},
        { name: '/match', description: 'Match request vs response (Usage: /match <reqId> <respId>)', execute: (args: string, _setLoadedFiles: unknown, loadedFiles: Map<string, LoadedFile>) => {
            const parts = args.trim().split(' ');
            if (parts.length < 2) return 'Usage: /match <requestId> <responseId>';
            const reqId = parts[0];
            const respId = parts[1];
            const reqFile = loadedFiles.get(reqId);
            const respFile = loadedFiles.get(respId);
            if (!reqFile) return "File '" + reqId + "' not loaded";
            if (!respFile) return "File '" + respId + "' not loaded";
            
            const reqHits = reqFile.data.hits.hits;
            const respHits = respFile.data.hits.hits;
            
            const reqHeaders = reqHits.length > 0 ? Object.keys(reqHits[0]._source).sort() : [];
            
            const keyFields = reqHeaders.filter((h: string) => ['isin', 'currency', 'exchange_code'].includes(h.toLowerCase()));
            
            if (keyFields.length === 0) return 'Request file must have ISIN, Currency, or ExchangeCode fields';
            
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
            
            setCompareData({
                fileA: reqFile,
                fileB: respFile,
                selectedIndex: 0,
                isMatchResult: true,
                matchData: matches
            });
            setViewMode('match');
            return '';
        }},
        { name: '/keyfield', description: 'Set key field (Usage: /keyfield <id>)', execute: (args: string, _setLoadedFiles: unknown, loadedFiles: Map<string, LoadedFile>) => {
            const fileId = args.trim();
            if (!fileId) return 'Usage: /keyfield <id>';
            const file = loadedFiles.get(fileId);
            if (!file) return "File '" + fileId + "' not loaded";
            setKeyFieldFileId(fileId);
            setDetailFieldIndex(0);
            setViewMode('keyfield');
            return '';
        }},
        { name: '/sort', description: 'Sort file (Usage: /sort <id> [field] [asc|desc])', execute: (args: string, setLoadedFiles: React.Dispatch<React.SetStateAction<Map<string, LoadedFile>>>, loadedFiles: Map<string, LoadedFile>) => {
            const parts = args.trim().split(' ');
            if (parts.length < 1) return 'Usage: /sort <id> [field] [asc|desc]';
            const fileId = parts[0];
            
            const file = loadedFiles.get(fileId);
            if (!file) return "File '" + fileId + "' not loaded";
            
            const field = parts[1] || file.keyField;
            if (!field) return 'No keyfield set. Usage: /sort <id> [field] [asc|desc]';
            
            let order: 'asc' | 'desc' = 'asc';
            const lastPart = parts[parts.length - 1];
            if (lastPart === 'asc' || lastPart === 'desc') {
                order = lastPart;
            } else if (lastPart !== field && parts.length > 2) {
                return 'Invalid order. Use asc or desc';
            }
            
            const hit = file.data.hits.hits[0];
            if (!hit) return 'File is empty';
            const fields = Object.keys(hit._source).sort();
            if (!fields.includes(field)) return "Field '" + field + "' not found. Available: " + fields.join(', ');
            
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
            
            setLoadedFiles(prev => {
                const newMap = new Map(prev);
                newMap.set(fileId, { ...file, data: newData, sortField: field, sortOrder: order });
                return newMap;
            });
            
            return "Sorted " + fileId + " by " + field + " (" + order + ")";
        }},
        { name: '/minus', description: 'Minus files (Usage: /minus <id1> <id2>)', execute: (args: string, setLoadedFiles: React.Dispatch<React.SetStateAction<Map<string, LoadedFile>>>, loadedFiles: Map<string, LoadedFile>) => {
            const parts = args.trim().split(' ');
            if (parts.length < 2) return 'Usage: /minus <id1> <id2>';
            const id1 = parts[0];
            const id2 = parts[1];
            
            const file1 = loadedFiles.get(id1);
            const file2 = loadedFiles.get(id2);
            if (!file1) return "File '" + id1 + "' not loaded";
            if (!file2) return "File '" + id2 + "' not loaded";
            
            const getKeyFields = (file: typeof file1): string[] => {
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
                return 'Both files need keyfield set. Use /keyfield <id>';
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
            setLoadedFiles(prev => {
                const newMap = new Map(prev);
                newMap.set(newId, { id: newId, name: file1.name + ' - ' + file2.name, data: newData });
                return newMap;
            });
            
            return "Created '" + newId + "' with " + minusHits.length + " records";
        }},
        { name: '/diff', description: 'Diff files - records in f1 missing from f2 (Usage: /diff <id1> <id2>)', execute: (args: string, setLoadedFiles: React.Dispatch<React.SetStateAction<Map<string, LoadedFile>>>, loadedFiles: Map<string, LoadedFile>) => {
            const parts = args.trim().split(' ');
            if (parts.length < 2) return 'Usage: /diff <id1> <id2>';
            const id1 = parts[0];
            const id2 = parts[1];

            const file1 = loadedFiles.get(id1);
            const file2 = loadedFiles.get(id2);
            if (!file1) return "File '" + id1 + "' not loaded";
            if (!file2) return "File '" + id2 + "' not loaded";

            const getKeyFields = (file: LoadedFile): string[] => {
                if (!file || file.data.hits.hits.length === 0) return [];
                const headers = Object.keys(file.data.hits.hits[0]._source);
                return headers.filter((h: string) => ['isin', 'currency', 'exchange_code'].includes(h.toLowerCase()));
            };

            const keyFields1 = getKeyFields(file1);
            const keyFields2 = getKeyFields(file2);

            const useKeyFields1 = keyFields1.length > 0 ? keyFields1 : (file1.keyField ? [file1.keyField] : []);
            const useKeyFields2 = keyFields2.length > 0 ? keyFields2 : (file2.keyField ? [file2.keyField] : []);

            if (useKeyFields1.length === 0 || useKeyFields2.length === 0) {
                return 'Both files need keyfield set. Use /keyfield <id>';
            }

            const sortByPreferred = (arr: string[]): string[] => {
                const preferredOrder = ['isin', 'currency', 'exchange_code'];
                return arr.slice().sort((a, b) => {
                    const ia = preferredOrder.indexOf(a.toLowerCase());
                    const ib = preferredOrder.indexOf(b.toLowerCase());
                    return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
                });
            };

            const sortedKeyFields1 = sortByPreferred(useKeyFields1);
            const sortedKeyFields2 = sortByPreferred(useKeyFields2);

            const getKey = (source: Record<string, unknown>, fields: string[]): string => {
                return fields.map(f => String(source[f] || '')).join('|');
            };

            const keys2Set = new Set(file2.data.hits.hits.map(h => getKey(h._source, sortedKeyFields2)));

            const inF1NotF2 = file1.data.hits.hits.filter(h => !keys2Set.has(getKey(h._source, sortedKeyFields1)));

            const newData: ElasticSearchResult = {
                took: 0,
                timed_out: false,
                hits: {
                    total: { value: inF1NotF2.length, relation: 'eq' },
                    max_score: 1.0,
                    hits: inF1NotF2
                }
            };

            const newId = id1 + '-diff-' + id2;
            setLoadedFiles(prev => {
                const newMap = new Map(prev);
                newMap.set(newId, { id: newId, name: id1 + ' diff ' + id2, data: newData });
                return newMap;
            });

            return "Created '" + newId + "' with " + inF1NotF2.length + " records";
        }},
        // TODO: Fix SQL issues, sql not accurate
        { name: '/sql', description: 'SQL query (Usage: /sql "SELECT * FROM f1")', execute: (args: string, setLoadedFiles: React.Dispatch<React.SetStateAction<Map<string, LoadedFile>>>, loadedFiles: Map<string, LoadedFile>) => {
            const query = args.trim();
            if (!query) return 'Usage: /sql "SELECT * FROM f1"';
            
            try {
                alasql('DROP TABLE IF EXISTS _tables');
                alasql('CREATE TABLE _tables (id VARCHAR(255), name VARCHAR(255))');
                
                for (const [id, file] of loadedFiles) {
                    const tableData = file.data.hits.hits.map(h => h._source);
                    alasql('DROP TABLE IF EXISTS ' + id);
                    alasql('CREATE TABLE ' + id + ' (' + Object.keys(tableData[0] || {}).map(c => c + ' VARCHAR(255)').join(', ') + ')');
                    
                    if (tableData.length > 0) {
                        alasql('INSERT INTO ' + id + ' SELECT * FROM ?', [tableData]);
                    }
                }
                
                const results = alasql(query);
                
                if (!Array.isArray(results) || results.length === 0) {
                    return 'No results found';
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
                setLoadedFiles(prev => {
                    const newMap = new Map(prev);
                    newMap.set(newId, { id: newId, name: 'SQL Result', data: newData });
                    return newMap;
                });
                
                return "Created '" + newId + "' with " + results.length + " records";
            } catch (error) {
                return "SQL Error: " + (error instanceof Error ? error.message : String(error));
            }
        }},
        { name: '/save', description: 'Save file (Usage: /save <id> <filepath>)', execute: (args: string, _setLoadedFiles: unknown, loadedFiles: Map<string, LoadedFile>) => {
            const parts = args.trim().split(' ');
            if (parts.length < 2) return 'Usage: /save <id> <filepath>';
            const fileId = parts[0];
            const fileName = parts.slice(1).join(' ');
            
            const file = loadedFiles.get(fileId);
            if (!file) return "File '" + fileId + "' not loaded";
            
            try {
                const data = file.data.hits.hits.map(h => h._source);
                fs.writeFileSync(fileName, JSON.stringify(data, null, 2), 'utf-8');
                return "Saved " + fileId + " to " + fileName;
            } catch (error) {
                return "Error: " + (error instanceof Error ? error.message : String(error));
            }
        }},
        { name: '/exit', description: 'Exit application', execute: () => '__EXIT__' }
    ];

    const suggestions = input.startsWith('/')
        ? (() => {
            const parts = input.trim().split(/\s+/);
            const cmd = parts[0].toLowerCase();
            const hasTrailingSpace = input.endsWith(' ');

            // /load and /loadCsv: suggest data files
            if ((cmd === '/load' || cmd === '/loadcsv') && (parts.length >= 2 || hasTrailingSpace)) {
                const dataFiles = getDataFiles();
                const query = hasTrailingSpace && parts.length < 2 ? '' : parts[parts.length - 1].toLowerCase();
                return dataFiles.filter(f => f.toLowerCase().startsWith(query));
            }

            // Commands expecting loaded file IDs
            if (parts.length >= 2 && fileCommandNames.has(cmd)) {
                const loadedIds = Array.from(loadedFiles.keys());
                const query = parts[parts.length - 1].toLowerCase();
                return loadedIds.filter(id => id.toLowerCase().startsWith(query));
            }
            // Otherwise suggest command names (as strings)
            return commands
                .filter(cmd => cmd.name.startsWith(input.toLowerCase()))
                .map(cmd => cmd.name);
        })()
        : [];

    // Reset suggestion index when suggestions change
    useEffect(() => {
        if (suggestions.length === 0) {
            setSuggestionIndex(-1);
        } else if (suggestionIndex >= suggestions.length) {
            setSuggestionIndex(0);
        }
    }, [input, suggestions.length]);

    const handleSubmit = (value: string) => {
        if (!value.trim()) return;

        const commandsList = value.split(/\s*&&\s*/).map(s => s.trim()).filter(s => s);
        
        for (const cmdValue of commandsList) {
            const newHistory = [...history, cmdValue];
            setHistory(newHistory);
            addCommand(cmdValue);
            setHistoryIndex(-1);

            if (cmdValue.startsWith('/')) {
                const parts = cmdValue.split(' ');
                const cmdName = parts[0].toLowerCase();
                const args = parts.slice(1).join(' ');
                const cmd = commands.find(c => c.name.toLowerCase() === cmdName);

                if (cmd) {
                    let result = cmd.execute(args, setLoadedFiles, loadedFiles);
                    if (result === '__CLEAR__') {
                        setHistory([]);
                        setOutput(['History cleared']);
                    } else if (result === '__EXIT__') {
                        if (globalUnmount) globalUnmount();
                        process.exit(0);
                    } else if (result === '') {
                        // preview mode - no output
                    } else {
                        setOutput(result.split('\n'));
                    }
                } else {
                    setOutput(["Unknown: " + cmdName]);
                }
            } else {
                setOutput(["Echo: " + cmdValue]);
            }
        }
        setInput('');
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
        const bannerLines = [
            '皇帝立国，维初在昔，嗣世称王',
            '讨伐乱逆，威动四极，武义直方',
            '戎臣奉诏，经时不久，灭六暴强',
            '廿有六年，上荐高号，孝道显明',
        ];

        return (
        <Box flexDirection="column">
            <Text dimColor>Type / for commands, ↑↓ for history</Text>
            <Text>────────────────</Text>

            {output.length > 0 && (
                <Box flexDirection="column">
                    {output.slice(-8).map((line: string, i: number) => (
                        <Text key={'out-' + i}>{line}</Text>
                    ))}
                </Box>
            )}

            <Box marginTop={1}>
                <Text bold color="green">{'>'} </Text>
                <TextInput
                    key={inputKey}
                    value={input}
                    onChange={setInput}
                    onSubmit={handleSubmit}
                    placeholder="Type command..."
                />
            </Box>

            {suggestions.length > 0 && (
                <Box flexDirection="column" marginTop={1}>
                    {(() => {
                        const parts = input.trim().split(/\s+/);
                        const cmd = parts[0].toLowerCase();
                        const hasTrailingSpace = input.endsWith(' ');
                        const isLoadCommand = cmd === '/load' || cmd === '/loadcsv';
                        const isFileSuggestion = (isLoadCommand && (parts.length >= 2 || hasTrailingSpace)) ||
                            (parts.length >= 2 && fileCommandNames.has(cmd));
                        return (
                            <>
                                <Text dimColor>{isFileSuggestion ? 'Files: (Tab to cycle)' : 'Commands: (Tab to cycle)'}</Text>
                                {suggestions.map((item: string, i: number) => {
                                    const isActive = i === suggestionIndex;
                                    return (
                                        <Text key={'sug-' + i} bold={isActive} color={isActive ? 'green' : 'yellow'}>
                                            {isActive ? '▸ ' : '  '}{item}
                                        </Text>
                                    );
                                })}
                            </>
                        );
                    })()}
                </Box>
            )}

            <Box marginTop={1}>
                <Text dimColor>History: {history.length} | Files: {loadedFiles.size}</Text>
            </Box>
        </Box>
        );
    };

    const renderKeyFieldPanel = () => {
        const file = loadedFiles.get(keyFieldFileId);
        if (!file) return null;
        
        const hit = file.data.hits.hits[0];
        if (!hit) return null;
        const fields = Object.keys(hit._source).sort();
        console.error('KeyField fields:', fields);
        
        return (
            <Box flexDirection="column" borderStyle="round" borderColor="green" padding={1}>
                <Text bold color="green">Select Key Field: {file.name} (ID: {keyFieldFileId})</Text>
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
            <Banner />
            {viewMode === 'main' && renderMainPanel()}
            {viewMode === 'preview' && renderPreviewPanel()}
            {viewMode === 'detail' && renderDetailPanel()}
            {viewMode === 'nested' && renderNestedPanel()}
            {(viewMode === 'compare' || viewMode === 'match') && renderComparePanel()}
            {viewMode === 'keyfield' && renderKeyFieldPanel()}
        </Box>
    );
};

const { unmount } = render(<App />);
globalUnmount = unmount;
