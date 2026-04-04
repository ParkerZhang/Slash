import { ElasticSearchHit, ElasticSearchResult, LoadedFile } from './types.js';

/**
 * Finds records in file1 that are NOT in file2 based on key fields.
 * Key fields are auto-detected from: isin, currency, exchange_code (preferred order),
 * or falls back to the file's manually set keyField.
 */
export function diffFiles(
    file1: LoadedFile,
    file2: LoadedFile,
    resultId: string = 'diff_result'
): { result: ElasticSearchResult; newId: string; recordCount: number } | string {
    const getKeyFields = (file: LoadedFile): string[] => {
        if (!file || file.data.hits.hits.length === 0) return [];
        const headers = Object.keys(file.data.hits.hits[0]._source);
        return headers.filter(
            (h: string) => ['isin', 'currency', 'exchange_code'].includes(h.toLowerCase())
        );
    };

    const keyFields1 = getKeyFields(file1);
    const keyFields2 = getKeyFields(file2);

    const useKeyFields1 =
        keyFields1.length > 0 ? keyFields1 : file1.keyField ? [file1.keyField] : [];
    const useKeyFields2 =
        keyFields2.length > 0 ? keyFields2 : file2.keyField ? [file2.keyField] : [];

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
        return fields.map((f) => String(source[f] || '')).join('|');
    };

    const keys2Set = new Set(
        file2.data.hits.hits.map((h) => getKey(h._source, sortedKeyFields2))
    );

    const inF1NotF2 = file1.data.hits.hits.filter(
        (h) => !keys2Set.has(getKey(h._source, sortedKeyFields1))
    );

    const newData: ElasticSearchResult = {
        took: 0,
        timed_out: false,
        hits: {
            total: { value: inF1NotF2.length, relation: 'eq' },
            max_score: 1.0,
            hits: inF1NotF2,
        },
    };

    return { result: newData, newId: resultId, recordCount: inF1NotF2.length };
}
