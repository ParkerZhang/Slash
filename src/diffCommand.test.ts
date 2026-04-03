import { describe, it, expect } from 'vitest';
import { diffFiles } from './diffCommand.js';
import type { LoadedFile, ElasticSearchResult } from './diffCommand.js';

// Helper to create a minimal LoadedFile
function createLoadedFile(
    id: string,
    name: string,
    records: Record<string, unknown>[],
    keyField?: string
): LoadedFile {
    const data: ElasticSearchResult = {
        took: 0,
        timed_out: false,
        hits: {
            total: { value: records.length, relation: 'eq' },
            max_score: 1.0,
            hits: records.map((source, idx) => ({
                _index: 'test',
                _id: String(idx + 1),
                _score: 1.0,
                _source: source,
            })),
        },
    };
    return { id, name, data, keyField };
}

describe('diffFiles', () => {
    describe('basic diff functionality', () => {
        it('should find records in file1 not present in file2 by isin', () => {
            const file1 = createLoadedFile('f1', 'file1.csv', [
                { isin: 'US0001', price: 100 },
                { isin: 'US0002', price: 200 },
                { isin: 'US0003', price: 300 },
            ]);
            const file2 = createLoadedFile('f2', 'file2.csv', [
                { isin: 'US0001', price: 105 },
                { isin: 'US0002', price: 195 },
            ]);

            const result = diffFiles(file1, file2);
            expect(result).not.toBe('string');
            if (typeof result !== 'string') {
                expect(result.recordCount).toBe(1);
                expect(result.result.hits.hits[0]._source.isin).toBe('US0003');
            }
        });

        it('should return all records when file2 is empty key set', () => {
            const file1 = createLoadedFile('f1', 'file1.csv', [
                { isin: 'US0001', price: 100 },
                { isin: 'US0002', price: 200 },
            ]);
            const file2 = createLoadedFile('f2', 'file2.csv', [
                { isin: 'US0099', price: 50 },
            ]);

            const result = diffFiles(file1, file2);
            if (typeof result !== 'string') {
                expect(result.recordCount).toBe(2);
            }
        });

        it('should return empty result when files are identical', () => {
            const records = [
                { isin: 'US0001', price: 100 },
                { isin: 'US0002', price: 200 },
            ];
            const file1 = createLoadedFile('f1', 'file1.csv', records);
            const file2 = createLoadedFile('f2', 'file2.csv', records);

            const result = diffFiles(file1, file2);
            if (typeof result !== 'string') {
                expect(result.recordCount).toBe(0);
            }
        });

        it('should handle file1 with no unique records', () => {
            const file1 = createLoadedFile('f1', 'file1.csv', [
                { isin: 'US0001', price: 100 },
            ]);
            const file2 = createLoadedFile('f2', 'file2.csv', [
                { isin: 'US0001', price: 100 },
                { isin: 'US0002', price: 200 },
            ]);

            const result = diffFiles(file1, file2);
            if (typeof result !== 'string') {
                expect(result.recordCount).toBe(0);
            }
        });
    });

    describe('multiple key fields', () => {
        it('should use isin + currency as composite key', () => {
            const file1 = createLoadedFile('f1', 'file1.csv', [
                { isin: 'US0001', currency: 'USD', price: 100 },
                { isin: 'US0001', currency: 'EUR', price: 90 },
                { isin: 'US0002', currency: 'USD', price: 200 },
            ]);
            const file2 = createLoadedFile('f2', 'file2.csv', [
                { isin: 'US0001', currency: 'USD', price: 105 },
                { isin: 'US0002', currency: 'USD', price: 195 },
            ]);

            const result = diffFiles(file1, file2);
            if (typeof result !== 'string') {
                expect(result.recordCount).toBe(1);
                expect(result.result.hits.hits[0]._source.currency).toBe('EUR');
            }
        });

        it('should prefer isin over currency in key ordering', () => {
            const file1 = createLoadedFile('f1', 'file1.csv', [
                { currency: 'USD', isin: 'US0001', price: 100 },
                { currency: 'EUR', isin: 'US0001', price: 90 },
            ]);
            const file2 = createLoadedFile('f2', 'file2.csv', [
                { currency: 'USD', isin: 'US0001', price: 105 },
            ]);

            const result = diffFiles(file1, file2);
            if (typeof result !== 'string') {
                expect(result.recordCount).toBe(1);
            }
        });
    });

    describe('manual keyField fallback', () => {
        it('should use manual keyField when auto-detect finds no standard fields', () => {
            const file1 = createLoadedFile(
                'f1',
                'file1.csv',
                [
                    { ticker: 'AAPL', price: 100 },
                    { ticker: 'GOOG', price: 200 },
                    { ticker: 'MSFT', price: 300 },
                ],
                'ticker'
            );
            const file2 = createLoadedFile(
                'f2',
                'file2.csv',
                [
                    { ticker: 'AAPL', price: 105 },
                    { ticker: 'GOOG', price: 195 },
                ],
                'ticker'
            );

            const result = diffFiles(file1, file2);
            if (typeof result !== 'string') {
                expect(result.recordCount).toBe(1);
                expect(result.result.hits.hits[0]._source.ticker).toBe('MSFT');
            }
        });

        it('should prefer auto-detected fields over manual keyField', () => {
            const file1 = createLoadedFile(
                'f1',
                'file1.csv',
                [
                    { isin: 'US0001', ticker: 'AAPL', price: 100 },
                    { isin: 'US0002', ticker: 'GOOG', price: 200 },
                ],
                'ticker'
            );
            const file2 = createLoadedFile(
                'f2',
                'file2.csv',
                [{ isin: 'US0001', ticker: 'AAPL', price: 105 }],
                'ticker'
            );

            // Auto-detect should find 'isin' and use it instead of 'ticker'
            const result = diffFiles(file1, file2);
            if (typeof result !== 'string') {
                expect(result.recordCount).toBe(1);
                expect(result.result.hits.hits[0]._source.isin).toBe('US0002');
            }
        });
    });

    describe('edge cases', () => {
        it('should return error when file1 has no key fields', () => {
            const file1 = createLoadedFile('f1', 'file1.csv', [
                { price: 100, volume: 1000 },
            ]);
            const file2 = createLoadedFile('f2', 'file2.csv', [
                { isin: 'US0001', price: 100 },
            ]);

            const result = diffFiles(file1, file2);
            expect(result).toBe('Both files need keyfield set. Use /keyfield <id>');
        });

        it('should return error when file2 has no key fields', () => {
            const file1 = createLoadedFile('f1', 'file1.csv', [
                { isin: 'US0001', price: 100 },
            ]);
            const file2 = createLoadedFile('f2', 'file2.csv', [
                { price: 100, volume: 1000 },
            ]);

            const result = diffFiles(file1, file2);
            expect(result).toBe('Both files need keyfield set. Use /keyfield <id>');
        });

        it('should return error when both files have no key fields', () => {
            const file1 = createLoadedFile('f1', 'file1.csv', [
                { price: 100 },
            ]);
            const file2 = createLoadedFile('f2', 'file2.csv', [
                { price: 200 },
            ]);

            const result = diffFiles(file1, file2);
            expect(result).toBe('Both files need keyfield set. Use /keyfield <id>');
        });

        it('should handle empty file1', () => {
            const file1 = createLoadedFile('f1', 'file1.csv', []);
            const file2 = createLoadedFile('f2', 'file2.csv', [
                { isin: 'US0001', price: 100 },
            ]);

            const result = diffFiles(file1, file2);
            expect(result).toBe('Both files need keyfield set. Use /keyfield <id>');
        });

        it('should handle empty file2', () => {
            const file1 = createLoadedFile('f1', 'file1.csv', [
                { isin: 'US0001', price: 100 },
            ]);
            const file2 = createLoadedFile('f2', 'file2.csv', []);

            const result = diffFiles(file1, file2);
            expect(result).toBe('Both files need keyfield set. Use /keyfield <id>');
        });

        it('should handle null/undefined key values', () => {
            const file1 = createLoadedFile('f1', 'file1.csv', [
                { isin: null, price: 100 },
                { isin: undefined, price: 150 },
                { price: 200 },
                { isin: 'US0001', price: 250 },
            ]);
            const file2 = createLoadedFile('f2', 'file2.csv', [
                { isin: 'US0001', price: 105 },
            ]);

            const result = diffFiles(file1, file2);
            if (typeof result !== 'string') {
                // null, undefined, and missing should all produce empty string key
                // US0001 is in file2, so only 3 records should be in diff
                expect(result.recordCount).toBe(3);
            }
        });

        it('should use custom resultId when provided', () => {
            const file1 = createLoadedFile('f1', 'file1.csv', [
                { isin: 'US0001', price: 100 },
            ]);
            const file2 = createLoadedFile('f2', 'file2.csv', [
                { isin: 'US0002', price: 200 },
            ]);

            const result = diffFiles(file1, file2, 'my_custom_diff');
            if (typeof result !== 'string') {
                expect(result.newId).toBe('my_custom_diff');
            }
        });

        it('should handle exchange_code as a key field', () => {
            const file1 = createLoadedFile('f1', 'file1.csv', [
                { exchange_code: 'NYSE', isin: 'US0001', price: 100 },
                { exchange_code: 'NASDAQ', isin: 'US0002', price: 200 },
            ]);
            const file2 = createLoadedFile('f2', 'file2.csv', [
                { exchange_code: 'NYSE', isin: 'US0001', price: 105 },
            ]);

            const result = diffFiles(file1, file2);
            if (typeof result !== 'string') {
                expect(result.recordCount).toBe(1);
            }
        });
    });

    describe('result structure', () => {
        it('should return valid ElasticSearchResult structure', () => {
            const file1 = createLoadedFile('f1', 'file1.csv', [
                { isin: 'US0001', price: 100 },
                { isin: 'US0002', price: 200 },
            ]);
            const file2 = createLoadedFile('f2', 'file2.csv', [
                { isin: 'US0001', price: 105 },
            ]);

            const result = diffFiles(file1, file2);
            if (typeof result !== 'string') {
                expect(result.result.took).toBe(0);
                expect(result.result.timed_out).toBe(false);
                expect(result.result.hits.total.value).toBe(1);
                expect(result.result.hits.total.relation).toBe('eq');
                expect(result.result.hits.max_score).toBe(1.0);
                expect(result.result.hits.hits).toHaveLength(1);
                expect(result.result.hits.hits[0]._source.isin).toBe('US0002');
            }
        });
    });
});
