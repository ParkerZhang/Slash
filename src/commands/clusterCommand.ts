import { Workspace, CommandResult } from '../core/types.js';
import { SlashCommand } from '../commands/commandRegistry.js';
import { kMeans, findDominantTermsForCluster, buildMultiDimensionalFeatures, evaluateClusters, formatEnhancedClusterSummary, formatMentalModelVisualization } from '../core/clustering.js';
import { getLocalEmbeddings } from '../core/localEmbeddings.js';

type ClusteringMode = 'spreading' | 'dominance';

// Mathematical scoring functions
function calcSpreadingScore(values: string[]): number {
    // Shannon Entropy: H = -Σ p(v) × log(p(v))
    // Higher entropy = more evenly distributed = better spreading
    const n = values.length;
    const freq: Record<string, number> = {};
    values.forEach(v => { freq[v] = (freq[v] || 0) + 1; });
    
    let entropy = 0;
    const uniqueValues = Object.values(freq);
    for (const count of uniqueValues) {
        const p = count / n;
        if (p > 0) entropy -= p * Math.log2(p);
    }
    
    // Normalize to 0-1 range (max entropy = log2(uniqueCount))
    const maxEntropy = Math.log2(uniqueValues.length);
    return maxEntropy > 0 ? entropy / maxEntropy : 0;
}

function calcDominanceScore(values: string[]): number {
    // Gini-Simpson Index: 1 - Σ p(v)²
    // Lower index = more concentrated = better dominance
    const n = values.length;
    const freq: Record<string, number> = {};
    values.forEach(v => { freq[v] = (freq[v] || 0) + 1; });
    
    let sumSquares = 0;
    for (const count of Object.values(freq)) {
        const p = count / n;
        sumSquares += p * p;
    }
    
    // Return dominance (higher = more concentrated)
    return sumSquares;
}

export const ClusterCommand: SlashCommand = {
    name: '/cluster',
    description: 'Cluster a loaded file (Usage: /cluster <fileId> [k] [--Spreading|--Dominance] [--key <field>])',
    execute: async (args, workspace) => {
        const parts = args.trim().split(/\s+/).filter(Boolean);
        if (parts.length < 1) {
            return { output: 'Usage: /cluster <fileId> [k] [--Spreading|--Dominance] [--key <field>]\n\nOptions:\n  --Spreading   Find evenly distributed fields (high entropy)\n  --Dominance   Find concentrated fields (low entropy) [DEFAULT]\n  --key <field> Force clustering on specific field\n\nExamples:\n  /cluster f1 4\n  /cluster f1 4 --Spreading\n  /cluster f1 3 --Dominance\n  /cluster f1 4 --key country\n  /cluster missing --key isin' };
        }

        const fileId = parts[0];
        const loadedFiles = workspace.getLoadedFiles();
        const file = loadedFiles.get(fileId);

        if (!file) return { output: `File '${fileId}' not loaded. Use /load or /loadCsv first.` };

        // Parse options
        let k = 4;
        let mode: ClusteringMode = 'dominance';
        let forceKey: string | null = null;
        
        for (let i = 1; i < parts.length; i++) {
            const part = parts[i];
            if (part === '--key' || part === '-k') {
                i++;
                forceKey = parts[i] || null;
            } else if (part.startsWith('--key=')) {
                forceKey = part.split('=')[1];
            } else if (part.startsWith('--')) {
                const option = part.slice(2).toLowerCase();
                if (option === 'spreading') mode = 'spreading';
                else if (option === 'dominance') mode = 'dominance';
            } else {
                const num = parseInt(part, 10);
                if (!isNaN(num)) k = num;
            }
        }

        const records = file.data.hits.hits.map((hit) => hit._source);

        if (records.length < k) {
            return { output: `Not enough records for ${k} clusters. File has ${records.length} records.` };
        }

        try {
            // If it's a multi-dimensional request (default without --key), use the advanced logic
            if (!forceKey) {
                workspace.setCommandOutput?.([
                    `Running multi-dimensional /cluster for '${fileId}'...`,
                    `Loading embedding model...`,
                ]);

                const { embeddings: nameEmbeddings } = await getLocalEmbeddings(
                    records,
                    (current, total) => {
                        workspace.setCommandOutput?.([
                            `Running multi-dimensional /cluster for '${fileId}'...`,
                            `Generating embeddings: ${current}/${total} records`,
                        ]);
                    }
                );

                const finalFeatures = buildMultiDimensionalFeatures(records, nameEmbeddings);
                const clusteringResult = kMeans(finalFeatures, k);
                const evaluation = evaluateClusters(clusteringResult.clusters, records);

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

                return {
                    output: [
                        formatEnhancedClusterSummary(clusteringResult.clusters, records, evaluation),
                        '',
                        formatMentalModelVisualization(clusteringResult.clusters, records),
                        '',
                        `Saved as: ${clusteringFileId}`,
                    ].join('\n')
                };
            }

            // Fallback to single-field clustering if --key is specified
            const fields = Object.keys(records[0]).filter(f => f !== '_cluster');
            if (!fields.includes(forceKey)) {
                return { output: `Field '${forceKey}' not found. Available fields: ${fields.join(', ')}` };
            }

            workspace.setCommandOutput?.([
                `Running /cluster for '${fileId}' on field '${forceKey}'...`,
                `Loading embedding model...`,
            ]);

            const { embeddings } = await getLocalEmbeddings(
                records.map(r => ({ [forceKey!]: String(r[forceKey!] || '') })),
                (current, total) => {
                    workspace.setCommandOutput?.([
                        `Running /cluster for '${fileId}' on field '${forceKey}'...`,
                        `Generating embeddings: ${current}/${total} records`,
                    ]);
                }
            );

            const clusteringResult = kMeans(embeddings, k);
            const evaluation = evaluateClusters(clusteringResult.clusters, records);

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

            return {
                output: [
                    formatEnhancedClusterSummary(clusteringResult.clusters, records, evaluation),
                    '',
                    formatMentalModelVisualization(clusteringResult.clusters, records),
                    '',
                    `Saved as: ${clusteringFileId}`,
                ].join('\n')
            };
        } catch (error) {
            return { output: `Cluster Error: ${error instanceof Error ? error.message : String(error)}` };
        }
    },
    suggestArgs: (_input, workspace) => {
        const files = Array.from(workspace.getLoadedFiles().keys());
        return [...files, '--Dominance', '--Spreading', '--key', '-k'];
    },
};
