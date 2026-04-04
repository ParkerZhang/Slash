import { Workspace, CommandResult } from '../core/types.js';
import { SlashCommand } from '../commands/commandRegistry.js';
import { kMeans, findDominantTermsForCluster } from '../core/clustering.js';
import { getLocalEmbeddings } from '../core/localEmbeddings.js';

type ClusteringMode = 'spreading' | 'dominance';

// Sector symbols for mental model visualization
const SECTOR_SYMBOLS: Record<string, string> = {
    'Tech': '▲',
    'Banking': '■',
    'Energy': '◆',
    'Automotive': '●',
    'Telecom': '◉',
    'Retail': '◈',
    'Other': '•',
};

function inferSector(name: string): string {
    const n = name.toLowerCase();
    if (n.includes('bank') || n.includes('financial') || n.includes('ufj') || n.includes('itau') || n.includes('galicia') || n.includes('merchants') || n.includes('insurance') || n.includes('ping an')) return 'Banking';
    if (n.includes('apple') || n.includes('microsoft') || n.includes('google') || n.includes('alphabet') || n.includes('sony') || n.includes('panasonic') || n.includes('tencent') || n.includes('alibaba') || n.includes('xiaomi') || n.includes('meituan') || n.includes('jd.com') || n.includes('amazon') || n.includes('meta') || n.includes('lenovo')) return 'Tech';
    if (n.includes('petro') || n.includes('vale') || n.includes('shell') || n.includes('enel') || n.includes('ypf') || n.includes('suzano')) return 'Energy';
    if (n.includes('motor') || n.includes('auto') || n.includes('toyota') || n.includes('geely')) return 'Automotive';
    if (n.includes('vodafone') || n.includes('kddi') || n.includes('telecom') || n.includes('america movil')) return 'Telecom';
    if (n.includes('seven & i') || n.includes('bimbo') || n.includes('cemex') || n.includes('walmart') || n.includes('falabella') || n.includes('localiza')) return 'Retail';
    return 'Other';
}

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
            // Find the best field based on mode, or use forced key
            const fields = Object.keys(records[0]).filter(f => f !== '_cluster');
            let bestField = '';
            let bestScore = mode === 'spreading' ? -1 : 0;
            let autoDetected = false;

            if (forceKey) {
                // Validate forced key
                if (!fields.includes(forceKey)) {
                    return { output: `Field '${forceKey}' not found. Available fields: ${fields.join(', ')}` };
                }
                bestField = forceKey;
            } else {
                // Auto-detect best field
                for (const field of fields) {
                    const values = records.map(r => String(r[field] || '')).filter(v => v);
                    if (values.length === 0) continue;
                    
                    const uniqueValues = new Set(values).size;
                    
                    // Skip fields with too few or too many unique values
                    if (uniqueValues < 2 || uniqueValues > values.length / 2) continue;
                    
                    let score: number;
                    if (mode === 'spreading') {
                        score = calcSpreadingScore(values);
                    } else {
                        score = calcDominanceScore(values);
                    }
                    
                    // Bonus for known meaningful fields
                    if (['country', 'exchange', 'xchg', 'currency', 'sector', 'name'].includes(field.toLowerCase())) {
                        score *= 1.2;
                    }
                    
                    if (mode === 'spreading' ? score > bestScore : score > bestScore) {
                        bestScore = score;
                        bestField = field;
                        autoDetected = true;
                    }
                }
            }

            if (!bestField) {
                return { output: `No suitable clustering field found. Try a file with more variation.` };
            }

            workspace.setCommandOutput?.([
                `Running /cluster for '${fileId}'...`,
                `Mode: ${mode === 'spreading' ? 'Spreading (high entropy)' : 'Dominance (concentration)'}`,
                `Field: ${forceKey ? `'${forceKey}' (forced)` : `${bestField} (auto-detected, score: ${bestScore.toFixed(3)})`}`,
                `Loading embedding model...`,
            ]);

            // Create embeddings from the single best field
            const { embeddings } = await getLocalEmbeddings(
                records.map((r, i) => ({ [bestField]: String(r[bestField] || '') })),
                (current, total) => {
                    workspace.setCommandOutput?.([
                        `Running /cluster for '${fileId}'...`,
                        `Generating embeddings: ${current}/${total} records`,
                    ]);
                }
            );

            workspace.setCommandOutput?.([
                `Running /cluster for '${fileId}'...`,
                `Embeddings generated. Running K-means clustering (${k} clusters)...`,
            ]);

            // Run KMeans
            const clusteringResult = kMeans(embeddings, k);

            workspace.setCommandOutput?.([
                `Running /cluster for '${fileId}'...`,
                `Clustering complete. Formatting output...`,
            ]);

            for (const cluster of clusteringResult.clusters) {
                cluster.dominantTerms = findDominantTermsForCluster(cluster.members, records, 10);
            }

            // Save clustering result with _cluster field
            const clusteringFileId = `${fileId}-clustering-${k}`;
            const clusteringData = {
                took: 0,
                timed_out: false,
                hits: {
                    total: { value: records.length, relation: 'eq' },
                    max_score: 1.0,
                    hits: records.map((record: Record<string, unknown>, idx: number) => ({
                        _index: 'clustering',
                        _id: String(idx + 1),
                        _score: 1.0,
                        _source: {
                            ...record,
                            _cluster: clusteringResult.assignments[idx],
                        },
                    })),
                },
            };

            workspace.addFile(clusteringFileId, `${file.name} (${k} clusters)`, clusteringData as any);

            // Build mental model visualization with indentation
            const lines: string[] = [];
            lines.push(`📊 Clustering Results (${mode})`);
            lines.push(`Field: ${bestField} | Records: ${records.length} | Clusters: ${k}`);
            lines.push('─'.repeat(50));
            
            const clusterData = clusteringResult.clusters.map((cluster: any) => {
                const members = cluster.members;
                const byValue: Record<string, number> = {};
                const bySector: Record<string, number> = {};
                
                members.forEach((idx: number) => {
                    const rec = records[idx];
                    if (rec) {
                        const value = String(rec[bestField] || 'N/A');
                        const sector = rec.sector || rec.Sector || inferSector(rec.name || rec.Name || '');
                        
                        byValue[value] = (byValue[value] || 0) + 1;
                        bySector[sector] = (bySector[sector] || 0) + 1;
                    }
                });
                
                const topSector = Object.entries(bySector).sort((a, b) => b[1] - a[1])[0];
                return { byValue, bySector, topSector: topSector?.[0] || 'Other' };
            });

            // Format with indentation:
            // Sector Symbol
            //   Value ●●●●
            //   Value ●●
            for (const cluster of clusterData) {
                const symbol = SECTOR_SYMBOLS[cluster.topSector] || '•';
                lines.push(`${cluster.topSector} ${symbol}`);
                
                const sortedValues = Object.entries(cluster.byValue).sort((a, b) => b[1] - a[1]);
                for (const [value, count] of sortedValues) {
                    const dots = '●'.repeat(Math.min(count, 10));
                    lines.push(`  ${value} ${dots}`);
                }
                lines.push('');
            }

            lines.push('─'.repeat(50));
            lines.push(`Saved as: ${clusteringFileId}`);
            lines.push('Tip: /files to list, /preview to browse');
            
            return {
                output: lines.join('\n'),
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
