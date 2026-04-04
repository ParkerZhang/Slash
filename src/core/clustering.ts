export interface ClusterResult {
    centroid: number[];
    members: number[];
    dominantTerms: string[];
    // Multi-dimensional analysis
    geoProfile?: Record<string, number>;
    nameProfile?: Record<string, number>;
    purity?: number;
}

export interface KMeansOutput {
    clusters: ClusterResult[];
    assignments: number[];
}

export interface ClusteringOptions {
    k: number;
    nameWeight?: number; // 0.0 to 1.0, default 0.7
    geoWeight?: number; // 0.0 to 1.0, default 0.3
    maxIterations?: number;
    tolerance?: number;
    twoStage?: boolean; // First cluster by geo, then by name
    useCosine?: boolean;
}

export interface ClusterEvaluation {
    totalRecords: number;
    numClusters: number;
    clusterSizes: number[];
    geoPurity: number; // How well clusters separate by geography
    namePurity: number; // How well clusters separate by name/sector
    balanceScore: number; // How balanced the clusters are (0-1)
    interpretation: string[];
}

export interface MultiDimensionalProfile {
    geoClusters: Record<string, number>; // geography -> count
    sectorClusters: Record<string, number>; // sector -> count
    exchangeCoverage: Record<string, number>; // exchange -> count
    crossRegionPresence: Record<number, Set<string>>; // cluster -> set of countries
}


function euclideanDistance(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        const diff = a[i] - b[i];
        sum += diff * diff;
    }
    return Math.sqrt(sum);
}

function cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator === 0 ? 0 : dotProduct / denominator;
}

function cosineDistance(a: number[], b: number[]): number {
    return 1 - cosineSimilarity(a, b);
}

function initializeCentroids(
    vectors: number[][],
    k: number,
    rng: () => number,
): number[][] {
    const centroids: number[][] = [vectors[Math.floor(rng() * vectors.length)].slice()];

    for (let c = 1; c < k; c++) {
        const distances = vectors.map((v) => {
            let minDist = Infinity;
            for (const centroid of centroids) {
                const d = euclideanDistance(v, centroid);
                if (d < minDist) minDist = d;
            }
            return minDist * minDist;
        });

        const totalDist = distances.reduce((sum, d) => sum + d, 0);
        let threshold = rng() * totalDist;
        let chosen = 0;

        for (let i = 0; i < distances.length; i++) {
            threshold -= distances[i];
            if (threshold <= 0) {
                chosen = i;
                break;
            }
        }

        centroids.push(vectors[chosen].slice());
    }

    return centroids;
}

export function kMeans(
    vectors: number[][],
    k: number,
    maxIterations: number = 100,
    tolerance: number = 1e-4,
    rng: () => number = Math.random,
): KMeansOutput {
    if (vectors.length === 0 || k <= 0) {
        return { clusters: [], assignments: [] };
    }

    const n = vectors.length;
    const effectiveK = Math.min(k, n);
    let centroids = initializeCentroids(vectors, effectiveK, rng);
    const assignments = new Array<number>(n).fill(0);

    for (let iter = 0; iter < maxIterations; iter++) {
        // Assignment step
        for (let i = 0; i < n; i++) {
            let bestCluster = 0;
            let bestDist = Infinity;
            for (let c = 0; c < centroids.length; c++) {
                const d = euclideanDistance(vectors[i], centroids[c]);
                if (d < bestDist) {
                    bestDist = d;
                    bestCluster = c;
                }
            }
            assignments[i] = bestCluster;
        }

        // Update step
        const newCentroids = centroids.map(() => new Array<number>(vectors[0].length).fill(0));
        const counts = new Array<number>(effectiveK).fill(0);

        for (let i = 0; i < n; i++) {
            const cluster = assignments[i];
            counts[cluster]++;
            for (let d = 0; d < vectors[i].length; d++) {
                newCentroids[cluster][d] += vectors[i][d];
            }
        }

        let maxShift = 0;
        for (let c = 0; c < effectiveK; c++) {
            if (counts[c] === 0) continue;
            for (let d = 0; d < newCentroids[c].length; d++) {
                newCentroids[c][d] /= counts[c];
            }
            const shift = euclideanDistance(centroids[c], newCentroids[c]);
            if (shift > maxShift) maxShift = shift;
        }

        centroids = newCentroids;

        if (maxShift < tolerance) {
            break;
        }
    }

    const clusters: ClusterResult[] = centroids.map((centroid, c) => ({
        centroid,
        members: assignments.reduce<number[]>((acc, cluster, idx) => {
            if (cluster === c) acc.push(idx);
            return acc;
        }, []),
        dominantTerms: [],
    }));

    return { clusters, assignments };
}

export function kMeansCosine(
    vectors: number[][],
    k: number,
    maxIterations: number = 100,
    tolerance: number = 1e-4,
    rng: () => number = Math.random,
): KMeansOutput {
    if (vectors.length === 0 || k <= 0) {
        return { clusters: [], assignments: [] };
    }

    const n = vectors.length;
    const effectiveK = Math.min(k, n);
    let centroids = initializeCentroids(vectors, effectiveK, rng);
    const assignments = new Array<number>(n).fill(0);

    for (let iter = 0; iter < maxIterations; iter++) {
        // Assignment step - use cosine distance
        for (let i = 0; i < n; i++) {
            let bestCluster = 0;
            let bestDist = Infinity;
            for (let c = 0; c < centroids.length; c++) {
                const d = cosineDistance(vectors[i], centroids[c]);
                if (d < bestDist) {
                    bestDist = d;
                    bestCluster = c;
                }
            }
            assignments[i] = bestCluster;
        }

        // Update step
        const newCentroids = centroids.map(() => new Array<number>(vectors[0].length).fill(0));
        const counts = new Array<number>(effectiveK).fill(0);

        for (let i = 0; i < n; i++) {
            const cluster = assignments[i];
            counts[cluster]++;
            for (let d = 0; d < vectors[i].length; d++) {
                newCentroids[cluster][d] += vectors[i][d];
            }
        }

        let maxShift = 0;
        for (let c = 0; c < effectiveK; c++) {
            if (counts[c] === 0) continue;
            for (let d = 0; d < newCentroids[c].length; d++) {
                newCentroids[c][d] /= counts[c];
            }
            const shift = euclideanDistance(centroids[c], newCentroids[c]);
            if (shift > maxShift) maxShift = shift;
        }

        centroids = newCentroids;

        if (maxShift < tolerance) {
            break;
        }
    }

    const clusters: ClusterResult[] = centroids.map((centroid, c) => ({
        centroid,
        members: assignments.reduce<number[]>((acc, cluster, idx) => {
            if (cluster === c) acc.push(idx);
            return acc;
        }, []),
        dominantTerms: [],
    }));

    return { clusters, assignments };
}

export function findDominantTermsForCluster(
    clusterIndices: number[],
    records: Record<string, unknown>[],
    topN: number = 10,
): string[] {
    const termFreq: Record<string, number> = {};
    const fieldFreq: Record<string, number> = {};

    for (const idx of clusterIndices) {
        const record = records[idx];
        if (!record) continue;

        for (const [key, value] of Object.entries(record)) {
            if (value === null || value === undefined) continue;

            const strValue = String(value).toLowerCase();
            if (strValue.length < 2) continue;

            const fieldKey = `${key}:${strValue}`;
            termFreq[fieldKey] = (termFreq[fieldKey] || 0) + 1;
            fieldFreq[key] = (fieldFreq[key] || 0) + 1;
        }
    }

    // Score terms by TF weighted by field specificity (rare fields are more informative)
    const totalRecords = clusterIndices.length;
    const scoredTerms = Object.entries(termFreq).map(([term, count]) => {
        const field = term.split(':')[0];
        const fieldCoverage = (fieldFreq[field] || 0) / totalRecords;
        // Penalize very common fields, boost specific values
        const specificity = 1 - fieldCoverage;
        const tf = count / totalRecords;
        return { term, score: tf * (1 + specificity) };
    });

    scoredTerms.sort((a, b) => b.score - a.score);
    return scoredTerms.slice(0, topN).map((t) => t.term);
}

export function formatClusterSummary(
    clusters: ClusterResult[],
    records: Record<string, unknown>[],
    topTerms: number = 10,
): string {
    const lines: string[] = [];
    lines.push(`Found ${clusters.length} clusters:\n`);

    for (let i = 0; i < clusters.length; i++) {
        const cluster = clusters[i];
        const dominantTerms = findDominantTermsForCluster(cluster.members, records, topTerms);

        lines.push(`Cluster ${i + 1} (${cluster.members.length} records):`);
        lines.push(`  Dominant characteristics:`);

        // Group by field for readability
        const byField: Record<string, string[]> = {};
        for (const term of dominantTerms) {
            const [field, value] = term.split(':');
            if (!byField[field]) byField[field] = [];
            byField[field].push(value);
        }

        for (const [field, values] of Object.entries(byField)) {
            lines.push(`    ${field}: ${values.slice(0, 4).join(', ')}${values.length > 4 ? '...' : ''}`);
        }
        lines.push('');
    }

    return lines.join('\n');
}

// ============================================================================
// Multi-Dimensional Clustering (V2.1)
// ============================================================================

/**
 * Normalize a vector to unit length
 */
export function normalizeVector(vector: number[]): number[] {
    const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
    if (norm === 0) return vector;
    return vector.map(v => v / norm);
}

/**
 * Build multi-dimensional feature vectors from records
 * Combines name embeddings with structured categorical encoding
 */
export function buildMultiDimensionalFeatures(
    records: Record<string, unknown>[],
    nameEmbeddings: number[][],
    nameWeight: number = 0.7,
    geoWeight: number = 0.3,
): number[][] {
    const geoFeatures = buildGeoFeatures(records);
    const features: number[][] = [];

    for (let i = 0; i < records.length; i++) {
        const nameVec = nameEmbeddings[i] || [];
        const geoVec = geoFeatures[i] || [];

        // Apply weights
        const weightedName = nameVec.map(v => v * nameWeight);
        const weightedGeo = geoVec.map(v => v * geoWeight);

        // Concatenate
        const combined = [...weightedName, ...weightedGeo];

        // Normalize to prevent one feature group from dominating
        const normalized = normalizeVector(combined);

        features.push(normalized);
    }

    return features;
}

/**
 * Build structured categorical features from geographic/categorical fields
 */
export function buildGeoFeatures(records: Record<string, unknown>[]): number[][] {
    // Extract categorical fields: country, exchange, currency
    const countries = new Set<string>();
    const exchanges = new Set<string>();
    const currencies = new Set<string>();

    for (const record of records) {
        if (record.country) countries.add(String(record.country).toLowerCase());
        if (record.xchg || record.exchange) exchanges.add(String(record.xchg || record.exchange).toLowerCase());
        if (record.currency) currencies.add(String(record.currency).toLowerCase());
    }

    const countryList = Array.from(countries).sort();
    const exchangeList = Array.from(exchanges).sort();
    const currencyList = Array.from(currencies).sort();

    const features: number[][] = [];

    for (const record of records) {
        const feature: number[] = [];

        // One-hot encode country
        const country = record.country ? String(record.country).toLowerCase() : '';
        for (const c of countryList) {
            feature.push(c === country ? 1 : 0);
        }

        // One-hot encode exchange
        const exchange = record.xchg || record.exchange ? String(record.xchg || record.exchange).toLowerCase() : '';
        for (const e of exchangeList) {
            feature.push(e === exchange ? 1 : 0);
        }

        // One-hot encode currency
        const currency = record.currency ? String(record.currency).toLowerCase() : '';
        for (const c of currencyList) {
            feature.push(c === currency ? 1 : 0);
        }

        features.push(feature);
    }

    return features;
}

/**
 * Two-stage clustering: first by geography, then by name within geo clusters
 */
export function twoStageClustering(
    records: Record<string, unknown>[],
    nameEmbeddings: number[][],
    geoK: number,
    nameK: number,
): KMeansOutput {
    // Stage 1: Cluster by geography
    const geoFeatures = buildGeoFeatures(records);
    const geoResult = kMeans(geoFeatures, geoK);

    // Stage 2: Within each geo cluster, sub-cluster by name
    const finalAssignments = new Array<number>(records.length).fill(0);
    const allClusters: ClusterResult[] = [];
    let clusterOffset = 0;

    for (let g = 0; g < geoK; g++) {
        const geoMembers = geoResult.assignments.reduce<number[]>((acc, cluster, idx) => {
            if (cluster === g) acc.push(idx);
            return acc;
        }, []);

        if (geoMembers.length === 0) continue;

        // Extract name embeddings for this geo cluster
        const geoNameEmbeddings = geoMembers.map(idx => nameEmbeddings[idx]);
        const effectiveK = Math.min(nameK, geoMembers.length);

        if (effectiveK <= 1 || geoMembers.length <= 2) {
            // Not enough diversity, assign all to one cluster
            for (const idx of geoMembers) {
                finalAssignments[idx] = clusterOffset;
            }
            allClusters.push({
                centroid: geoNameEmbeddings[0] || [],
                members: geoMembers,
                dominantTerms: [],
            });
            clusterOffset++;
        } else {
            const subResult = kMeans(geoNameEmbeddings, effectiveK);
            for (let i = 0; i < geoMembers.length; i++) {
                finalAssignments[geoMembers[i]] = clusterOffset + subResult.assignments[i];
            }
            allClusters.push(...subResult.clusters.map(c => ({
                ...c,
                members: geoMembers.filter((_, idx) => subResult.assignments[idx] === allClusters.length - clusterOffset),
            })));
            clusterOffset += effectiveK;
        }
    }

    return {
        clusters: allClusters,
        assignments: finalAssignments,
    };
}

/**
 * Evaluate cluster quality across multiple dimensions
 */
export function evaluateClusters(
    clusters: ClusterResult[],
    records: Record<string, unknown>[],
): ClusterEvaluation {
    const clusterSizes = clusters.map(c => c.members.length);
    const totalRecords = records.length;

    // Calculate geographic purity
    const geoCounts: Record<number, Record<string, number>> = {};
    const nameCounts: Record<number, Record<string, number>> = {};

    for (let c = 0; c < clusters.length; c++) {
        geoCounts[c] = {};
        nameCounts[c] = {};

        for (const idx of clusters[c].members) {
            const record = records[idx];
            const country = record.country ? String(record.country).toLowerCase() : 'unknown';
            const name = record.name ? String(record.name).toLowerCase() : 'unknown';

            geoCounts[c][country] = (geoCounts[c][country] || 0) + 1;
            nameCounts[c][name] = (nameCounts[c][name] || 0) + 1;
        }
    }

    // Geo purity: for each cluster, what % is from the dominant country
    let geoPuritySum = 0;
    let namePuritySum = 0;

    for (let c = 0; c < clusters.length; c++) {
        const geoValues = Object.values(geoCounts[c]);
        const nameValues = Object.values(nameCounts[c]);

        const geoMax = Math.max(...geoValues);
        const nameMax = Math.max(...nameValues);

        geoPuritySum += geoMax / (clusters[c].members.length || 1);
        namePuritySum += nameMax / (clusters[c].members.length || 1);
    }

    const geoPurity = geoPuritySum / (clusters.length || 1);
    const namePurity = namePuritySum / (clusters.length || 1);

    // Balance score: how equal the cluster sizes are (1 = perfectly balanced)
    const meanSize = totalRecords / (clusters.length || 1);
    const variance = clusterSizes.reduce((sum, size) => sum + Math.pow(size - meanSize, 2), 0) / (clusters.length || 1);
    const stdDev = Math.sqrt(variance);
    const balanceScore = Math.max(0, 1 - (stdDev / (meanSize || 1)));

    // Build interpretation
    const interpretation: string[] = [];

    // Check cross-region presence
    const crossRegionCount: Record<number, Set<string>> = {};
    for (let c = 0; c < clusters.length; c++) {
        crossRegionCount[c] = new Set<string>();
        for (const idx of clusters[c].members) {
            const country = records[idx].country ? String(records[idx].country).toLowerCase() : 'unknown';
            crossRegionCount[c].add(country);
        }
    }

    const multiRegionClusters = Object.values(crossRegionCount).filter(countries => countries.size > 1).length;
    if (multiRegionClusters > 0) {
        interpretation.push(`${multiRegionClusters} cluster(s) span multiple regions (good cross-region grouping)`);
    }

    if (geoPurity < 0.5) {
        interpretation.push('Clusters are not dominated by single geography (good name-based clustering)');
    }

    if (balanceScore > 0.7) {
        interpretation.push('Clusters are well-balanced in size');
    } else if (balanceScore < 0.4) {
        interpretation.push('Clusters are imbalanced - consider adjusting weights');
    }

    return {
        totalRecords,
        numClusters: clusters.length,
        clusterSizes,
        geoPurity,
        namePurity,
        balanceScore,
        interpretation,
    };
}

/**
 * Build multi-dimensional profile for each cluster
 */
export function buildClusterProfile(
    cluster: ClusterResult,
    records: Record<string, unknown>[],
): MultiDimensionalProfile {
    const geoClusters: Record<string, number> = {};
    const sectorClusters: Record<string, number> = {};
    const exchangeCoverage: Record<string, number> = {};
    const crossRegionPresence: Record<number, Set<string>> = { 0: new Set<string>() };

    for (const idx of cluster.members) {
        const record = records[idx];

        if (record.country) {
            const country = String(record.country).toLowerCase();
            geoClusters[country] = (geoClusters[country] || 0) + 1;
            crossRegionPresence[0].add(country);
        }

        if (record.name) {
            // Simple sector inference from name
            const name = String(record.name).toLowerCase();
            let sector = 'Other';
            if (name.includes('bank') || name.includes('financial') || name.includes('ufj') || 
                name.includes('itau') || name.includes('galicia') || name.includes('merchants') ||
                name.includes('industrial bank')) sector = 'Banking';
            else if (name.includes('apple') || name.includes('microsoft') || name.includes('google') || 
                     name.includes('alphabet') || name.includes('sony') || name.includes('panasonic') ||
                     name.includes('tencent') || name.includes('alibaba') || name.includes('health info')) sector = 'Tech';
            else if (name.includes('petro') || name.includes('vale') || name.includes('shell') || 
                     name.includes('enel')) sector = 'Energy';
            else if (name.includes('motor') || name.includes('auto') || name.includes('toyota') || 
                     name.includes('geely') || name.includes('movil')) sector = 'Automotive';
            else if (name.includes('vodafone') || name.includes('kddi') || name.includes('telecom') ||
                     name.includes('america movil')) sector = 'Telecom';
            else if (name.includes('seven & i') || name.includes('walmart')) sector = 'Retail';

            sectorClusters[sector] = (sectorClusters[sector] || 0) + 1;
        }

        const exchange = record.xchg || record.exchange;
        if (exchange) {
            const xchg = String(exchange).toLowerCase();
            exchangeCoverage[xchg] = (exchangeCoverage[xchg] || 0) + 1;
        }
    }

    return {
        geoClusters,
        sectorClusters,
        exchangeCoverage,
        crossRegionPresence,
    };
}

/**
 * Format multi-dimensional cluster visualization in mental model style
 * 
 * Example output:
 * Tech ▲
 * LATAM ● ●    JP ●    US
 * 
 * Banks ■
 * LATAM ●      JP ● ●  US ●
 */
export function formatMentalModelVisualization(
    clusters: ClusterResult[],
    records: Record<string, unknown>[],
): string {
    const lines: string[] = [];
    lines.push('🧠 Mental Model Visualization\n');

    // Extract all countries and sectors
    const allCountries = new Set<string>();
    const sectorSymbols: Record<string, string> = {
        'Tech': '▲',
        'Banking': '■',
        'Energy': '◆',
        'Automotive': '●',
        'Telecom': '◉',
        'Retail': '◈',
        'Other': '•',
    };

    for (const record of records) {
        if (record.country) {
            allCountries.add(String(record.country).toUpperCase());
        }
    }

    const countryList = Array.from(allCountries).sort();

    // Analyze each cluster
    for (let c = 0; c < clusters.length; c++) {
        const cluster = clusters[c];
        const profile = buildClusterProfile(cluster, records);

        // Find dominant sector
        const dominantSector = Object.entries(profile.sectorClusters).sort((a, b) => b[1] - a[1])[0];
        const sectorName = dominantSector ? dominantSector[0] : 'Mixed';
        const symbol = sectorSymbols[sectorName] || '•';

        lines.push(`${sectorName} ${symbol}`);

        // Build country distribution row
        const countryRow: string[] = [];
        for (const country of countryList) {
            const count = profile.geoClusters[country.toLowerCase()] || 0;
            const dots = '●'.repeat(Math.min(count, 10)); // Cap at 10 dots
            if (count > 0) {
                countryRow.push(`${country} ${dots}`);
            }
        }

        lines.push(countryRow.join('    '));
        lines.push('');
    }

    return lines.join('\n');
}

/**
 * Enhanced cluster summary with multi-dimensional analysis
 */
export function formatEnhancedClusterSummary(
    clusters: ClusterResult[],
    records: Record<string, unknown>[],
    evaluation: ClusterEvaluation,
    topTerms: number = 10,
): string {
    const lines: string[] = [];
    lines.push(`📊 Multi-Dimensional Clustering Analysis\n`);
    lines.push(`Found ${clusters.length} clusters from ${evaluation.totalRecords} records\n`);

    // Evaluation metrics
    lines.push('📈 Cluster Quality Metrics:');
    lines.push(`  Geography Purity: ${(evaluation.geoPurity * 100).toFixed(1)}% (lower = better name-based clustering)`);
    lines.push(`  Name/Sector Purity: ${(evaluation.namePurity * 100).toFixed(1)}%`);
    lines.push(`  Balance Score: ${(evaluation.balanceScore * 100).toFixed(1)}%`);

    if (evaluation.interpretation.length > 0) {
        lines.push('\n💡 Insights:');
        for (const insight of evaluation.interpretation) {
            lines.push(`  ✓ ${insight}`);
        }
    }

    lines.push('\n' + '─'.repeat(60) + '\n');

    // Detailed cluster breakdown
    for (let i = 0; i < clusters.length; i++) {
        const cluster = clusters[i];
        const profile = buildClusterProfile(cluster, records);
        const dominantTerms = findDominantTermsForCluster(cluster.members, records, topTerms);

        lines.push(`Cluster ${i + 1} (${cluster.members.length} records):`);

        // Sector breakdown
        if (Object.keys(profile.sectorClusters).length > 0) {
            const topSectors = Object.entries(profile.sectorClusters).sort((a, b) => b[1] - a[1]).slice(0, 3);
            lines.push(`  Dominant sectors: ${topSectors.map(([name, count]) => `${name} (${count})`).join(', ')}`);
        }

        // Geographic breakdown
        if (Object.keys(profile.geoClusters).length > 0) {
            const topCountries = Object.entries(profile.geoClusters).sort((a, b) => b[1] - a[1]).slice(0, 3);
            lines.push(`  Geography: ${topCountries.map(([name, count]) => `${name.toUpperCase()} (${count})`).join(', ')}`);
        }

        // Exchange coverage
        if (Object.keys(profile.exchangeCoverage).length > 0) {
            const topExchanges = Object.entries(profile.exchangeCoverage).sort((a, b) => b[1] - a[1]).slice(0, 3);
            lines.push(`  Exchanges: ${topExchanges.map(([name, count]) => `${name.toUpperCase()} (${count})`).join(', ')}`);
        }

        // Cross-region presence
        const countryCount = cluster.members.reduce((count, idx) => {
            if (records[idx].country) count.add(String(records[idx].country).toLowerCase());
            return count;
        }, new Set<string>()).size;

        if (countryCount > 1) {
            lines.push(`  ✓ Cross-region cluster (${countryCount} countries)`);
        }

        lines.push('');
    }

    return lines.join('\n');
}

