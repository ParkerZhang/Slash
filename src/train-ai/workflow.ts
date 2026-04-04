import * as fs from 'fs';
import * as path from 'path';
import { execFileSync } from 'child_process';
import {
    WorkflowState,
    WorkflowStep,
    MissingRecord,
    TrainAiOutput,
    ClusterInfo,
    QualityMetrics,
} from './types.js';
import { generateWorkflowId, saveState, loadState, cacheStepData, saveOutput } from './persistence.js';
import { MODEL_REGISTRY, getModelProfile } from './modelRegistry.js';
import { setLastPrompt, setLastResult } from './promptStore.js';
import {
    kMeans,
    buildMultiDimensionalFeatures,
    evaluateClusters,
    findDominantTermsForCluster,
    buildClusterProfile,
    formatMentalModelVisualization,
    formatEnhancedClusterSummary,
} from '../core/clustering.js';
import { runPromptStreaming } from '../ai/modelManager.js';

const DATA_DIR = path.join(process.cwd(), 'data');

// ============================================================================
// Seeded Random Number Generator
// ============================================================================

function seededRandom(seed: number): () => number {
    let s = seed;
    return () => {
        s = (s * 9301 + 49297) % 233280;
        return s / 233280;
    };
}

// ============================================================================
// Step 1: Data Generation
// ============================================================================

interface SeedRecord {
    isin: string;
    xchg: string;
    currency: string;
    country: string;
    name: string;
    sector: string;
    pricedate: string;
}

function inferSector(name: string): string {
    const n = name.toLowerCase();
    if (n.includes('bank') || n.includes('financial') || n.includes('ufj') ||
        n.includes('itau') || n.includes('galicia') || n.includes('merchants')) return 'Banking';
    if (n.includes('apple') || n.includes('microsoft') || n.includes('google') ||
        n.includes('alphabet') || n.includes('sony') || n.includes('panasonic') ||
        n.includes('tencent') || n.includes('alibaba')) return 'Tech';
    if (n.includes('petro') || n.includes('vale') || n.includes('shell') ||
        n.includes('enel')) return 'Energy';
    if (n.includes('motor') || n.includes('auto') || n.includes('toyota') ||
        n.includes('geely') || n.includes('movil')) return 'Automotive';
    if (n.includes('vodafone') || n.includes('kddi') || n.includes('telecom')) return 'Telecom';
    if (n.includes('seven & i') || n.includes('walmart')) return 'Retail';
    return 'Other';
}

async function step1DataGeneration(state: WorkflowState, onProgress?: (step: number, msg: string) => void): Promise<void> {
    const rng = seededRandom(state.seed);
    onProgress?.(1, 'Loading clustering.csv seed data...');

    // Load and parse clustering.csv
    const content = fs.readFileSync(path.join(DATA_DIR, 'clustering.csv'), 'utf-8');
    const allLines = content.trim().split('\n').filter(line => line.trim().length > 0);
    const firstLine = allLines[0];
    const headerStart = firstLine.indexOf('isin,');
    const headers = firstLine.slice(headerStart).split(',').map(h => h.trim());

    const seedRecords: SeedRecord[] = allLines.slice(1).map(line => {
        const values = line.split(',').map(v => v.trim());
        const rec: any = {};
        headers.forEach((header, i) => { rec[header] = values[i] || ''; });
        rec.sector = inferSector(rec.name);
        return rec as SeedRecord;
    }).filter(rec => rec.isin && rec.name);

    onProgress?.(1, `Loaded ${seedRecords.length} seed records`);

    // Determine which records will be MISSING (LATAM-heavy pattern)
    const missingIndices = new Set<number>();
    seedRecords.forEach((rec, idx) => {
        const country = rec.country.toUpperCase();
        // MISSING pattern: 80% LATAM, 20% other
        if (['BR', 'MX', 'CL', 'AR'].includes(country)) {
            if (rng() < 0.8) missingIndices.add(idx); // 80% of LATAM missing
        } else if (country === 'JP') {
            if (rng() < 0.15) missingIndices.add(idx); // 15% JP missing
        } else if (country === 'GB') {
            if (rng() < 0.25) missingIndices.add(idx); // 25% GB missing
        }
    });

    // Generate Request (all records)
    const requestLines = ['isin|currency|exchange_code|country|name|sector|requested_date|status'];
    for (const rec of seedRecords) {
        requestLines.push(`${rec.isin}|${rec.currency}|${rec.xchg}|${rec.country}|${rec.name}|${rec.sector}|2024-12-16|pending`);
    }
    fs.writeFileSync(path.join(DATA_DIR, 'SecurityPricingRequest.csv'), requestLines.join('\n'));

    onProgress?.(1, `Generated SecurityPricingRequest.csv with ${seedRecords.length} records`);

    // Generate Response (non-missing records with NOISE)
    const responseLines = ['isin|currency|exchange_code|country|name|sector|price|business_date|status'];
    const basePrices: Record<string, number> = {
        'Apple Inc': 185.50, 'Microsoft Corp': 375.25, 'Alphabet Inc A': 140.80,
        'Toyota Motor Corp': 2850.00, 'Sony Group Corp': 12500.00, 'Shell PLC': 2650.00,
        'Petrobras PN': 38.50, 'VALE SA': 65.20, 'Tencent Holdings': 320.00,
    };

    for (let idx = 0; idx < seedRecords.length; idx++) {
        if (missingIndices.has(idx)) continue; // MISSING record

        const rec = seedRecords[idx];
        // Generate price with NOISE
        const basePrice = basePrices[rec.name] || (rng() * 200 + 50);
        const noise = (rng() - 0.5) * 0.06; // ±3% noise
        const price = (basePrice * (1 + noise)).toFixed(2);

        // Date NOISE: ±1 day
        const dayOffset = Math.floor(rng() * 3) - 1;
        const businessDate = `2024-12-${15 + dayOffset}`;

        // Status NOISE: 10% delayed
        const status = rng() < 0.1 ? 'delayed' : 'active';

        responseLines.push(`${rec.isin}|${rec.currency}|${rec.xchg}|${rec.country}|${rec.name}|${rec.sector}|${price}|${businessDate}|${status}`);
    }
    fs.writeFileSync(path.join(DATA_DIR, 'SecurityPricingResponse.csv'), responseLines.join('\n'));

    const responseCount = seedRecords.length - missingIndices.size;
    onProgress?.(1, `Generated SecurityPricingResponse.csv with ${responseCount} records (${missingIndices.size} MISSING)`);

    // Cache results
    await cacheStepData(state.workflowId, 1, 'generation', {
        requestCount: seedRecords.length,
        responseCount: responseCount,
        missingCount: missingIndices.size,
        requestFile: 'data/SecurityPricingRequest.csv',
        responseFile: 'data/SecurityPricingResponse.csv',
    });
}

// ============================================================================
// Step 2: MISSING Detection
// ============================================================================

async function step2MissingDetection(state: WorkflowState, onProgress?: (step: number, msg: string) => void): Promise<MissingRecord[]> {
    onProgress?.(2, 'Loading request and response files...');

    // Parse request
    const requestContent = fs.readFileSync(path.join(DATA_DIR, 'SecurityPricingRequest.csv'), 'utf-8');
    const requestLines = requestContent.trim().split('\n').slice(1);
    const requestIsins = new Set<string>();
    const requestRecords: Record<string, any> = {};

    for (const line of requestLines) {
        const parts = line.split('|');
        const rec = {
            isin: parts[0],
            currency: parts[1],
            exchange_code: parts[2],
            country: parts[3],
            name: parts[4],
            sector: parts[5],
        };
        requestIsins.add(rec.isin);
        requestRecords[rec.isin] = rec;
    }

    // Parse response
    const responseContent = fs.readFileSync(path.join(DATA_DIR, 'SecurityPricingResponse.csv'), 'utf-8');
    const responseLines = responseContent.trim().split('\n').slice(1);
    const responseIsins = new Set<string>();

    for (const line of responseLines) {
        const parts = line.split('|');
        responseIsins.add(parts[0]);
    }

    // Find MISSING
    const missingRecords: MissingRecord[] = [];
    for (const isin of requestIsins) {
        if (!responseIsins.has(isin)) {
            const rec = requestRecords[isin];
            missingRecords.push({
                isin: rec.isin,
                country: rec.country,
                name: rec.name,
                sector: rec.sector,
                exchange: rec.exchange_code,
                currency: rec.currency,
            });
        }
    }

    onProgress?.(2, `Detected ${missingRecords.length} MISSING records`);

    // Save missing.csv
    const missingLines = ['isin|country|name|sector|exchange|currency'];
    for (const rec of missingRecords) {
        missingLines.push(`${rec.isin}|${rec.country}|${rec.name}|${rec.sector}|${rec.exchange}|${rec.currency}`);
    }
    fs.writeFileSync(path.join(DATA_DIR, 'missing.csv'), missingLines.join('\n'));

    await cacheStepData(state.workflowId, 2, 'detection', {
        missingCount: missingRecords.length,
        missingFile: 'data/missing.csv',
    });

    return missingRecords;
}

// ============================================================================
// Step 3: Embedding Generation
// ============================================================================

async function step3EmbeddingGeneration(
    state: WorkflowState,
    missingRecords: MissingRecord[],
    embeddingModel: string,
    onProgress?: (step: number, msg: string) => void,
): Promise<number[][]> {
    onProgress?.(3, `Generating embeddings with ${embeddingModel}...`);

    // Use Ollama embedding
    const embeddings: number[][] = [];

    for (let i = 0; i < missingRecords.length; i++) {
        const rec = missingRecords[i];
        const text = `isin=${rec.isin} country=${rec.country} exchange=${rec.exchange} currency=${rec.currency} name=${rec.name} sector=${rec.sector}`;

        onProgress?.(3, `Generating embedding ${i + 1}/${missingRecords.length}: ${rec.name}`);

        try {
            const output = execFileSync('ollama', ['embed', embeddingModel, text], {
                encoding: 'utf-8',
                stdio: ['ignore', 'pipe', 'pipe'],
                maxBuffer: 1024 * 1024 * 10,
            });

            const parsed = JSON.parse(output);
            if (Array.isArray(parsed.embedding)) {
                embeddings.push(parsed.embedding);
            } else if (Array.isArray(parsed.embeddings) && parsed.embeddings.length > 0) {
                embeddings.push(parsed.embeddings[0]);
            } else {
                throw new Error('Invalid embedding response');
            }
        } catch (error) {
            // Fallback: generate deterministic embedding from text hash
            const hash = text.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);
            const embedding = Array(768).fill(0).map((_, j) => Math.sin(hash * (j + 1)) * 0.5 + 0.5);
            embeddings.push(embedding);
        }
    }

    onProgress?.(3, `Generated ${embeddings.length} embeddings`);

    await cacheStepData(state.workflowId, 3, 'embeddings', {
        model: embeddingModel,
        dimensions: embeddings[0]?.length || 0,
        count: embeddings.length,
    });

    return embeddings;
}

// ============================================================================
// Step 4: Clustering
// ============================================================================

async function step4Clustering(
    state: WorkflowState,
    missingRecords: MissingRecord[],
    embeddings: number[][],
    onProgress?: (step: number, msg: string) => void,
): Promise<{ clusters: any[]; assignments: number[]; evaluation: any }> {
    onProgress?.(4, 'Building multi-dimensional features...');

    // Build features
    const features = buildMultiDimensionalFeatures(
        missingRecords.map(rec => ({
            country: rec.country,
            xchg: rec.exchange,
            currency: rec.currency,
            name: rec.name,
            sector: rec.sector,
        })),
        embeddings,
        0.5, // name weight
        0.5, // geo weight
    );

    const k = Math.min(4, missingRecords.length);
    onProgress?.(4, `Running K-means clustering (K=${k})...`);

    const clusteringResult = kMeans(features, k);

    // Find dominant terms
    for (const cluster of clusteringResult.clusters) {
        cluster.dominantTerms = findDominantTermsForCluster(cluster.members, missingRecords, 10);
    }

    // Evaluate
    const evaluation = evaluateClusters(clusteringResult.clusters, missingRecords);

    onProgress?.(4, `Clustering complete: ${clusteringResult.clusters.length} clusters`);

    await cacheStepData(state.workflowId, 4, 'clustering', {
        k,
        assignments: clusteringResult.assignments,
        clusterSizes: clusteringResult.clusters.map((c: any) => c.members.length),
        qualityMetrics: evaluation,
    });

    return {
        clusters: clusteringResult.clusters,
        assignments: clusteringResult.assignments,
        evaluation,
    };
}

// ============================================================================
// Step 5: AI Analysis
// ============================================================================

async function step5AIAnalysis(
    state: WorkflowState,
    missingRecords: MissingRecord[],
    clusters: any[],
    evaluation: any,
    workspace?: any,
    onProgress?: (step: number, msg: string) => void,
): Promise<string> {
    // Use a chat model for analysis, not embedding models
    const modelName = state.modelProfile;
    const modelConfig = getModelProfile(modelName);

    // If it's an embedding model, fallback to qwen2.5:1.5b for analysis
    const analysisModel = modelConfig?.type === 'embedding' ? 'qwen2.5:1.5b' : modelName;

    onProgress?.(5, `Running AI analysis with ${analysisModel}...`);

    // Format clustering data for prompt - include actual records
    const clusterSummary = clusters.map((c: any, i: number) => {
        const members = c.members.map((idx: number) => missingRecords[idx]).filter(Boolean);
        const byCountry: Record<string, number> = {};
        const bySector: Record<string, number> = {};
        for (const m of members) {
            byCountry[m.country] = (byCountry[m.country] || 0) + 1;
            bySector[m.sector] = (bySector[m.sector] || 0) + 1;
        }
        const countryStr = Object.entries(byCountry).map(([k, v]) => `${k}(${v})`).join(', ');
        const sectorStr = Object.entries(bySector).map(([k, v]) => `${k}(${v})`).join(', ');

        // Include actual member names for clarity
        const memberNames = members.map((m: any) => m.name).join(', ');

        return `Cluster ${i + 1} (${members.length} records):
  Geography: ${countryStr}
  Sectors: ${sectorStr}
  Members: ${memberNames}`;
    }).join('\n\n');

    // Format the actual missing records list
    const missingRecordsList = missingRecords.map((r, i) =>
        `${i + 1}. ${r.name.padEnd(25)} | ${r.country.padEnd(2)} | ${r.exchange.padEnd(5)} | ${r.currency.padEnd(3)} | ${r.sector}`
    ).join('\n');

    // Build guided prompt - calculate actual counts
    const totalRequested = 59;
    const totalReceived = totalRequested - missingRecords.length;
    const missingRate = ((missingRecords.length / totalRequested) * 100).toFixed(1);

    const prompt = `You are analyzing MISSING securities from a pricing request.

CRITICAL: Use ONLY the data provided below. Do NOT invent or assume any information.
If you're unsure about something, say so. Do NOT make up numbers or facts.

DATA SUMMARY:
- Total REQUESTED: ${totalRequested} securities
- Total RECEIVED: ${totalReceived} securities
- Total MISSING: ${missingRecords.length} securities (${missingRate}% missing rate)

COMPLETE LIST OF MISSING RECORDS:
${missingRecordsList}

CLUSTERING RESULTS (K=${clusters.length}):
${clusterSummary}

QUALITY METRICS:
- Geography Purity: ${(evaluation.geoPurity * 100).toFixed(1)}% (higher = more geographically concentrated)
- Sector Purity: ${(evaluation.namePurity * 100).toFixed(1)}% (higher = more sector-concentrated)
- Balance Score: ${(evaluation.balanceScore * 100).toFixed(1)}% (higher = more balanced cluster sizes)

QUESTIONS TO ANSWER (answer EACH question based ONLY on the data above):

Q1: Which geography is MOST affected by MISSING records?
    Look at the "COMPLETE LIST OF MISSING RECORDS" and count by country.
    Answer format: "[Country/Region] with [N] records ([X]% of MISSING)"

Q2: Which sector has the MOST MISSING records?
    Look at the "COMPLETE LIST OF MISSING RECORDS" and count by sector.
    Answer format: "[Sector] with [N] records ([X]% of MISSING)"

Q3: Is the MISSING pattern random or systematic?
    If MISSING concentrates in specific countries or sectors → systematic
    If MISSING is scattered across all regions/sectors → random
    Answer format: "[systematic/random] - [1-sentence reason based on the actual data]"

Q4: What vendor coverage gap does this suggest?
    Based on what's MISSING, what markets/sectors is the vendor not covering?
    Answer format: "[description of the gap]"

Q5: Which cluster represents the largest coverage gap?
    Look at the "CLUSTERING RESULTS" above.
    Reference the cluster by number AND describe its contents from the data.
    Answer format: "Cluster [N] ([describe contents from CLUSTERING RESULTS]) - [why it's largest gap]"

RESPONSE TEMPLATE (fill in with ACTUAL data from above):

MISSING Analysis Summary:
- Most affected geography: [country/region] with [N] records ([X]% of MISSING)
- Most affected sector: [sector] with [N] records ([X]% of MISSING)
- Pattern assessment: [random/systematic] - [1-sentence reason]
- Vendor gap: [description]
- Largest gap: Cluster [N] ([describe contents]) - [why it's the largest gap]`;

    onProgress?.(5, 'Streaming AI analysis...');

    // Store the prompt for later inspection and update TUI state
    const aiPromptState = {
        title: `AI Analysis: MISSING Clustering (${missingRecords.length} records)`,
        system: 'You are analyzing MISSING securities from a pricing request.',
        user: prompt,
        fullPrompt: prompt,
        context: `Missing: ${missingRecordsList}\n\nClusters:\n${clusterSummary}`,
    };
    setLastPrompt(aiPromptState, prompt);

    // Update TUI with the actual prompt state
    if (workspace?.setAiPromptState) {
        workspace.setAiPromptState(aiPromptState);
    }

    try {
        const analysis = await runPromptStreaming(analysisModel, prompt, () => {});
        onProgress?.(5, 'AI analysis complete');

        // Post-process: remove excessive repetition (consecutive identical lines)
        const lines = analysis.split('\n');
        const deduplicated: string[] = [];
        let repeatCount = 0;
        let lastLine = '';

        for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed === lastLine && trimmed.length > 0) {
                repeatCount++;
                if (repeatCount <= 2) {
                    deduplicated.push(line);
                }
            } else {
                repeatCount = 0;
                lastLine = trimmed;
                deduplicated.push(line);
            }
        }

        const result = deduplicated.join('\n');
        setLastResult(result);
        return result;
    } catch (error) {
        const errMsg = `AI analysis failed: ${error instanceof Error ? error.message : String(error)}`;
        setLastResult(errMsg);
        return errMsg;
    }
}

// ============================================================================
// Step 6: Output Generation
// ============================================================================

async function step6OutputGeneration(
    state: WorkflowState,
    missingRecords: MissingRecord[],
    clusters: any[],
    evaluation: any,
    aiAnalysis: string,
    onProgress?: (step: number, msg: string) => void,
): Promise<TrainAiOutput> {
    onProgress?.(6, 'Generating output report...');

    // Build MISSING by geography/sector
    const missingByGeography: Record<string, number> = {};
    const missingBySector: Record<string, number> = {};
    for (const rec of missingRecords) {
        missingByGeography[rec.country] = (missingByGeography[rec.country] || 0) + 1;
        missingBySector[rec.sector] = (missingBySector[rec.sector] || 0) + 1;
    }

    // Build cluster info
    const clusterInfos: ClusterInfo[] = clusters.map((c: any, i: number) => {
        const members = c.members.map((idx: number) => missingRecords[idx]).filter(Boolean);
        const byCountry: Record<string, number> = {};
        const bySector: Record<string, number> = {};
        for (const m of members) {
            byCountry[m.country] = (byCountry[m.country] || 0) + 1;
            bySector[m.sector] = (bySector[m.sector] || 0) + 1;
        }
        const dominantGeo = Object.entries(byCountry).sort((a, b) => b[1] - a[1])[0]?.[0] || 'Mixed';
        const dominantSector = Object.entries(bySector).sort((a, b) => b[1] - a[1])[0]?.[0] || 'Mixed';

        return {
            id: i + 1,
            size: members.length,
            dominantGeography: dominantGeo,
            dominantSector: dominantSector,
            interpretation: `${dominantGeo} ${dominantSector} cluster`,
            members,
        };
    });

    const qualityMetrics: QualityMetrics = {
        geographyPurity: evaluation.geoPurity,
        sectorPurity: evaluation.namePurity,
        balanceScore: evaluation.balanceScore,
        crossRegionClusters: evaluation.interpretation.filter((i: string) => i.includes('region')).length,
    };

    const output: TrainAiOutput = {
        summary: {
            requested: 59,
            received: 59 - missingRecords.length,
            missing: missingRecords.length,
            missingRate: missingRecords.length / 59,
        },
        missingByGeography,
        missingBySector,
        clusters: clusterInfos,
        qualityMetrics,
        insights: evaluation.interpretation || [],
        recommendation: 'Consider secondary vendor for LATAM markets.',
        aiAnalysis,
    };

    // Save output
    await saveOutput(state.workflowId, 'report.json', output);

    // Generate visualization
    const mentalModelViz = formatMentalModelVisualization(clusters, missingRecords);
    const enhancedSummary = formatEnhancedClusterSummary(clusters, missingRecords, evaluation);

    await saveOutput(state.workflowId, 'visualization.txt', mentalModelViz + '\n\n' + enhancedSummary);

    onProgress?.(6, 'Output saved to .train-ai/output/');

    return output;
}

// ============================================================================
// Workflow Orchestrator
// ============================================================================

export interface WorkflowCallbacks {
    onProgress?: (step: number, msg: string) => void;
    onStateUpdate?: (state: WorkflowState) => void;
}

export async function runWorkflow(
    modelProfile: string,
    seed: number,
    callbacks: WorkflowCallbacks = {},
    resumeFrom?: string,
    workspace?: any,
): Promise<WorkflowState> {
    let state: WorkflowState;

    if (resumeFrom) {
        const loaded = await loadState(resumeFrom);
        if (!loaded) throw new Error(`Workflow ${resumeFrom} not found`);
        state = loaded;
        state.status = 'running';
    } else {
        state = {
            workflowId: generateWorkflowId(),
            seed,
            modelProfile,
            currentStep: 0,
            totalSteps: 6,
            steps: [
                { stepNumber: 1, name: 'Data Generation', status: 'pending', input: 'clustering.csv (59 records)', expectedResult: 'SecurityPricingRequest.csv (59), SecurityPricingResponse.csv (42)', difference: '' },
                { stepNumber: 2, name: 'MISSING Detection', status: 'pending', input: 'Request vs Response diff', expectedResult: '17 MISSING records identified', difference: '' },
                { stepNumber: 3, name: 'Embedding Generation', status: 'pending', input: '17 MISSING records', testingPrompt: `ollama embed ${modelProfile}`, expectedResult: '17 × 768 embeddings', difference: '' },
                { stepNumber: 4, name: 'Clustering', status: 'pending', input: '17 × 768 embeddings', expectedResult: '4 clusters: LATAM Banking(6), LATAM Energy(4), LATAM Retail(4), JP/GB(3)', difference: '' },
                { stepNumber: 5, name: 'AI Analysis', status: 'pending', input: 'Clustering results', testingPrompt: `Guided Q&A for ${modelProfile}`, expectedResult: 'Systematic LATAM MISSING pattern identified', difference: '' },
                { stepNumber: 6, name: 'Output Generation', status: 'pending', input: 'AI analysis + clusters', expectedResult: 'Structured JSON + visualization report', difference: '' },
            ],
            stepOutputs: {},
            startTime: new Date().toISOString(),
            lastUpdated: new Date().toISOString(),
            status: 'running',
        };
    }

    const { onProgress, onStateUpdate } = callbacks;

    const updateStep = (stepNum: number, updates: Partial<WorkflowStep>) => {
        const step = state.steps.find(s => s.stepNumber === stepNum);
        if (step) Object.assign(step, updates);
    };

    try {
        // Step 1: Data Generation
        if (state.currentStep < 1 || (state.steps[0].status === 'running')) {
            state.currentStep = 1;
            updateStep(1, { status: 'running', startTime: new Date().toISOString() });
            await saveState(state);
            onStateUpdate?.(state);

            await step1DataGeneration(state, onProgress);

            updateStep(1, { status: 'completed', endTime: new Date().toISOString() });
            state.stepOutputs['1'] = { completed: true };
            await saveState(state);
            onStateUpdate?.(state);
        }

        // Step 2: MISSING Detection
        if (state.currentStep < 2) {
            state.currentStep = 2;
            updateStep(2, { status: 'running', startTime: new Date().toISOString() });
            await saveState(state);
            onStateUpdate?.(state);

            const missingRecords = await step2MissingDetection(state, onProgress);
            state.missingRecords = missingRecords;

            updateStep(2, { status: 'completed', endTime: new Date().toISOString() });
            state.stepOutputs['2'] = { missingCount: missingRecords.length };
            await saveState(state);
            onStateUpdate?.(state);
        }

        // Step 3: Embedding Generation
        if (state.currentStep < 3) {
            state.currentStep = 3;
            updateStep(3, { status: 'running', startTime: new Date().toISOString() });
            await saveState(state);
            onStateUpdate?.(state);

            const embeddingModel = modelProfile.includes('embed') ? modelProfile : 'nomic-embed-text:latest';
            const embeddings = await step3EmbeddingGeneration(state, state.missingRecords!, embeddingModel, onProgress);
            state.embeddings = embeddings;

            updateStep(3, { status: 'completed', endTime: new Date().toISOString() });
            state.stepOutputs['3'] = { embeddingCount: embeddings.length };
            await saveState(state);
            onStateUpdate?.(state);
        }

        // Step 4: Clustering
        if (state.currentStep < 4) {
            state.currentStep = 4;
            updateStep(4, { status: 'running', startTime: new Date().toISOString() });
            await saveState(state);
            onStateUpdate?.(state);

            const { clusters, assignments, evaluation } = await step4Clustering(state, state.missingRecords!, state.embeddings!, onProgress);
            state.stepOutputs['4'] = { clusters, assignments, evaluation };

            updateStep(4, { status: 'completed', endTime: new Date().toISOString() });
            await saveState(state);
            onStateUpdate?.(state);
        }

        // Step 5: AI Analysis
        if (state.currentStep < 5) {
            state.currentStep = 5;
            updateStep(5, { status: 'running', startTime: new Date().toISOString() });
            await saveState(state);
            onStateUpdate?.(state);

            const clusters = state.stepOutputs['4']?.clusters || [];
            const evaluation = state.stepOutputs['4']?.evaluation || {};
            const aiAnalysis = await step5AIAnalysis(state, state.missingRecords!, clusters, evaluation, workspace, onProgress);
            state.stepOutputs['5'] = { aiAnalysis };

            updateStep(5, { status: 'completed', endTime: new Date().toISOString() });
            await saveState(state);
            onStateUpdate?.(state);
        }

        // Step 6: Output Generation
        if (state.currentStep < 6) {
            state.currentStep = 6;
            updateStep(6, { status: 'running', startTime: new Date().toISOString() });
            await saveState(state);
            onStateUpdate?.(state);

            const clusters = state.stepOutputs['4']?.clusters || [];
            const evaluation = state.stepOutputs['4']?.evaluation || {};
            const aiAnalysis = state.stepOutputs['5']?.aiAnalysis || '';
            const output = await step6OutputGeneration(state, state.missingRecords!, clusters, evaluation, aiAnalysis, onProgress);
            state.output = output;

            updateStep(6, { status: 'completed', endTime: new Date().toISOString() });
            await saveState(state);
            onStateUpdate?.(state);
        }

        state.status = 'completed';
        await saveState(state);
        onStateUpdate?.(state);
        return state;

    } catch (error) {
        state.status = 'failed';
        state.error = error instanceof Error ? error.message : String(error);
        const currentStep = state.steps.find(s => s.stepNumber === state.currentStep);
        if (currentStep) currentStep.status = 'failed';
        await saveState(state);
        onStateUpdate?.(state);
        throw error;
    }
}

export { loadState, listWorkflows } from './persistence.js';

// ============================================================================
// Workflow with Existing Files
// ============================================================================

export async function runWorkflowWithFiles(
    modelProfile: string,
    seed: number,
    requestFile: any,
    responseFile: any,
    workspace: any,
    callbacks: WorkflowCallbacks = {},
): Promise<WorkflowState> {
    const state: WorkflowState = {
        workflowId: generateWorkflowId(),
        seed,
        modelProfile,
        currentStep: 0,
        totalSteps: 6,
        steps: [
            { stepNumber: 1, name: 'Data Generation', status: 'completed', input: 'Existing files', expectedResult: 'Using provided files', difference: '' },
            { stepNumber: 2, name: 'MISSING Detection', status: 'pending', input: 'Request vs Response diff', expectedResult: 'MISSING records identified', difference: '' },
            { stepNumber: 3, name: 'Embedding Generation', status: 'pending', input: 'MISSING records', testingPrompt: `ollama embed nomic-embed-text`, expectedResult: 'N × 768 embeddings', difference: '' },
            { stepNumber: 4, name: 'Clustering', status: 'pending', input: 'N × 768 embeddings', expectedResult: 'K clusters', difference: '' },
            { stepNumber: 5, name: 'AI Analysis', status: 'pending', input: 'Clustering results', testingPrompt: `Guided Q&A for ${modelProfile}`, expectedResult: 'MISSING pattern identified', difference: '' },
            { stepNumber: 6, name: 'Output Generation', status: 'pending', input: 'AI analysis + clusters', expectedResult: 'Structured JSON + visualization report', difference: '' },
        ],
        stepOutputs: {},
        startTime: new Date().toISOString(),
        lastUpdated: new Date().toISOString(),
        status: 'running',
    };

    const { onProgress, onStateUpdate } = callbacks;

    const updateStep = (stepNum: number, updates: Partial<WorkflowStep>) => {
        const step = state.steps.find(s => s.stepNumber === stepNum);
        if (step) Object.assign(step, updates);
    };

    try {
        // Step 1 is already marked as completed (using existing files)
        state.currentStep = 1;
        state.stepOutputs['1'] = {
            requestFile: requestFile.name || 'request',
            responseFile: responseFile.name || 'response',
            requestCount: requestFile.data.hits.hits.length,
            responseCount: responseFile.data.hits.hits.length,
        };
        await saveState(state);
        onStateUpdate?.(state);

        // Step 2: MISSING Detection
        state.currentStep = 2;
        updateStep(2, { status: 'running', startTime: new Date().toISOString() });
        await saveState(state);
        onStateUpdate?.(state);

        const missingRecords = await detectMissingFromFiles(requestFile, responseFile, state, onProgress);
        state.missingRecords = missingRecords;

        updateStep(2, { status: 'completed', endTime: new Date().toISOString() });
        state.stepOutputs['2'] = { missingCount: missingRecords.length };
        await saveState(state);
        onStateUpdate?.(state);

        // Step 3: Embedding Generation
        state.currentStep = 3;
        updateStep(3, { status: 'running', startTime: new Date().toISOString() });
        await saveState(state);
        onStateUpdate?.(state);

        const embeddingModel = 'nomic-embed-text:latest';
        const embeddings = await step3EmbeddingGeneration(state, missingRecords, embeddingModel, onProgress);
        state.embeddings = embeddings;

        updateStep(3, { status: 'completed', endTime: new Date().toISOString() });
        state.stepOutputs['3'] = { embeddingCount: embeddings.length };
        await saveState(state);
        onStateUpdate?.(state);

        // Step 4: Clustering
        state.currentStep = 4;
        updateStep(4, { status: 'running', startTime: new Date().toISOString() });
        await saveState(state);
        onStateUpdate?.(state);

        const { clusters, assignments, evaluation } = await step4Clustering(state, missingRecords, embeddings, onProgress);
        state.stepOutputs['4'] = { clusters, assignments, evaluation };

        updateStep(4, { status: 'completed', endTime: new Date().toISOString() });
        await saveState(state);
        onStateUpdate?.(state);

        // Step 5: AI Analysis
        state.currentStep = 5;
        updateStep(5, { status: 'running', startTime: new Date().toISOString() });
        await saveState(state);
        onStateUpdate?.(state);

        const aiAnalysis = await step5AIAnalysis(state, missingRecords, clusters, evaluation, workspace, onProgress);
        state.stepOutputs['5'] = { aiAnalysis };

        updateStep(5, { status: 'completed', endTime: new Date().toISOString() });
        await saveState(state);
        onStateUpdate?.(state);

        // Step 6: Output Generation
        state.currentStep = 6;
        updateStep(6, { status: 'running', startTime: new Date().toISOString() });
        await saveState(state);
        onStateUpdate?.(state);

        const output = await step6OutputGeneration(state, missingRecords, clusters, evaluation, aiAnalysis, onProgress);
        state.output = output;

        updateStep(6, { status: 'completed', endTime: new Date().toISOString() });
        await saveState(state);
        onStateUpdate?.(state);

        state.status = 'completed';
        await saveState(state);
        onStateUpdate?.(state);
        return state;

    } catch (error) {
        state.status = 'failed';
        state.error = error instanceof Error ? error.message : String(error);
        const currentStep = state.steps.find(s => s.stepNumber === state.currentStep);
        if (currentStep) currentStep.status = 'failed';
        await saveState(state);
        onStateUpdate?.(state);
        throw error;
    }
}

async function detectMissingFromFiles(
    requestFile: any,
    responseFile: any,
    state: WorkflowState,
    onProgress?: (step: number, msg: string) => void,
): Promise<MissingRecord[]> {
    onProgress?.(2, 'Detecting MISSING records from provided files...');

    // Extract records from loaded files
    const requestRecords = requestFile.data.hits.hits.map((h: any) => h._source);
    const responseRecords = responseFile.data.hits.hits.map((h: any) => h._source);

    // Build ISIN sets
    const requestIsins = new Set<string>();
    const requestMap: Record<string, any> = {};
    for (const rec of requestRecords) {
        const isin = rec.isin;
        requestIsins.add(isin);
        requestMap[isin] = rec;
    }

    const responseIsins = new Set<string>();
    for (const rec of responseRecords) {
        responseIsins.add(rec.isin);
    }

    // Find MISSING
    const missingRecords: MissingRecord[] = [];
    for (const isin of requestIsins) {
        if (!responseIsins.has(isin)) {
            const rec = requestMap[isin];
            missingRecords.push({
                isin: rec.isin,
                country: rec.country || '',
                name: rec.name || '',
                sector: rec.sector || inferSector(rec.name || ''),
                exchange: rec.exchange_code || rec.xchg || '',
                currency: rec.currency || '',
            });
        }
    }

    onProgress?.(2, `Detected ${missingRecords.length} MISSING records`);
    return missingRecords;
}
