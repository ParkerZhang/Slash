import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { listWorkflows } from '../train-ai/workflow.js';
import { MODEL_REGISTRY, listCandidateModels } from '../train-ai/modelRegistry.js';
import { visualizeWorkflowState, visualizeCandidateModels } from '../train-ai/command.js';

describe('Train-AI Module', () => {
    describe('Model Registry', () => {
        it('should have candidate models', () => {
            const candidates = listCandidateModels();
            expect(candidates.length).toBeGreaterThan(0);
            expect(candidates.some(c => c.model === 'qwen2.5:1.5b')).toBe(true);
            expect(candidates.some(c => c.status === 'recommended')).toBe(true);
        });

        it('should have model profiles with ideal prompts', () => {
            expect(MODEL_REGISTRY['qwen2.5:1.5b']).toBeDefined();
            expect(MODEL_REGISTRY['qwen2.5:1.5b'].type).toBe('chat');
            expect(MODEL_REGISTRY['qwen2.5:1.5b'].idealPrompts).toBeDefined();
            expect(MODEL_REGISTRY['qwen2.5:1.5b'].idealPrompts?.guidedAnalysis).toBeDefined();
            expect(MODEL_REGISTRY['qwen2.5:1.5b'].idealPrompts?.jsonOutput).toBeDefined();
        });

        it('should have embedding models', () => {
            expect(MODEL_REGISTRY['nomic-embed-text:latest']).toBeDefined();
            expect(MODEL_REGISTRY['nomic-embed-text:latest'].type).toBe('embedding');
            expect(MODEL_REGISTRY['all-minilm:latest']).toBeDefined();
        });
    });

    describe('Visualization', () => {
        it('should visualize candidate models', () => {
            const output = visualizeCandidateModels();
            expect(output).toContain('CANDIDATE MODELS');
            expect(output).toContain('qwen2.5:1.5b');
            expect(output).toContain('recommended');
            expect(output).toContain('Embedding');
        });

        it('should visualize workflow state', () => {
            const mockState = {
                workflowId: 'test-workflow-001',
                seed: 42,
                modelProfile: 'qwen2.5:1.5b',
                currentStep: 0,
                totalSteps: 6,
                steps: [
                    { stepNumber: 1, name: 'Data Generation', status: 'pending', input: 'clustering.csv (59 records)', expectedResult: 'SecurityPricingRequest.csv (59), SecurityPricingResponse.csv (42)' },
                    { stepNumber: 2, name: 'MISSING Detection', status: 'pending', input: 'Request vs Response diff', expectedResult: '17 MISSING records identified' },
                    { stepNumber: 3, name: 'Embedding Generation', status: 'pending', input: '17 MISSING records', expectedResult: '17 × 768 embeddings' },
                    { stepNumber: 4, name: 'Clustering', status: 'pending', input: '17 × 768 embeddings', expectedResult: '4 clusters' },
                    { stepNumber: 5, name: 'AI Analysis', status: 'pending', input: 'Clustering results', expectedResult: 'Systematic LATAM MISSING pattern' },
                    { stepNumber: 6, name: 'Output Generation', status: 'pending', input: 'AI analysis + clusters', expectedResult: 'Structured JSON + visualization' },
                ],
                startTime: new Date().toISOString(),
                lastUpdated: new Date().toISOString(),
                status: 'running',
                stepOutputs: {},
            };

            const output = visualizeWorkflowState(mockState);
            expect(output).toContain('test-workflow-001');
            expect(output).toContain('qwen2.5:1.5b');
            expect(output).toContain('Data Generation');
            expect(output).toContain('MISSING Detection');
            expect(output).toContain('Embedding Generation');
            expect(output).toContain('Clustering');
            expect(output).toContain('AI Analysis');
            expect(output).toContain('Output Generation');
        });

        it('should visualize completed workflow with output', () => {
            const mockState = {
                workflowId: 'test-workflow-002',
                seed: 42,
                modelProfile: 'qwen2.5:1.5b',
                currentStep: 6,
                totalSteps: 6,
                steps: [
                    { stepNumber: 1, name: 'Data Generation', status: 'completed', input: 'clustering.csv', expectedResult: 'Request/Response', startTime: new Date().toISOString(), endTime: new Date().toISOString() },
                    { stepNumber: 2, name: 'MISSING Detection', status: 'completed', input: 'Diff', expectedResult: 'MISSING records', startTime: new Date().toISOString(), endTime: new Date().toISOString() },
                    { stepNumber: 3, name: 'Embedding Generation', status: 'completed', input: 'MISSING', expectedResult: 'Embeddings', startTime: new Date().toISOString(), endTime: new Date().toISOString() },
                    { stepNumber: 4, name: 'Clustering', status: 'completed', input: 'Embeddings', expectedResult: '4 clusters', startTime: new Date().toISOString(), endTime: new Date().toISOString() },
                    { stepNumber: 5, name: 'AI Analysis', status: 'completed', input: 'Clusters', expectedResult: 'Analysis', testingPrompt: 'Guided Q&A', startTime: new Date().toISOString(), endTime: new Date().toISOString() },
                    { stepNumber: 6, name: 'Output Generation', status: 'completed', input: 'Analysis', expectedResult: 'Report', startTime: new Date().toISOString(), endTime: new Date().toISOString() },
                ],
                startTime: new Date().toISOString(),
                lastUpdated: new Date().toISOString(),
                status: 'completed',
                stepOutputs: {},
                output: {
                    summary: { requested: 59, received: 42, missing: 17, missingRate: 0.288 },
                    missingByGeography: { BR: 6, MX: 4, CL: 3, AR: 2, JP: 2 },
                    missingBySector: { Banking: 6, Energy: 5, Retail: 3, Telecom: 2, Automotive: 1 },
                    clusters: [
                        { id: 1, size: 6, dominantGeography: 'LATAM', dominantSector: 'Banking', interpretation: 'LATAM Banking' },
                        { id: 2, size: 4, dominantGeography: 'LATAM', dominantSector: 'Energy', interpretation: 'LATAM Energy' },
                    ],
                    qualityMetrics: { geographyPurity: 0.813, sectorPurity: 0.785, balanceScore: 0.621, crossRegionClusters: 1 },
                    insights: ['LATAM vendor coverage is limited', 'Banking most affected'],
                    recommendation: 'Consider secondary vendor',
                    aiAnalysis: 'LATAM is most affected with 14 records (82.4%)',
                },
            };

            const output = visualizeWorkflowState(mockState);
            expect(output).toContain('✓'); // completed status
            expect(output).toContain('MISSING ANALYSIS SUMMARY');
            expect(output).toContain('Requested: 59');
            expect(output).toContain('MISSING: 17');
            expect(output).toContain('AI ANALYSIS');
        });
    });

    describe('Workflow State Persistence', () => {
        it('should list workflows (empty)', async () => {
            const workflows = await listWorkflows();
            expect(Array.isArray(workflows)).toBe(true);
        });
    });

    describe('Data Files Generation', () => {
        it('should have clustering.csv seed file', () => {
            const clusteringFile = path.join(process.cwd(), 'data/clustering.csv');
            expect(fs.existsSync(clusteringFile)).toBe(true);
        });
    });
});
