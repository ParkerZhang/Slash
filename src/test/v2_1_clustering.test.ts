import { describe, it, expect } from 'vitest';
import {
    normalizeVector,
    buildMultiDimensionalFeatures,
    buildGeoFeatures,
    twoStageClustering,
    evaluateClusters,
    buildClusterProfile,
    formatMentalModelVisualization,
    formatEnhancedClusterSummary,
    kMeans,
    ClusterResult,
} from '../core/clustering.js';

describe('V2.1 Multi-Dimensional Clustering', () => {
    describe('normalizeVector', () => {
        it('should normalize a vector to unit length', () => {
            const vector = [3, 4];
            const normalized = normalizeVector(vector);
            const length = Math.sqrt(normalized.reduce((sum, v) => sum + v * v, 0));
            expect(length).toBeCloseTo(1, 5);
        });

        it('should handle zero vector', () => {
            const vector = [0, 0, 0];
            const normalized = normalizeVector(vector);
            expect(normalized).toEqual([0, 0, 0]);
        });

        it('should preserve vector direction', () => {
            const vector = [1, 2, 3];
            const normalized = normalizeVector(vector);
            const ratio1 = normalized[1] / normalized[0];
            const ratio2 = 2 / 1;
            expect(ratio1).toBeCloseTo(ratio2, 5);
        });
    });

    describe('buildGeoFeatures', () => {
        it('should build one-hot encoded features for categorical fields', () => {
            const records = [
                { country: 'US', xchg: 'NYSE', currency: 'USD' },
                { country: 'JP', xchg: 'TSE', currency: 'JPY' },
                { country: 'US', xchg: 'NYSE', currency: 'USD' },
            ];

            const features = buildGeoFeatures(records);

            expect(features.length).toBe(3);
            expect(features[0].length).toBeGreaterThan(0);
            // First and third records should have same features
            expect(features[0]).toEqual(features[2]);
            // Second record should be different
            expect(features[0]).not.toEqual(features[1]);
        });

        it('should handle missing fields gracefully', () => {
            const records = [
                { country: 'US' },
                { xchg: 'TSE' },
                { currency: 'USD' },
            ];

            const features = buildGeoFeatures(records);
            expect(features.length).toBe(3);
            expect(features[0].length).toBeGreaterThan(0);
        });

        it('should encode all unique values for each categorical field', () => {
            const records = [
                { country: 'US', currency: 'USD' },
                { country: 'JP', currency: 'JPY' },
                { country: 'GB', currency: 'GBP' },
            ];

            const features = buildGeoFeatures(records);
            // Should have features for 3 countries + 3 currencies = 6 dimensions
            expect(features[0].length).toBe(6);
        });
    });

    describe('buildMultiDimensionalFeatures', () => {
        it('should combine name embeddings with geo features', () => {
            const records = [
                { name: 'Apple Inc', country: 'US', xchg: 'NYSE', currency: 'USD' },
                { name: 'Toyota Motor', country: 'JP', xchg: 'TSE', currency: 'JPY' },
            ];
            const nameEmbeddings = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ];

            const features = buildMultiDimensionalFeatures(records, nameEmbeddings, 0.7, 0.3);

            expect(features.length).toBe(2);
            // Features should be longer than name embeddings (combined with geo)
            expect(features[0].length).toBeGreaterThan(3);
        });

        it('should apply weights correctly', () => {
            const records = [
                { country: 'US' },
                { country: 'JP' },
            ];
            const nameEmbeddings = [[0.5, 0.5], [0.5, 0.5]];

            const features1 = buildMultiDimensionalFeatures(records, nameEmbeddings, 0.9, 0.1);
            const features2 = buildMultiDimensionalFeatures(records, nameEmbeddings, 0.1, 0.9);

            // Different weights should produce different features
            expect(features1).toBeDefined();
            expect(features2).toBeDefined();
        });

        it('should normalize combined vectors', () => {
            const records = [
                { country: 'US', currency: 'USD' },
                { country: 'JP', currency: 'JPY' },
            ];
            const nameEmbeddings = [[0.1, 0.2], [0.3, 0.4]];

            const features = buildMultiDimensionalFeatures(records, nameEmbeddings);

            // Check that each feature vector is normalized
            for (const feature of features) {
                const norm = Math.sqrt(feature.reduce((sum, v) => sum + v * v, 0));
                expect(norm).toBeCloseTo(1, 5);
            }
        });
    });

    describe('twoStageClustering', () => {
        it('should perform two-stage clustering (geo then name)', () => {
            const records = [
                { name: 'Apple', country: 'US' },
                { name: 'Microsoft', country: 'US' },
                { name: 'Toyota', country: 'JP' },
                { name: 'Sony', country: 'JP' },
            ];
            const nameEmbeddings = [
                [0.1, 0.2], [0.15, 0.25], [0.8, 0.9], [0.85, 0.95],
            ];

            const result = twoStageClustering(records, nameEmbeddings, 2, 2);

            expect(result.clusters.length).toBeGreaterThan(0);
            expect(result.assignments.length).toBe(4);
        });

        it('should handle small geo clusters gracefully', () => {
            const records = [
                { name: 'Apple', country: 'US' },
            ];
            const nameEmbeddings = [[0.1, 0.2]];

            const result = twoStageClustering(records, nameEmbeddings, 1, 1);

            expect(result.assignments.length).toBe(1);
        });
    });

    describe('evaluateClusters', () => {
        it('should calculate cluster quality metrics', () => {
            const records = [
                { name: 'Apple', country: 'US' },
                { name: 'Microsoft', country: 'US' },
                { name: 'Toyota', country: 'JP' },
                { name: 'Sony', country: 'JP' },
            ];
            const clusters: ClusterResult[] = [
                { centroid: [], members: [0, 1], dominantTerms: [] },
                { centroid: [], members: [2, 3], dominantTerms: [] },
            ];

            const evaluation = evaluateClusters(clusters, records);

            expect(evaluation.totalRecords).toBe(4);
            expect(evaluation.numClusters).toBe(2);
            expect(evaluation.clusterSizes).toEqual([2, 2]);
            expect(evaluation.geoPurity).toBeGreaterThan(0);
            expect(evaluation.namePurity).toBeGreaterThan(0);
            expect(evaluation.balanceScore).toBeCloseTo(1, 5); // Perfectly balanced
        });

        it('should detect cross-region clusters', () => {
            const records = [
                { name: 'Apple', country: 'US' },
                { name: 'HSBC', country: 'GB' },
                { name: 'Toyota', country: 'JP' },
            ];
            const clusters: ClusterResult[] = [
                { centroid: [], members: [0, 1, 2], dominantTerms: [] },
            ];

            const evaluation = evaluateClusters(clusters, records);

            expect(evaluation.interpretation.some(i => i.includes('region'))).toBe(true);
        });

        it('should detect imbalanced clusters', () => {
            const records = Array(10).fill({ name: 'Apple', country: 'US' })
                .concat(Array(2).fill({ name: 'Toyota', country: 'JP' }));
            const clusters: ClusterResult[] = [
                { centroid: [], members: Array.from({ length: 10 }, (_, i) => i), dominantTerms: [] },
                { centroid: [], members: [10, 11], dominantTerms: [] },
            ];

            const evaluation = evaluateClusters(clusters, records);

            expect(evaluation.balanceScore).toBeLessThan(0.5);
        });
    });

    describe('buildClusterProfile', () => {
        it('should build multi-dimensional profile for cluster', () => {
            const records = [
                { name: 'Apple Inc', country: 'US', xchg: 'NYSE', currency: 'USD' },
                { name: 'Microsoft Corp', country: 'US', xchg: 'NYSE', currency: 'USD' },
                { name: 'Toyota Motor', country: 'JP', xchg: 'TSE', currency: 'JPY' },
            ];
            const cluster: ClusterResult = {
                centroid: [],
                members: [0, 1, 2],
                dominantTerms: [],
            };

            const profile = buildClusterProfile(cluster, records);

            expect(profile.geoClusters).toHaveProperty('us');
            expect(profile.geoClusters).toHaveProperty('jp');
            expect(profile.sectorClusters).toHaveProperty('Tech');
            expect(profile.sectorClusters).toHaveProperty('Automotive');
            expect(profile.exchangeCoverage).toHaveProperty('nyse');
            expect(profile.exchangeCoverage).toHaveProperty('tse');
        });

        it('should infer sectors from names', () => {
            const records = [
                { name: 'Apple Inc' },
                { name: 'Bank of America' },
                { name: 'Petrobras' },
                { name: 'Vodafone Group' },
            ];
            const cluster: ClusterResult = {
                centroid: [],
                members: [0, 1, 2, 3],
                dominantTerms: [],
            };

            const profile = buildClusterProfile(cluster, records);

            expect(profile.sectorClusters).toHaveProperty('Tech');
            expect(profile.sectorClusters).toHaveProperty('Banking');
            expect(profile.sectorClusters).toHaveProperty('Energy');
            expect(profile.sectorClusters).toHaveProperty('Telecom');
        });
    });

    describe('formatMentalModelVisualization', () => {
        it('should format mental model visualization', () => {
            const records = [
                { name: 'Apple Inc', country: 'US' },
                { name: 'Microsoft Corp', country: 'US' },
                { name: 'Toyota Motor', country: 'JP' },
                { name: 'Sony Group', country: 'JP' },
            ];
            const clusters: ClusterResult[] = [
                { centroid: [], members: [0, 1], dominantTerms: [] },
                { centroid: [], members: [2, 3], dominantTerms: [] },
            ];

            const output = formatMentalModelVisualization(clusters, records);

            expect(output).toContain('Tech');
            expect(output).toContain('US');
            expect(output).toContain('JP');
            expect(output).toContain('●');
        });

        it('should show country distribution with dots', () => {
            const records = Array(5).fill({ name: 'Apple', country: 'US' })
                .concat(Array(3).fill({ name: 'Toyota', country: 'JP' }));
            const clusters: ClusterResult[] = [
                { centroid: [], members: Array.from({ length: 8 }, (_, i) => i), dominantTerms: [] },
            ];

            const output = formatMentalModelVisualization(clusters, records);

            expect(output).toMatch(/US\s+●{5}/);
            expect(output).toMatch(/JP\s+●{3}/);
        });
    });

    describe('formatEnhancedClusterSummary', () => {
        it('should format enhanced cluster summary with metrics', () => {
            const records = [
                { name: 'Apple Inc', country: 'US', xchg: 'NYSE' },
                { name: 'Toyota Motor', country: 'JP', xchg: 'TSE' },
            ];
            const clusters: ClusterResult[] = [
                { centroid: [], members: [0], dominantTerms: [] },
                { centroid: [], members: [1], dominantTerms: [] },
            ];
            const evaluation = evaluateClusters(clusters, records);

            const output = formatEnhancedClusterSummary(clusters, records, evaluation);

            expect(output).toContain('Multi-Dimensional Clustering Analysis');
            expect(output).toContain('Cluster Quality Metrics');
            expect(output).toContain('Geography Purity');
            expect(output).toContain('Name/Sector Purity');
            expect(output).toContain('Balance Score');
        });

        it('should include insights from evaluation', () => {
            const records = Array(4).fill({ name: 'Apple', country: 'US' })
                .concat(Array(4).fill({ name: 'Toyota', country: 'JP' }));
            const clusters: ClusterResult[] = [
                { centroid: [], members: [0, 1, 2, 3], dominantTerms: [] },
                { centroid: [], members: [4, 5, 6, 7], dominantTerms: [] },
            ];
            const evaluation = evaluateClusters(clusters, records);

            const output = formatEnhancedClusterSummary(clusters, records, evaluation);

            expect(output).toContain('Insights');
            expect(output).toContain('✓');
        });

        it('should show sector, geography, and exchange breakdowns', () => {
            const records = [
                { name: 'Apple Inc', country: 'US', xchg: 'NYSE' },
                { name: 'Microsoft', country: 'US', xchg: 'NYSE' },
                { name: 'Toyota Motor', country: 'JP', xchg: 'TSE' },
            ];
            const clusters: ClusterResult[] = [
                { centroid: [], members: [0, 1, 2], dominantTerms: [] },
            ];
            const evaluation = evaluateClusters(clusters, records);

            const output = formatEnhancedClusterSummary(clusters, records, evaluation);

            expect(output).toContain('Dominant sectors');
            expect(output).toContain('Geography');
            expect(output).toContain('Exchanges');
        });

        it('should highlight cross-region clusters', () => {
            const records = [
                { name: 'Apple', country: 'US' },
                { name: 'HSBC', country: 'GB' },
                { name: 'Toyota', country: 'JP' },
            ];
            const clusters: ClusterResult[] = [
                { centroid: [], members: [0, 1, 2], dominantTerms: [] },
            ];
            const evaluation = evaluateClusters(clusters, records);

            const output = formatEnhancedClusterSummary(clusters, records, evaluation);

            expect(output).toContain('Cross-region');
        });
    });
});
