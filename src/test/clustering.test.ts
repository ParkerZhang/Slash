import { describe, it, expect } from 'vitest';
import { kMeans, kMeansCosine, findDominantTermsForCluster, formatClusterSummary } from '../core/clustering.js';

describe('K-Means Clustering', () => {
    it('should cluster 2D points into 2 groups', () => {
        const vectors = [
            [1, 1], [1.1, 1.2], [0.9, 0.8], [1.05, 1.1],
            [9, 9], [9.1, 8.9], [8.8, 9.2], [9.05, 9.1],
        ];

        const result = kMeans(vectors, 2, 100, 1e-4, () => 0.5);

        expect(result.clusters.length).toBe(2);
        expect(result.assignments.length).toBe(8);
        expect(result.clusters[0].members.length + result.clusters[1].members.length).toBe(8);

        // First 4 should be in one cluster, last 4 in another
        const cluster0 = result.clusters[0].members;
        const cluster1 = result.clusters[1].members;
        expect(cluster0.length).toBe(4);
        expect(cluster1.length).toBe(4);
    });

    it('should handle k greater than data points', () => {
        const vectors = [[1, 2], [3, 4]];
        const result = kMeans(vectors, 5);

        expect(result.clusters.length).toBe(2);
        expect(result.assignments.length).toBe(2);
    });

    it('should handle empty input', () => {
        const result = kMeans([], 3);
        expect(result.clusters.length).toBe(0);
        expect(result.assignments.length).toBe(0);
    });

    it('should converge within tolerance', () => {
        const vectors = [
            [0, 0], [0.1, 0], [0, 0.1],
            [10, 10], [10.1, 10], [10, 10.1],
        ];

        const result = kMeans(vectors, 2);
        expect(result.clusters.length).toBe(2);
    });

    it('should assign all points to clusters', () => {
        const vectors = [
            [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
        ];

        const result = kMeans(vectors, 3);
        const allAssigned = result.assignments.every((a) => a >= 0 && a < 3);
        expect(allAssigned).toBe(true);
        expect(result.assignments.length).toBe(4);
    });
});

describe('K-Means Cosine Clustering', () => {
    it('should cluster using cosine distance', () => {
        const vectors = [
            [1, 0, 0], [0.9, 0.1, 0],
            [0, 1, 0], [0.1, 0.9, 0],
        ];

        const result = kMeansCosine(vectors, 2, 100, 1e-4, () => 0.3);

        expect(result.clusters.length).toBe(2);
        expect(result.assignments.length).toBe(4);
    });
});

describe('findDominantTermsForCluster', () => {
    it('should find dominant terms in cluster', () => {
        const records = [
            { exchange: 'NYSE', currency: 'USD', country: 'US' },
            { exchange: 'NYSE', currency: 'USD', country: 'US' },
            { exchange: 'NYSE', currency: 'USD', country: 'US' },
            { exchange: 'TSE', currency: 'JPY', country: 'JP' },
        ];

        const terms = findDominantTermsForCluster([0, 1, 2], records, 5);
        expect(terms.length).toBeGreaterThan(0);
        expect(terms.some((t) => t.includes('exchange:nyse'))).toBe(true);
        expect(terms.some((t) => t.includes('currency:usd'))).toBe(true);
    });

    it('should handle empty cluster', () => {
        const records = [{ field: 'value' }];
        const terms = findDominantTermsForCluster([], records);
        expect(terms.length).toBe(0);
    });

    it('should group by field', () => {
        const records = [
            { color: 'red', size: 'large' },
            { color: 'red', size: 'large' },
            { color: 'blue', size: 'small' },
        ];

        const terms = findDominantTermsForCluster([0, 1], records, 10);
        expect(terms.some((t) => t.startsWith('color:'))).toBe(true);
        expect(terms.some((t) => t.startsWith('size:'))).toBe(true);
    });
});

describe('formatClusterSummary', () => {
    it('should format cluster summary', () => {
        const records = [
            { exchange: 'NYSE', currency: 'USD' },
            { exchange: 'NYSE', currency: 'USD' },
            { exchange: 'TSE', currency: 'JPY' },
            { exchange: 'TSE', currency: 'JPY' },
        ];

        const clusters = [
            { centroid: [0, 0], members: [0, 1], dominantTerms: [] as string[] },
            { centroid: [1, 1], members: [2, 3], dominantTerms: [] as string[] },
        ];

        const summary = formatClusterSummary(clusters, records, 5);
        expect(summary).toContain('Found 2 clusters');
        expect(summary).toContain('Cluster 1');
        expect(summary).toContain('Cluster 2');
    });

    it('should handle single cluster', () => {
        const records = [{ field: 'value' }];
        const clusters = [
            { centroid: [0], members: [0], dominantTerms: [] as string[] },
        ];

        const summary = formatClusterSummary(clusters, records);
        expect(summary).toContain('Found 1 clusters');
    });
});
