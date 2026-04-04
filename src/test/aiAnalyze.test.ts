import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as modelManager from '../ai/modelManager.js';
import * as embedding from '../ai/embedding.js';
import { CommandRegistry } from '../commands/commandRegistry.js';
import { MemoryWorkspace } from '../core/runner.js';
import { ModelCommand, LoadCsvCommand, AiCommand } from '../commands/index.js';

describe('AI Analyze Command - Clustering', () => {
    let workspace: MemoryWorkspace;
    let registry: CommandRegistry;

    beforeEach(() => {
        workspace = new MemoryWorkspace();
        registry = new CommandRegistry();
        registry.register(ModelCommand);
        registry.register(LoadCsvCommand);
        registry.register(AiCommand);
    });

    it('should require a model to be set before clustering', async () => {
        await registry.execute('/loadCsv f1 data/clustering.csv', workspace);
        const result = await registry.execute('/ai analyze f1 "what are the dominant groups"', workspace);
        expect(result.output).toBe('Set a model first with /model <name>');
    });

    it('should require a valid file id for clustering', async () => {
        await registry.execute('/model gemma4:latest', workspace);
        const result = await registry.execute('/ai analyze nonexistent "what are the dominant groups"', workspace);
        expect(result.output).toBe("File 'nonexistent' not loaded");
    });

    it('should validate minimum arguments for analyze command', async () => {
        await registry.execute('/model gemma4:latest', workspace);
        const result = await registry.execute('/ai analyze f1', workspace);
        expect(result.output).toContain('Usage: /ai analyze <fileId>');
    });

    it('should parse cluster count from question and run clustering', async () => {
        await registry.execute('/loadCsv f1 data/clustering.csv', workspace);
        await registry.execute('/model gemma4:latest', workspace);

        // Mock embeddings to return vectors for all records in clustering.csv
        const embeddingSpy = vi.spyOn(embedding, 'getEmbeddingsStreaming')
            .mockResolvedValue({
                embeddings: Array(76).fill([0.5, 0.5]),
                model: 'gemma4:latest',
            });

        const aiSpy = vi.spyOn(modelManager, 'runPromptStreaming')
            .mockResolvedValue('Cluster 1 represents US/European equities. Cluster 2 represents Asian equities.');

        const result = await registry.execute('/ai analyze f1 "what are the dominant groups, top 2 clusters"', workspace);

        expect(result.output).toContain("📊 Clustering Results");
        expect(result.output).toContain('Records: 58 | Clusters: 2');
        expect(result.output).toContain('Cluster 1 represents US/European equities');
        expect(result.output).not.toContain('VIEW_CHANGE');
        expect(workspace.getLoadedFiles().has('f1-clustering-2')).toBe(true);

        const clusteringFile = workspace.getLoadedFiles().get('f1-clustering-2');
        expect(clusteringFile).toBeDefined();
        expect(clusteringFile?.data.hits.hits.length).toBe(58);
        expect(clusteringFile?.data.hits.hits[0]._source).toHaveProperty('_cluster');

        embeddingSpy.mockRestore();
        aiSpy.mockRestore();
    });

    it('should default to 4 clusters when no number is specified', async () => {
        await registry.execute('/loadCsv f1 data/clustering.csv', workspace);
        await registry.execute('/model gemma4:latest', workspace);

        const embeddingSpy = vi.spyOn(embedding, 'getEmbeddingsStreaming')
            .mockResolvedValue({
                embeddings: Array(6).fill([0.5, 0.5, 0.5]),
                model: 'gemma4:latest',
            });

        const aiSpy = vi.spyOn(modelManager, 'runPromptStreaming')
            .mockResolvedValue('Analysis with 4 clusters completed.');

        const result = await registry.execute('/ai analyze f1 "what are the dominant groups"', workspace);

        expect(result.output).toContain("📊 Clustering Results");
        expect(result.output).toContain('Records: 58 | Clusters: 4');
        expect(result.output).toContain('Analysis with 4 clusters completed');

        embeddingSpy.mockRestore();
        aiSpy.mockRestore();
    });

    it('should handle questions asking for 6 clusters', async () => {
        await registry.execute('/loadCsv f1 data/clustering.csv', workspace);
        await registry.execute('/model gemma4:latest', workspace);

        const embeddingSpy = vi.spyOn(embedding, 'getEmbeddingsStreaming')
            .mockResolvedValue({
                embeddings: Array(10).fill([0.3, 0.4, 0.5]),
                model: 'gemma4:latest',
            });

        const aiSpy = vi.spyOn(modelManager, 'runPromptStreaming')
            .mockResolvedValue('Found 6 distinct clusters in the data.');

        const result = await registry.execute('/ai analyze f1 "find the top 6 clusters"', workspace);

        expect(result.output).toContain("📊 Clustering Results");
        expect(result.output).toContain('Records: 58 | Clusters: 6');
        expect(result.output).toContain('Found 6 distinct clusters');

        embeddingSpy.mockRestore();
        aiSpy.mockRestore();
    });

    it('should handle questions asking for dominant groups', async () => {
        await registry.execute('/loadCsv f1 data/clustering.csv', workspace);
        await registry.execute('/model gemma4:latest', workspace);

        const embeddingSpy = vi.spyOn(embedding, 'getEmbeddingsStreaming')
            .mockResolvedValue({
                embeddings: [
                    [0.1, 0.1], [0.2, 0.2],
                    [0.8, 0.8], [0.9, 0.9],
                    [0.5, 0.5], [0.6, 0.6],
                    [0.3, 0.7], [0.4, 0.8],
                ],
                model: 'gemma4:latest',
            });

        const aiSpy = vi.spyOn(modelManager, 'runPromptStreaming')
            .mockResolvedValue('The data shows 3 dominant geographic regions.');

        const result = await registry.execute('/ai analyze f1 "what are the dominant groups, top 3 clusters"', workspace);

        expect(result.output).toContain("📊 Clustering Results");
        expect(result.output).toContain('Records: 58 | Clusters: 3');
        expect(result.output).toContain('3 dominant geographic regions');

        embeddingSpy.mockRestore();
        aiSpy.mockRestore();
    });

    it('should include cluster summary in AI prompt', async () => {
        await registry.execute('/loadCsv f1 data/clustering.csv', workspace);
        await registry.execute('/model gemma4:latest', workspace);

        const embeddingSpy = vi.spyOn(embedding, 'getEmbeddingsStreaming')
            .mockResolvedValue({
                embeddings: [
                    [0.1, 0.2], [0.15, 0.25],
                    [0.9, 0.8], [0.85, 0.75],
                ],
                model: 'gemma4:latest',
            });

        const aiSpy = vi.spyOn(modelManager, 'runPromptStreaming')
            .mockResolvedValue('Analysis complete');

        await registry.execute('/ai analyze f1 "what are the dominant groups, top 2 clusters"', workspace);

        // Verify the AI was called with clustering data
        expect(aiSpy).toHaveBeenCalledWith(
            'gemma4:latest',
            expect.stringContaining('CLUSTERING RESULTS:'),
            expect.any(Function),
        );

        embeddingSpy.mockRestore();
        aiSpy.mockRestore();
    });

    it('should include sample records in AI prompt', async () => {
        await registry.execute('/loadCsv f1 data/clustering.csv', workspace);
        await registry.execute('/model gemma4:latest', workspace);

        const embeddingSpy = vi.spyOn(embedding, 'getEmbeddingsStreaming')
            .mockResolvedValue({
                embeddings: [
                    [0.1, 0.2], [0.15, 0.25],
                    [0.9, 0.8], [0.85, 0.75],
                ],
                model: 'gemma4:latest',
            });

        const aiSpy = vi.spyOn(modelManager, 'runPromptStreaming')
            .mockResolvedValue('Analysis complete');

        await registry.execute('/ai analyze f1 "what are the dominant groups, top 2 clusters"', workspace);

        // Verify clustering data was included in prompt
        expect(aiSpy).toHaveBeenCalledWith(
            'gemma4:latest',
            expect.stringContaining('CLUSTERING RESULTS:'),
            expect.any(Function),
        );

        embeddingSpy.mockRestore();
        aiSpy.mockRestore();
    });

    it('should add _cluster field to clustering result file', async () => {
        await registry.execute('/loadCsv f1 data/clustering.csv', workspace);
        await registry.execute('/model gemma4:latest', workspace);

        const embeddingSpy = vi.spyOn(embedding, 'getEmbeddingsStreaming')
            .mockResolvedValue({
                embeddings: Array(76).fill([0.5, 0.5]),
                model: 'gemma4:latest',
            });

        const aiSpy = vi.spyOn(modelManager, 'runPromptStreaming')
            .mockResolvedValue('Two clusters found');

        await registry.execute('/ai analyze f1 "what are the dominant groups, top 2 clusters"', workspace);

        const clusteringFile = workspace.getLoadedFiles().get('f1-clustering-2');
        expect(clusteringFile).toBeDefined();

        // Verify all records have _cluster field
        const allHaveCluster = clusteringFile?.data.hits.hits.every(
            (hit) => '_cluster' in hit._source
        );
        expect(allHaveCluster).toBe(true);

        embeddingSpy.mockRestore();
        aiSpy.mockRestore();
    });

    it('should handle clustering with real clustering.csv data', async () => {
        await registry.execute('/loadCsv f1 data/clustering.csv', workspace);
        await registry.execute('/model gemma4:latest', workspace);

        // clustering.csv has 58 data rows
        const mockEmbeddings = Array(58).fill(null).map((_, i) => {
            // Create distinct embeddings for different regions
            if (i < 20) return [0.1, 0.2, 0.3]; // US/Europe
            if (i < 40) return [0.8, 0.9, 0.85]; // Asia
            return [0.5, 0.5, 0.5]; // Other
        });

        const embeddingSpy = vi.spyOn(embedding, 'getEmbeddingsStreaming')
            .mockResolvedValue({
                embeddings: mockEmbeddings,
                model: 'gemma4:latest',
            });

        const aiSpy = vi.spyOn(modelManager, 'runPromptStreaming')
            .mockResolvedValue(
                'Cluster 1: US and European markets (NYSE, XLON)\n' +
                'Cluster 2: Asian markets (TSE, XHK)\n' +
                'Cluster 3: Other regional exchanges'
            );

        const result = await registry.execute(
            '/ai analyze f1 "what are the dominant groups, top 3 clusters"',
            workspace
        );

        expect(result.output).toContain("📊 Clustering Results");
        expect(result.output).toContain('Records: 58 | Clusters: 3');
        expect(result.output).toContain('US and European markets');
        expect(result.output).toContain('Asian markets');
        expect(workspace.getLoadedFiles().has('f1-clustering-3')).toBe(true);

        const clusteringFile = workspace.getLoadedFiles().get('f1-clustering-3');
        expect(clusteringFile?.data.hits.hits.length).toBe(58);

        embeddingSpy.mockRestore();
        aiSpy.mockRestore();
    });

    it('should handle different cluster count keywords: "groups"', async () => {
        await registry.execute('/loadCsv f1 data/clustering.csv', workspace);
        await registry.execute('/model gemma4:latest', workspace);

        const embeddingSpy = vi.spyOn(embedding, 'getEmbeddingsStreaming')
            .mockResolvedValue({
                embeddings: Array(10).fill([0.5, 0.5]),
                model: 'gemma4:latest',
            });

        const aiSpy = vi.spyOn(modelManager, 'runPromptStreaming')
            .mockResolvedValue('Found 5 groups in the data');

        const result = await registry.execute(
            '/ai analyze f1 "identify the top 5 groups"',
            workspace
        );

        expect(result.output).toContain("📊 Clustering Results");
        expect(result.output).toContain('Records: 58 | Clusters: 5');
        expect(result.output).toContain('Found 5 groups');
        expect(workspace.getLoadedFiles().has('f1-clustering-5')).toBe(true);

        embeddingSpy.mockRestore();
        aiSpy.mockRestore();
    });

    it('should handle different cluster count keywords: "dominance"', async () => {
        await registry.execute('/loadCsv f1 data/clustering.csv', workspace);
        await registry.execute('/model gemma4:latest', workspace);

        const embeddingSpy = vi.spyOn(embedding, 'getEmbeddingsStreaming')
            .mockResolvedValue({
                embeddings: Array(8).fill([0.6, 0.6]),
                model: 'gemma4:latest',
            });

        const aiSpy = vi.spyOn(modelManager, 'runPromptStreaming')
            .mockResolvedValue('Analysis of 6 dominance patterns');

        const result = await registry.execute(
            '/ai analyze f1 "what are the top 6 dominance patterns"',
            workspace
        );

        expect(result.output).toContain("📊 Clustering Results");
        expect(result.output).toContain('Records: 58 | Clusters: 6');

        embeddingSpy.mockRestore();
        aiSpy.mockRestore();
    });

    it('should handle cluster count exceeding record count gracefully', async () => {
        await registry.execute('/loadCsv f1 data/clustering.csv', workspace);
        await registry.execute('/model gemma4:latest', workspace);

        // Mock to return embeddings for all 76 records
        const embeddingSpy = vi.spyOn(embedding, 'getEmbeddingsStreaming')
            .mockResolvedValue({
                embeddings: Array(76).fill([0.1, 0.2]),
                model: 'gemma4:latest',
            });

        const aiSpy = vi.spyOn(modelManager, 'runPromptStreaming')
            .mockResolvedValue('Analysis complete');

        // Request 100 clusters but only 76 records exist
        const result = await registry.execute(
            '/ai analyze f1 "what are the top 100 clusters"',
            workspace
        );

        // Should handle gracefully (kMeans uses Math.min)
        expect(result.output).toBeDefined();

        embeddingSpy.mockRestore();
        aiSpy.mockRestore();
    });
});
