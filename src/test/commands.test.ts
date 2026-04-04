import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as modelManager from '../ai/modelManager.js';
import { CommandRegistry } from '../commands/commandRegistry.js';
import { MemoryWorkspace } from '../core/runner.js';
import * as fs from 'fs';
import { HelpCommand, LoadJsonCommand, LoadCsvCommand, DiffCommand, SqlCommand, ModelCommand, AiCommand, SchemaCommand, SaveSchemaCommand, AnalyzeCommand } from '../commands/index.js';

describe('Slash Commands', () => {
    let workspace: MemoryWorkspace;
    let registry: CommandRegistry;

    beforeEach(() => {
        workspace = new MemoryWorkspace();
        registry = new CommandRegistry();
        registry.register(HelpCommand);
        registry.register(ModelCommand);
        registry.register(LoadJsonCommand);
        registry.register(LoadCsvCommand);
        registry.register(AnalyzeCommand);
        registry.register(DiffCommand);
        registry.register(SqlCommand);
        registry.register(AiCommand);
        registry.register(SchemaCommand);
        registry.register(SaveSchemaCommand);
    });

    it('HelpCommand should list registered commands', async () => {
        const result = await registry.execute('/help', workspace);
        expect(result.output).toContain('/help');
        expect(result.output).toContain('/model');
        expect(result.output).toContain('/load');
        expect(result.output).toContain('/analyze');
        expect(result.output).toContain('/diff');
        expect(result.output).toContain('/schema');
        expect(result.output).toContain('/saveSchema');
    });

    it('AiAnalyzeCommand should require a file id', async () => {
        const result = await registry.execute('/ai', workspace);
        expect(result.output).toBe('Usage: /ai <id>');
    });

    it('AiAnalyzeCommand should require a configured model', async () => {
        await registry.execute('/loadJson f1 data/elasticSearchResult.json', workspace);
        const result = await registry.execute('/ai f1', workspace);
        expect(result.output).toBe('Set a model first with /model <name>');
    });

    it('AiAnalyzeCommand should support conversational prompts in the same session', async () => {
        await registry.execute('/model gemma4:latest', workspace);

        const spy = vi.spyOn(modelManager, 'runPromptStreaming')
            .mockResolvedValueOnce('It looks like f2 is richer than f1.')
            .mockResolvedValueOnce('The pattern is deeper nested structure and broader coverage.');

        const first = await registry.execute('/ai what is the high level difference between files', workspace);
        expect(first.output).toContain('AI chat:');
        expect(first.output).toContain('It looks like f2 is richer than f1.');

        const second = await registry.execute('/ai what pattern do you see', workspace);
        expect(second.output).toContain('The pattern is deeper nested structure');
        expect(spy).toHaveBeenNthCalledWith(
            2,
            'gemma4:latest',
            expect.stringContaining('Conversation so far:\nUser: what is the high level difference between files'),
            expect.any(Function),
        );
        spy.mockRestore();
    });

    it('AiAnalyzeCommand should run compare analysis across two files', async () => {
        await registry.execute('/loadJson f1 data/elasticSearchResult.json', workspace);
        await registry.execute('/loadJson f2 data/elasticSearchResult_2.json', workspace);
        workspace.updateFile('f1', { keyField: 'name' });
        workspace.updateFile('f2', { keyField: 'name' });
        await registry.execute('/model gemma4:latest', workspace);

        const spy = vi.spyOn(modelManager, 'runPromptStreaming').mockResolvedValue(
            'f2 adds richer nested specs and reviews, while names are expanded and product coverage is broader.',
        );

        const result = await registry.execute(
            '/ai compare f1 f2 what f1 missing in f2 and what is the pattern',
            workspace,
        );
        expect(result.output).toContain("AI compare analysis for 'f1' vs 'f2':");
        expect(result.output).toContain('f2 adds richer nested specs and reviews');
        expect(spy).toHaveBeenCalledWith(
            'gemma4:latest',
            expect.stringContaining('Question: what f1 missing in f2 and what is the pattern'),
            expect.any(Function),
        );
        expect(spy).toHaveBeenCalledWith(
            'gemma4:latest',
            expect.stringContaining('Definition: "missing" means a file 1 record whose comparison key does not appear in file 2.'),
            expect.any(Function),
        );
        expect(spy).toHaveBeenCalledWith(
            'gemma4:latest',
            expect.stringContaining('Definition: "pattern" means the majority/common trend across the compared rows'),
            expect.any(Function),
        );
        spy.mockRestore();
    });

    it('AiAnalyzeCommand should return schema from the model', async () => {
        await registry.execute('/loadJson f1 data/elasticSearchResult.json', workspace);
        await registry.execute('/model gemma4:latest', workspace);

        const spy = vi.spyOn(modelManager, 'inferSchemaForDataStreaming').mockResolvedValue({
            type: 'object',
            properties: {
                name: { type: 'string' },
            },
            required: ['name'],
        });

        const result = await registry.execute('/ai f1', workspace);
        expect(result.output).toContain("Schema for 'f1' via gemma4:latest saved as 'f1-schema':");
        expect(result.output).toContain('"type": "object"');
        expect(result.output).toContain('"name"');
        expect(result.action).toBe('VIEW_CHANGE');
        expect(workspace.viewMode).toBe('preview');
        expect(workspace.getLoadedFiles().has('f1-schema')).toBe(true);
        const sourceFile = workspace.getLoadedFiles().get('f1');
        expect(sourceFile?.schema).toBeDefined();
        expect(sourceFile?.subSchemas).toBeDefined();
        spy.mockRestore();
    });

    it('SchemaCommand should require schema on the file', async () => {
        await registry.execute('/loadJson f1 data/elasticSearchResult.json', workspace);
        const result = await registry.execute('/schema f1', workspace);
        expect(result.output).toBe("File 'f1' has no schema. Run /ai f1 first");
    });

    it('SchemaCommand should switch to schema view', async () => {
        await registry.execute('/loadJson f1 data/elasticSearchResult.json', workspace);
        workspace.updateFile('f1', {
            schema: {
                type: 'object',
                properties: {
                    name: { type: 'string' },
                    details: {
                        type: 'object',
                        properties: {
                            price: { type: 'number' },
                        },
                    },
                },
            },
            subSchemas: {
                name: { type: 'string' },
                details: { type: 'object' },
                'details.price': { type: 'number' },
            },
            selectedSubSchemaPaths: ['name'],
        });

        const result = await registry.execute('/schema f1', workspace);
        expect(result.action).toBe('VIEW_CHANGE');
        expect(workspace.viewMode).toBe('schema');
    });

    it('AiAnalyzeCommand should create selected-only schema output from saved selections', async () => {
        await registry.execute('/loadJson f1 data/elasticSearchResult.json', workspace);
        workspace.updateFile('f1', {
            schema: {
                type: 'object',
                properties: {
                    name: { type: 'string' },
                    details: { type: 'object' },
                },
            },
            subSchemas: {
                name: { type: 'string' },
                details: { type: 'object' },
            },
            selectedSubSchemaPaths: ['name'],
        });

        const result = await registry.execute('/ai f1 --selected-only', workspace);
        expect(result.output).toContain("Selected schema for 'f1' saved as 'f1-schema-selected':");
        expect(workspace.getLoadedFiles().has('f1-schema-selected')).toBe(true);
        expect(workspace.viewMode).toBe('preview');
    });

    it('AiAnalyzeCommand should pass extra instructions to the model helper', async () => {
        await registry.execute('/loadJson f1 data/elasticSearchResult.json', workspace);
        await registry.execute('/model gemma4:latest', workspace);

        const spy = vi.spyOn(modelManager, 'inferSchemaForDataStreaming').mockResolvedValue({
            type: 'object',
        });

        await registry.execute('/ai f1 only include top-level fields', workspace);
        expect(spy).toHaveBeenCalledWith(
            'gemma4:latest',
            'f1',
            expect.any(Object),
            'json',
            'only include top-level fields',
            expect.any(Function),
        );
        spy.mockRestore();
    });

    it('AiAnalyzeCommand should handle analyze subcommand with clustering', async () => {
        await registry.execute('/loadCsv f1 data/clustering.csv', workspace);
        await registry.execute('/model gemma4:latest', workspace);

        // Mock embedding and AI response
        const embeddingSpy = vi.spyOn(await import('../ai/embedding.js'), 'getEmbeddingsStreaming')
            .mockResolvedValue({
                embeddings: [
                    [0.1, 0.2], [0.15, 0.25], [0.12, 0.22],
                    [0.9, 0.8], [0.85, 0.75], [0.88, 0.82],
                ],
                model: 'gemma4:latest',
            });

        const aiSpy = vi.spyOn(modelManager, 'runPromptStreaming')
            .mockResolvedValue('Cluster 1 represents US/European equities. Cluster 2 represents Asian equities.');

        const result = await registry.execute('/ai analyze f1 "what are the dominant groups, top 2 clusters"', workspace);

        expect(result.output).toContain("📊 Clustering Results");
        expect(result.output).toContain('Records: 58 | Clusters: 2');
        expect(result.output).toContain('Cluster 1 represents US/European equities');
        expect(workspace.getLoadedFiles().has('f1-clustering-2')).toBe(true);

        embeddingSpy.mockRestore();
        aiSpy.mockRestore();
    });

    it('SaveSchemaCommand should save selected-only schema to disk', async () => {
        await registry.execute('/loadJson f1 data/elasticSearchResult.json', workspace);
        workspace.updateFile('f1', {
            schema: {
                type: 'object',
                properties: {
                    name: { type: 'string' },
                    details: { type: 'object' },
                },
            },
            subSchemas: {
                name: { type: 'string' },
                details: { type: 'object' },
            },
            selectedSubSchemaPaths: ['details'],
        });

        const outputPath = '/tmp/tui-slash-selected-schema.json';
        const result = await registry.execute(`/saveSchema f1 ${outputPath} --selected-only`, workspace);
        expect(result.output).toBe(`Saved selected schema for f1 to ${outputPath}`);
        const saved = JSON.parse(fs.readFileSync(outputPath, 'utf-8')) as Record<string, unknown>;
        expect(saved.title).toBe('SelectedSubSchemas');
        expect(saved.properties).toHaveProperty('details');
    });

    it('ModelCommand should show the current model', async () => {
        const spy = vi.spyOn(modelManager, 'getAvailableModels').mockReturnValue(['gemma4:latest', 'qwen3.5:latest']);
        const result = await registry.execute('/model', workspace);
        expect(result.output).toContain('Current model: default');
        expect(result.output).toContain('gemma4:latest');
        spy.mockRestore();
    });

    it('ModelCommand should set the current model', async () => {
        const result = await registry.execute('/model gpt-5.4', workspace);
        expect(result.output).toBe('Model set to: gpt-5.4');
        expect(workspace.getModel()).toBe('gpt-5.4');
    });

    it('ModelCommand should suggest matching Ollama models', () => {
        const spy = vi.spyOn(modelManager, 'getAvailableModels').mockReturnValue(['gemma4:latest', 'qwen3.5:latest']);
        const suggestions = ModelCommand.suggestArgs?.('gem', workspace) || [];
        expect(suggestions).toEqual(['gemma4:latest']);
        spy.mockRestore();
    });

    it('LoadJsonCommand should handle missing arguments', async () => {
        const result = await registry.execute('/loadJson myid', workspace);
        expect(result.output).toBe('Usage: /loadJson <id> <filepath>');
    });

    it('LoadJsonCommand should load a JSON file', async () => {
        const result = await registry.execute('/loadJson f1 data/elasticSearchResult.json', workspace);
        expect(result.output).toContain("Loaded data/elasticSearchResult.json as 'f1'");
        expect(workspace.getLoadedFiles().has('f1')).toBe(true);
    });

    it('AnalyzeCommand should load both CSVs and open the diff result', async () => {
        const result = await registry.execute(
            '/analyze SecurityPricingRequest.csv SecurityPricingResponse.csv',
            workspace,
        );
        expect(result.output).toContain("Loaded SecurityPricingRequest.csv as 'request'");
        expect(result.output).toContain("Loaded SecurityPricingResponse.csv as 'response'");
        expect(result.output).toContain("Created 'request-diff-response'");
        expect(result.action).toBe('VIEW_CHANGE');
        expect(workspace.getLoadedFiles().has('request')).toBe(true);
        expect(workspace.getLoadedFiles().has('response')).toBe(true);
        expect(workspace.getLoadedFiles().has('request-diff-response')).toBe(true);
        expect(workspace.viewMode).toBe('preview');
    });

    it('AnalyzeCommand should apply an explicit key field before diffing', async () => {
        const result = await registry.execute(
            '/analyze SecurityPricingRequest.csv SecurityPricingResponse.csv isin',
            workspace,
        );
        expect(result.output).toContain("Set key field 'isin' on 'request' and 'response'");
        expect(workspace.getLoadedFiles().get('request')?.keyField).toBe('isin');
        expect(workspace.getLoadedFiles().get('response')?.keyField).toBe('isin');
        expect(workspace.getLoadedFiles().has('request-diff-response')).toBe(true);
    });

    it('DiffCommand should diff two loaded files', async () => {
        await registry.execute('/loadJson f1 data/elasticSearchResult.json', workspace);
        await registry.execute('/loadJson f2 data/elasticSearchResult_2.json', workspace);
        
        // Set keyField since these files don't have isin/currency/etc.
        workspace.updateFile('f1', { keyField: 'name' });
        workspace.updateFile('f2', { keyField: 'name' });
        
        const result = await registry.execute('/diff f1 f2', workspace);
        expect(result.output).toContain("Created 'f1-diff-f2'");
        expect(workspace.getLoadedFiles().has('f1-diff-f2')).toBe(true);
    });

    it('SqlCommand should execute SQL and switch to preview', async () => {
        await registry.execute('/loadJson f1 data/elasticSearchResult.json', workspace);
        const result = await registry.execute('/sql SELECT name, price FROM f1', workspace);
        
        expect(result.output).toContain("Created 'sql_result_");
        expect(result.action).toBe('VIEW_CHANGE');
        expect(workspace.viewMode).toBe('preview');
        
        // Check if the result file exists in workspace
        const files = workspace.getLoadedFiles();
        const sqlResultId = Array.from(files.keys()).find(id => id.startsWith('sql_result_'));
        expect(sqlResultId).toBeDefined();
    });
});
