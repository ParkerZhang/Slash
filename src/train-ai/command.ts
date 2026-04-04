import { Workspace, CommandResult, AiPromptState } from '../core/types.js';
import { SlashCommand, CommandRegistry } from '../commands/commandRegistry.js';
import { runWorkflow, runWorkflowWithFiles, loadState, listWorkflows } from './workflow.js';
import { listCandidateModels, MODEL_REGISTRY, getModelProfile } from './modelRegistry.js';
import { getAvailableModels, runPromptStreaming } from '../ai/modelManager.js';
import { getLastPrompt, getLastResult, getPromptHistory, getPromptHistoryEntry, clearPromptHistory } from './promptStore.js';
import {
    kMeans,
    buildMultiDimensionalFeatures,
    evaluateClusters,
    buildClusterProfile,
    formatMentalModelVisualization,
    formatEnhancedClusterSummary,
} from '../core/clustering.js';
import { getEmbeddingsStreaming } from '../ai/embedding.js';
import { loadAiProviderConfig, saveAiProviderConfig, resetAiProviderConfig, AiProviderConfig, PROVIDER_INFO, AiProviderType } from './aiProvider.js';
import * as fs from 'fs';
import * as path from 'path';

const TRAIN_AI_DIR = path.join(process.cwd(), '.train-ai');

function formatDuration(start: string, end: string): string {
    const startTime = new Date(start).getTime();
    const endTime = new Date(end).getTime();
    const seconds = Math.floor((endTime - startTime) / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
}

function formatStatus(status: string): string {
    const symbols: Record<string, string> = {
        'pending': '○',
        'running': '◐',
        'completed': '✓',
        'failed': '✗',
    };
    return symbols[status] || '?';
}

function visualizeWorkflowState(state: any): string {
    const lines: string[] = [];

    lines.push('═══════════════════════════════════════════════════════════');
    lines.push(`🧠 TRAIN-AI WORKFLOW: ${state.workflowId}`);
    lines.push('═══════════════════════════════════════════════════════════');
    lines.push('');
    lines.push(`Model: ${state.modelProfile}`);
    lines.push(`Seed: ${state.seed}`);
    lines.push(`Status: ${state.status.toUpperCase()}`);
    lines.push(`Progress: Step ${state.currentStep}/${state.totalSteps}`);
    if (state.error) lines.push(`Error: ${state.error}`);
    lines.push('');

    lines.push('📊 WORKFLOW STEPS:');
    lines.push('─'.repeat(100));
    lines.push(
        'Step'.padEnd(6) +
        'Status'.padEnd(8) +
        'Name'.padEnd(25) +
        'Input'.padEnd(35) +
        'Expected Result'.padEnd(45)
    );
    lines.push('─'.repeat(100));

    for (const step of state.steps) {
        const statusSymbol = formatStatus(step.status);
        lines.push(
            `${step.stepNumber}`.padEnd(6) +
            `${statusSymbol} ${step.status === 'running' ? '...' : '  '}`.padEnd(8) +
            `${step.name}`.padEnd(25) +
            `${step.input || ''}`.padEnd(35) +
            `${step.expectedResult || ''}`.padEnd(45)
        );

        if (step.status === 'running' || step.status === 'completed') {
            if (step.testingPrompt) {
                lines.push(`       Prompt: ${step.testingPrompt}`);
            }
            if (step.result) {
                lines.push(`       Result: ${step.result.slice(0, 80)}${step.result.length > 80 ? '...' : ''}`);
            }
            if (step.difference) {
                lines.push(`       Difference: ${step.difference}`);
            }
            if (step.startTime && step.endTime) {
                lines.push(`       Duration: ${formatDuration(step.startTime, step.endTime)}`);
            }
        }
    }

    lines.push('─'.repeat(100));
    lines.push('');

    // Show output summary if available
    if (state.output) {
        const output = state.output;
        lines.push('📈 MISSING ANALYSIS SUMMARY:');
        lines.push(`  Requested: ${output.summary.requested}`);
        lines.push(`  Received: ${output.summary.received}`);
        lines.push(`  MISSING: ${output.summary.missing} (${(output.summary.missingRate * 100).toFixed(1)}%)`);
        lines.push('');

        lines.push('  MISSING by Geography:');
        for (const [geo, count] of Object.entries(output.missingByGeography)) {
            lines.push(`    ${geo}: ${count}`);
        }
        lines.push('');

        lines.push('  MISSING by Sector:');
        for (const [sector, count] of Object.entries(output.missingBySector)) {
            lines.push(`    ${sector}: ${count}`);
        }
        lines.push('');

        lines.push('  Clusters:');
        for (const cluster of output.clusters) {
            lines.push(`    Cluster ${cluster.id} (${cluster.size} records): ${cluster.interpretation}`);
            // Show sample members for clarity
            if (cluster.members && cluster.members.length > 0) {
                const samples = cluster.members.slice(0, 3);
                const sampleNames = samples.map((m: any) => m.name).join(', ');
                lines.push(`      Examples: ${sampleNames}${cluster.members.length > 3 ? '...' : ''}`);
            }
        }
        lines.push('');

        lines.push('  Quality Metrics:');
        lines.push(`    Geography Purity: ${(output.qualityMetrics.geographyPurity * 100).toFixed(1)}%`);
        lines.push(`    Sector Purity: ${(output.qualityMetrics.sectorPurity * 100).toFixed(1)}%`);
        lines.push(`    Balance Score: ${(output.qualityMetrics.balanceScore * 100).toFixed(1)}%`);
        lines.push('');

        if (output.insights.length > 0) {
            lines.push('  Insights:');
            for (const insight of output.insights) {
                lines.push(`    ✓ ${insight}`);
            }
            lines.push('');
        }

        if (output.aiAnalysis) {
            lines.push('🤖 AI ANALYSIS:');
            lines.push('─'.repeat(80));
            lines.push(output.aiAnalysis);
            lines.push('─'.repeat(80));
            lines.push('');
        }
    }

    lines.push(`Started: ${new Date(state.startTime).toLocaleString()}`);
    lines.push(`Last Updated: ${new Date(state.lastUpdated).toLocaleString()}`);
    lines.push('');
    lines.push(`Workflow state saved to: .train-ai/${state.workflowId}.json`);
    lines.push(`Resume with: /ai train resume ${state.workflowId}`);

    return lines.join('\n');
}

function visualizeCandidateModels(): string {
    const candidates = listCandidateModels();
    const availableModels = getAvailableModels();
    const lines: string[] = [];

    lines.push('═══════════════════════════════════════════════════════════');
    lines.push('🤖 CANDIDATE MODELS');
    lines.push('═══════════════════════════════════════════════════════════');
    lines.push('');
    lines.push(
        'Model'.padEnd(35) +
        'Status'.padEnd(15) +
        'Embedding'.padEnd(12) +
        'Available'.padEnd(12) +
        'Notes'
    );
    lines.push('─'.repeat(100));

    for (const candidate of candidates) {
        const isAvailable = availableModels.includes(candidate.model) || 
                           MODEL_REGISTRY[candidate.model]?.type === 'embedding';
        lines.push(
            `${candidate.model}`.padEnd(35) +
            `${candidate.status}`.padEnd(15) +
            `${candidate.embedding ? '✓' : '✗'}`.padEnd(12) +
            `${isAvailable ? '✓' : '✗ (pull with: ollama pull ' + candidate.model + ')'}`.padEnd(12) +
            `${candidate.notes}`
        );
    }

    lines.push('');
    lines.push('Usage: /ai train [model] [seed]');
    lines.push('Example: /ai train qwen2.5:1.5b 42');
    lines.push('');
    
    if (availableModels.length > 0) {
        lines.push('Available models in Ollama:');
        for (const model of availableModels) {
            const inRegistry = MODEL_REGISTRY[model] ? ' (in registry)' : '';
            lines.push(`  - ${model}${inRegistry}`);
        }
    } else {
        lines.push('No models found in Ollama. Run: ollama pull qwen2.5:1.5b');
    }

    return lines.join('\n');
}

async function executeAiTrain(
    args: string,
    workspace: Workspace,
    onProgress?: (step: number, msg: string) => void,
): Promise<CommandResult> {
    const parts = args.trim().split(/\s+/).filter(Boolean);

    // Handle empty args - show candidate models
    if (parts.length === 0 || (parts.length === 1 && parts[0] === '')) {
        return { output: visualizeCandidateModels() };
    }

    // Handle subcommands
    if (parts[0] === 'list' || parts[0] === 'models') {
        return { output: visualizeCandidateModels() };
    }

    if (parts[0] === 'resume') {
        if (parts.length < 2) {
            const workflows = await listWorkflows();
            if (workflows.length === 0) {
                return { output: 'No workflows found. Start a new one with: /ai train <model> [seed]' };
            }

            const lines = ['Recent workflows:'];
            for (const wf of workflows.slice(0, 5)) {
                lines.push(`  ${wf.workflowId} - ${wf.modelProfile} - ${wf.status} (Step ${wf.currentStep}/${wf.totalSteps})`);
            }
            return { output: lines.join('\n') };
        }

        const workflowId = parts[1];
        const state = await loadState(workflowId);
        if (!state) {
            return { output: `Workflow ${workflowId} not found` };
        }

        workspace.setCommandOutput?.([`Resuming workflow ${workflowId}...`]);

        const result = await runWorkflow(state.modelProfile, state.seed, {
            onProgress: (step, msg) => {
                workspace.setCommandOutput?.([`Step ${step}: ${msg}`]);
            },
            onStateUpdate: (state) => {
                workspace.setCommandOutput?.([`Workflow updated: ${state.workflowId}`]);
            },
        }, workflowId);

        return {
            output: visualizeWorkflowState(result),
            action: 'VIEW_CHANGE',
        };
    }

    if (parts[0] === 'status') {
        if (parts.length < 2) {
            const workflows = await listWorkflows();
            const active = workflows.find(w => w.status === 'running' || w.status === 'paused');
            if (!active) return { output: 'No active workflows' };
            return { output: visualizeWorkflowState(active) };
        }

        const workflowId = parts[1];
        const state = await loadState(workflowId);
        if (!state) return { output: `Workflow ${workflowId} not found` };
        return { output: visualizeWorkflowState(state) };
    }

    // Check if using existing files: /ai train <model> <requestFileId> <responseFileId>
    if (parts.length >= 3) {
        const modelName = parts[0];
        const seed = parts[1] && !isNaN(Number(parts[1])) ? parseInt(parts[1], 10) : 42;
        
        // Check if parts[1] is a seed number or a file ID
        const fileStartIndex = parts[1] && !isNaN(Number(parts[1])) ? 2 : 1;
        
        if (parts.length >= fileStartIndex + 2) {
            const requestFileId = parts[fileStartIndex];
            const responseFileId = parts[fileStartIndex + 1];

            const loadedFiles = workspace.getLoadedFiles();
            const requestFile = loadedFiles.get(requestFileId);
            const responseFile = loadedFiles.get(responseFileId);

            if (!requestFile) return { output: `File '${requestFileId}' not loaded. Use /loadCsv first.` };
            if (!responseFile) return { output: `File '${responseFileId}' not loaded. Use /loadCsv first.` };

            if (!MODEL_REGISTRY[modelName]) {
                const availableInOllama = getAvailableModels();
                if (availableInOllama.includes(modelName)) {
                    MODEL_REGISTRY[modelName] = {
                        name: modelName,
                        type: 'chat',
                        dimensions: 2048,
                        contextWindow: 32768,
                        temperature: 0.1,
                        maxTokens: 500,
                    };
                } else {
                    return { output: `Model '${modelName}' not available. Pull with: ollama pull ${modelName}` };
                }
            }

            workspace.setCommandOutput?.([
                `Starting TRAIN-AI workflow with existing files...`,
                `Model: ${modelName}`,
                `Seed: ${seed}`,
                `Request: ${requestFileId} (${requestFile.data.hits.hits.length} records)`,
                `Response: ${responseFileId} (${responseFile.data.hits.hits.length} records)`,
            ]);

            try {
                const result = await runWorkflowWithFiles(modelName, seed, requestFile, responseFile, workspace, {
                    onProgress: (step, msg) => {
                        workspace.setCommandOutput?.([`Step ${step}/6: ${msg}`]);
                    },
                    onStateUpdate: (state) => {
                        workspace.setCommandOutput?.([`Workflow ${state.workflowId}: Step ${state.currentStep}/6 - ${state.steps[state.currentStep - 1]?.status}`]);
                    },
                });

                return {
                    output: visualizeWorkflowState(result),
                    action: 'VIEW_CHANGE',
                };
            } catch (error) {
                return {
                    output: `Train-AI Error: ${error instanceof Error ? error.message : String(error)}`,
                };
            }
        }
    }

    // Default: run new workflow with generated data (Step 1 will create/overwrite files)
    const modelName = parts[0] || 'qwen2.5:1.5b';
    const seed = parts[1] ? parseInt(parts[1], 10) : 42;

    if (!MODEL_REGISTRY[modelName]) {
        const availableInOllama = getAvailableModels();
        const registryModels = Object.keys(MODEL_REGISTRY);
        
        // Check if the model is available in Ollama but just not in our registry
        if (availableInOllama.includes(modelName)) {
            workspace.setCommandOutput?.([
                `Model '${modelName}' is available in Ollama but not in the train-ai registry.`,
                `Using it with default settings...`,
            ]);
            // Add it to registry temporarily with default settings
            MODEL_REGISTRY[modelName] = {
                name: modelName,
                type: 'chat',
                dimensions: 2048,
                contextWindow: 32768,
                temperature: 0.1,
                maxTokens: 500,
            };
        } else {
            const notInstalled = registryModels.filter(m => 
                !availableInOllama.includes(m) && MODEL_REGISTRY[m]?.type === 'chat'
            );
            const installed = registryModels.filter(m => 
                availableInOllama.includes(m)
            );
            
            let errorMsg = `Model '${modelName}' not available in Ollama.`;
            if (installed.length > 0) {
                errorMsg += `\nAvailable: ${installed.join(', ')}`;
            }
            if (notInstalled.length > 0) {
                errorMsg += `\nNot installed (pull with: ollama pull <model>): ${notInstalled.join(', ')}`;
            }
            return { output: errorMsg };
        }
    }

    workspace.setCommandOutput?.([
        `Starting TRAIN-AI workflow...`,
        `Model: ${modelName}`,
        `Seed: ${seed}`,
    ]);

    try {
        const result = await runWorkflow(modelName, seed, {
            onProgress: (step, msg) => {
                workspace.setCommandOutput?.([
                    `Step ${step}/6: ${msg}`,
                ]);
            },
            onStateUpdate: (state) => {
                workspace.setCommandOutput?.([
                    `Workflow ${state.workflowId}: Step ${state.currentStep}/6 - ${state.steps[state.currentStep - 1]?.status}`,
                ]);
            },
        }, undefined, workspace);

        return {
            output: visualizeWorkflowState(result),
            action: 'VIEW_CHANGE',
        };
    } catch (error) {
        return {
            output: `Train-AI Error: ${error instanceof Error ? error.message : String(error)}`,
        };
    }
}

export const AiTrainCommand: SlashCommand = {
    name: '/ai',
    description: 'AI training workflow: /ai train [model] [seed] | /ai train list | /ai train resume [id]',
    execute: async (args, workspace) => {
        const trimmedArgs = args.trim();

        // Detect "train" subcommand
        if (trimmedArgs.toLowerCase().startsWith('train')) {
            return await executeAiTrain(trimmedArgs.slice(5).trim(), workspace);
        }

        // Fall through to existing /ai command
        // This will be handled by AiAnalyzeCommand
        return { output: 'FALLBACK_TO_EXISTING_AI_COMMAND' };
    },
    suggestArgs: (_input, workspace) => {
        const models = Object.keys(MODEL_REGISTRY);
        return ['train', ...models];
    },
};

export { executeAiTrain, visualizeWorkflowState, visualizeCandidateModels };

export const ShowPromptCommand: SlashCommand = {
    name: '/showPrompt',
    description: 'Show last AI prompt and result (Usage: /showPrompt [full|history|<index>|clear])',
    execute: async (args) => {
        const parts = args.trim().split(/\s+/).filter(Boolean);
        const subcommand = parts[0]?.toLowerCase();

        // Show history
        if (subcommand === 'history' || subcommand === 'list') {
            const history = getPromptHistory();
            if (history.length === 0) {
                return { output: 'No prompt history. Run /ai train first.' };
            }
            const lines = ['📜 Prompt History:'];
            lines.push('─'.repeat(80));
            for (let i = 0; i < history.length; i++) {
                const entry = history[i];
                lines.push(`[${i}] ${entry.purpose} (${entry.model}) - ${new Date(entry.timestamp).toLocaleTimeString()}`);
                lines.push(`    Result: ${entry.result.slice(0, 60)}...`);
            }
            lines.push('─'.repeat(80));
            lines.push('View with: /showPrompt <index>');
            return { output: lines.join('\n') };
        }

        // Clear history
        if (subcommand === 'clear') {
            clearPromptHistory();
            return { output: 'Prompt history cleared.' };
        }

        // Show specific history entry by index
        if (subcommand && !isNaN(Number(subcommand))) {
            const index = parseInt(subcommand, 10);
            const entry = getPromptHistoryEntry(index);
            if (!entry) {
                return { output: `Entry ${index} not found. Use /showPrompt history to list.` };
            }
            const lines: string[] = [];
            lines.push(`═══════════════════════════════════════════════════════════`);
            lines.push(`📝 Prompt Entry #${index}`);
            lines.push(`═══════════════════════════════════════════════════════════`);
            lines.push('');
            lines.push(`Model: ${entry.model}`);
            lines.push(`Purpose: ${entry.purpose}`);
            lines.push(`Time: ${new Date(entry.timestamp).toLocaleString()}`);
            lines.push('');
            lines.push('┌─ PROMPT ─'.padEnd(60) + '┐');
            lines.push(entry.prompt);
            lines.push('└────────────────────────────────────────────────────────'.padEnd(60));
            lines.push('');
            lines.push('┌─ RESULT ─'.padEnd(60) + '┐');
            lines.push(entry.result);
            lines.push('└────────────────────────────────────────────────────────'.padEnd(60));
            return { output: lines.join('\n') };
        }

        // Show last prompt and result (default)
        const lastPrompt = getLastPrompt();
        const lastResult = getLastResult();

        if (!lastPrompt && !lastResult) {
            return { output: 'No prompt or result yet. Run /ai train first.' };
        }

        const lines: string[] = [];
        lines.push('═══════════════════════════════════════════════════════════');
        lines.push('📝 LAST AI PROMPT & RESULT');
        lines.push('═══════════════════════════════════════════════════════════');
        lines.push('');

        if (lastPrompt) {
            lines.push(`Title: ${lastPrompt.title}`);
            lines.push('');
            lines.push('┌─ FULL PROMPT ─'.padEnd(60) + '┐');
            const promptText = lastPrompt.fullPrompt || lastPrompt.user;
            // Show first 100 lines of prompt to avoid overwhelming output
            const promptLines = promptText.split('\n');
            if (promptLines.length > 100) {
                lines.push(promptLines.slice(0, 100).join('\n'));
                lines.push(`\n... (${promptLines.length - 100} more lines, use /showPrompt full to see all)`);
            } else {
                lines.push(promptText);
            }
            lines.push('└────────────────────────────────────────────────────────'.padEnd(60));
            lines.push('');
        }

        if (lastResult) {
            lines.push('┌─ AI RESULT ─'.padEnd(60) + '┐');
            lines.push(lastResult);
            lines.push('└────────────────────────────────────────────────────────'.padEnd(60));
        }

        lines.push('');
        lines.push('Commands:');
        lines.push('  /showPrompt full    - Show full prompt (no truncation)');
        lines.push('  /showPrompt history - List all prompt history');
        lines.push('  /showPrompt <index> - View specific history entry');
        lines.push('  /showPrompt clear   - Clear history');

        return { output: lines.join('\n') };
    },
    suggestArgs: () => ['full', 'history', 'clear'],
};

export async function executeAiDebug(
    args: string,
    workspace: Workspace,
    registry: CommandRegistry,
): Promise<CommandResult> {
    const debugArgs = args.trim().slice(6).trim().split(/\s+/).filter(Boolean);
    const subcommand = debugArgs[0]?.toLowerCase();

    if (subcommand === 'prompt') {
        const lastPrompt = getLastPrompt();
        const lastResult = getLastResult();

        if (!lastPrompt && !lastResult) {
            return { output: 'No prompt or result yet. Run /ai train first.' };
        }

        const lines: string[] = [];
        lines.push('🐛 DEBUG: Last AI Prompt & Result');
        lines.push('═'.repeat(80));

        if (lastPrompt) {
            lines.push(`Title: ${lastPrompt.title}`);
            lines.push('');
            lines.push('─── FULL PROMPT ───');
            lines.push(lastPrompt.fullPrompt || lastPrompt.user);
            lines.push('');
        }

        if (lastResult) {
            lines.push('─── AI RESULT ───');
            lines.push(lastResult);
        }

        return { output: lines.join('\n') };
    }

    if (subcommand === 'train') {
        if (debugArgs.length < 3) {
            return { output: 'Usage: /ai debug train <model> <fileId>\nExample: /ai debug train qwen3.5:0.8b request' };
        }

        const modelName = debugArgs[1];
        const fileId = debugArgs[2];
        const loadedFiles = workspace.getLoadedFiles();
        const file = loadedFiles.get(fileId);

        if (!file) return { output: `File '${fileId}' not loaded.` };

        const lines: string[] = [];
        lines.push('🐛 DEBUG: Train-AI Workflow Step-by-Step');
        lines.push('═'.repeat(80));
        lines.push(`File: ${fileId} (${file.name})`);
        lines.push(`Records: ${file.data.hits.hits.length}`);
        lines.push(`Format: ${file.fileFormat}`);
        lines.push('');

        const records = file.data.hits.hits.map((hit) => hit._source);

        // Step 1: Show records
        lines.push('Step 1: LOADED RECORDS');
        lines.push('─'.repeat(60));
        for (let i = 0; i < Math.min(records.length, 5); i++) {
            lines.push(`[${i}] ${JSON.stringify(records[i])}`);
        }
        if (records.length > 5) lines.push(`... (${records.length - 5} more)`);
        lines.push('');

        // Step 2: Simulate embeddings
        lines.push('Step 2: EMBEDDING SIMULATION');
        lines.push('─'.repeat(60));
        const mockEmbeddings = records.map((_, i) => {
            const vec = Array(10).fill(0).map((_, j) => Math.sin(i * (j + 1)) * 0.5);
            return vec;
        });
        lines.push(`Generated ${mockEmbeddings.length} embeddings (10-dim mock)`);
        lines.push('');

        // Step 3: Clustering
        lines.push('Step 3: CLUSTERING');
        lines.push('─'.repeat(60));
        const k = Math.min(4, records.length);
        const clusteringResult = kMeans(mockEmbeddings, k);

        for (let i = 0; i < clusteringResult.clusters.length; i++) {
            const cluster = clusteringResult.clusters[i];
            lines.push(`Cluster ${i + 1} (${cluster.members.length} records):`);
            for (const idx of cluster.members.slice(0, 3)) {
                const rec = records[idx];
                lines.push(`  - ${rec.name || rec.isin} (${rec.country || 'N/A'})`);
            }
            if (cluster.members.length > 3) lines.push(`  ... and ${cluster.members.length - 3} more`);
        }
        lines.push('');

        // Step 4: Evaluation
        lines.push('Step 4: CLUSTER EVALUATION');
        lines.push('─'.repeat(60));
        const evaluation = evaluateClusters(clusteringResult.clusters, records);
        lines.push(`Geography Purity: ${(evaluation.geoPurity * 100).toFixed(1)}%`);
        lines.push(`Name/Sector Purity: ${(evaluation.namePurity * 100).toFixed(1)}%`);
        lines.push(`Balance Score: ${(evaluation.balanceScore * 100).toFixed(1)}%`);
        lines.push('');

        // Step 5: Mental Model Visualization
        lines.push('Step 5: MENTAL MODEL VISUALIZATION');
        lines.push('─'.repeat(60));
        const mentalModelViz = formatMentalModelVisualization(clusteringResult.clusters, records);
        lines.push(mentalModelViz);
        lines.push('');

        // Step 6: Enhanced Summary
        lines.push('Step 6: ENHANCED CLUSTER SUMMARY');
        lines.push('─'.repeat(60));
        const enhancedSummary = formatEnhancedClusterSummary(clusteringResult.clusters, records, evaluation);
        lines.push(enhancedSummary);
        lines.push('');

        // Step 7: Sample Prompt
        lines.push('Step 7: AI PROMPT (First 50 lines)');
        lines.push('─'.repeat(60));
        const missingRecords = records.map((r, i) => ({
            isin: r.isin || `REC${i}`,
            country: r.country || 'N/A',
            exchange: r.exchange_code || r.xchg || 'N/A',
            currency: r.currency || 'N/A',
            name: r.name || 'Unknown',
            sector: r.sector || 'Other',
        }));

        const missingRecordsList = missingRecords.slice(0, 10).map((r, i) =>
            `${i + 1}. ${r.name.padEnd(25)} | ${r.country.padEnd(2)} | ${r.exchange.padEnd(5)} | ${r.currency.padEnd(3)} | ${r.sector}`
        ).join('\n');

        if (missingRecords.length > 10) {
            lines.push(`... (${missingRecords.length} total MISSING records)`);
        }
        lines.push(missingRecordsList);
        lines.push('');
        lines.push('Full prompt would be sent to: ' + modelName);

        return { output: lines.join('\n') };
    }

    return { output: 'Unknown debug command. Use: /ai debug train <model> <fileId> | /ai debug prompt' };
}

export const AiProviderCommand: SlashCommand = {
    name: '/aiProvider',
    description: 'Configure AI provider (Usage: /aiProvider [show|set <provider> [model]|reset])',
    execute: async (args) => {
        const parts = args.trim().split(/\s+/).filter(Boolean);
        const subcommand = parts[0]?.toLowerCase();

        // Show current provider
        if (!subcommand || subcommand === 'show' || subcommand === 'status') {
            const config = loadAiProviderConfig();
            const info = PROVIDER_INFO[config.provider];

            const lines: string[] = [];
            lines.push('═══════════════════════════════════════════════════════════');
            lines.push('🤖 AI PROVIDER CONFIGURATION');
            lines.push('═══════════════════════════════════════════════════════════');
            lines.push('');
            lines.push(`Current provider: ${config.provider} ${info?.status === 'stable' ? '✓' : '⚠'}`);
            lines.push(`Model: ${config.modelName}${config.modelPath ? ` (${config.modelPath})` : ''}`);
            if (config.host) lines.push(`Host: ${config.host}`);
            lines.push(`Temperature: ${config.temperature}`);
            lines.push(`Max Tokens: ${config.maxTokens}`);
            lines.push('');
            lines.push('───────────────────────────────────────────────────────────');
            lines.push('Available Providers:');
            lines.push('');

            for (const [name, providerInfo] of Object.entries(PROVIDER_INFO)) {
                const isCurrent = name === config.provider;
                const marker = isCurrent ? '►' : ' ';
                const statusBadge = providerInfo.status === 'stable' ? '✓' : providerInfo.status === 'experimental' ? '⚠' : '○';
                lines.push(`  ${marker} ${name.padEnd(12)} ${statusBadge} ${providerInfo.description}`);
            }

            lines.push('');
            lines.push('Usage:');
            lines.push('  /aiProvider set ollama [model]    - Switch to Ollama');
            lines.push('  /aiProvider set llama-cpp <path>  - Switch to llama.cpp with GGUF file');
            lines.push('  /aiProvider set none              - Disable AI (clustering only)');
            lines.push('  /aiProvider reset                 - Reset to defaults');
            lines.push('  /aiProvider show                  - Show current config');

            return { output: lines.join('\n') };
        }

        // Set provider
        if (subcommand === 'set') {
            const providerName = parts[1]?.toLowerCase() as AiProviderType;

            if (!providerName || !PROVIDER_INFO[providerName]) {
                const available = Object.keys(PROVIDER_INFO).join(', ');
                return { output: `Unknown provider: '${parts[1]}'. Available: ${available}` };
            }

            if (providerName === 'ollama') {
                const model = parts[2] || 'qwen2.5:1.5b';
                saveAiProviderConfig({
                    provider: 'ollama',
                    modelName: model,
                    modelPath: undefined,
                });
                return { output: `AI provider set to: ollama (${model})` };
            }

            if (providerName === 'llama-cpp') {
                const modelPath = parts[2];
                if (!modelPath) {
                    return { output: 'Usage: /aiProvider set llama-cpp <path-to-gguf>\nExample: /aiProvider set llama-cpp ./models/qwen2.5-1.5b.Q4_K_M.gguf' };
                }

                // Validate file exists
                if (!fs.existsSync(modelPath)) {
                    return { output: `Model file not found: ${modelPath}` };
                }

                const fileName = path.basename(modelPath);
                saveAiProviderConfig({
                    provider: 'llama-cpp',
                    modelName: fileName,
                    modelPath,
                });
                return { output: `AI provider set to: llama-cpp (${fileName})` };
            }

            if (providerName === 'none') {
                saveAiProviderConfig({
                    provider: 'none',
                    modelName: 'none',
                });
                return { output: 'AI provider disabled. Clustering only, no AI analysis.' };
            }

            return { output: `Provider '${providerName}' configuration not yet implemented.` };
        }

        // Reset to defaults
        if (subcommand === 'reset' || subcommand === 'default') {
            resetAiProviderConfig();
            return { output: 'AI provider configuration reset to defaults.' };
        }

        return { output: 'Unknown command. Use: /aiProvider [show|set|reset]' };
    },
    suggestArgs: () => ['show', 'set ollama', 'set llama-cpp', 'set none', 'reset'],
};
