export interface ModelProfile {
    name: string;
    type: 'chat' | 'embedding';
    dimensions: number;
    contextWindow: number;
    temperature: number;
    maxTokens: number;
    fallbackModel?: string;
    idealPrompts?: Record<string, string>;
}

export interface MissingRecord {
    isin: string;
    country: string;
    name: string;
    sector: string;
    exchange: string;
    currency: string;
}

export interface ClusterInfo {
    id: number;
    size: number;
    dominantGeography: string;
    dominantSector: string;
    interpretation: string;
    members: MissingRecord[];
}

export interface QualityMetrics {
    geographyPurity: number;
    sectorPurity: number;
    balanceScore: number;
    crossRegionClusters: number;
}

export interface TrainAiOutput {
    summary: {
        requested: number;
        received: number;
        missing: number;
        missingRate: number;
    };
    missingByGeography: Record<string, number>;
    missingBySector: Record<string, number>;
    clusters: ClusterInfo[];
    qualityMetrics: QualityMetrics;
    insights: string[];
    recommendation: string;
    aiAnalysis: string;
}

export interface WorkflowStep {
    stepNumber: number;
    name: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    input: string;
    testingPrompt?: string;
    result?: string;
    expectedResult: string;
    difference?: string;
    startTime?: string;
    endTime?: string;
}

export interface WorkflowState {
    workflowId: string;
    seed: number;
    modelProfile: string;
    currentStep: number;
    totalSteps: number;
    steps: WorkflowStep[];
    stepOutputs: Record<string, any>;
    missingRecords?: MissingRecord[];
    embeddings?: number[][];
    output?: TrainAiOutput;
    startTime: string;
    lastUpdated: string;
    status: 'running' | 'paused' | 'completed' | 'failed';
    error?: string;
}
