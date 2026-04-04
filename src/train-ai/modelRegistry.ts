import { ModelProfile } from './types.js';

export const GUIDED_PROMPT_QWEN25_1_5B = `REQUEST: 59 securities from global markets (US, JP, HK, GB, BR, MX, CL, AR)
RESPONSE: 42 securities with prices
MISSING: 17 securities (present in request, absent in response)

MISSING Records:
1. Petrobras PN       | BR | LATAM  | BRL | Energy
2. VALE SA            | BR | LATAM  | BRL | Energy
3. Itau Unibanco      | BR | LATAM  | BRL | Banking
4. Banco de Chile     | CL | LATAM  | CLP | Banking
5. Grupo Financiero   | AR | LATAM  | ARS | Banking
6. America Movil      | MX | LATAM  | MXN | Telecom
7. Walmart de Mexico  | MX | LATAM  | MXN | Retail
8. Enel Chile         | CL | LATAM  | CLP | Energy
9. Banco do Brasil    | BR | LATAM  | BRL | Banking
10. Bradesco SA       | BR | LATAM  | BRL | Banking
11. Cemex SAB         | MX | LATAM  | MXN | Energy
12. YPF SA            | AR | LATAM  | ARS | Energy
13. Falabella         | CL | LATAM  | CLP | Retail
14. Grupo Bimbo       | MX | LATAM  | MXN | Retail
15. Mitsubishi UFJ    | JP | TSE    | JPY | Banking
16. KDDI Corp         | JP | TSE    | JPY | Telecom
17. Shell PLC         | GB | XLON   | GBP | Energy

CLUSTERING RESULTS (K=4):
Cluster 1 (6): BR(4), CL(1), AR(1) | Banking(6) | LATAM(6)
Cluster 2 (4): BR(2), CL(1), AR(1) | Energy(4)  | LATAM(4)
Cluster 3 (4): MX(3), CL(1)        | Retail(3), Telecom(1) | LATAM(4)
Cluster 4 (3): JP(2), GB(1)        | Mixed sectors | TSE(2), XLON(1)

QUESTIONS TO ANSWER:
Q1: Which geography is MOST affected by MISSING records?
Q2: Which sector has the MOST MISSING records?
Q3: Is the MISSING pattern random or systematic?
Q4: What vendor coverage gap does this suggest?
Q5: Which cluster represents the largest coverage gap?

RESPONSE TEMPLATE:
MISSING Analysis Summary:
- Most affected geography: [country/region] with [N] records ([X]% of MISSING)
- Most affected sector: [sector] with [N] records ([X]% of MISSING)
- Pattern assessment: [random/systematic] - [1-sentence reason]
- Vendor gap: [description]
- Largest gap: Cluster [N] - [description]`;

export const STANDARD_PROMPT_QWEN25_3B = `Analyze these MISSING securities from a pricing request.

REQUEST: 59 securities from global markets
RESPONSE: 42 securities received
MISSING: 17 securities (28.8% missing rate)

CLUSTERING RESULTS:
Cluster 1 (6): LATAM Banking - BR(4), CL(1), AR(1)
Cluster 2 (4): LATAM Energy - BR(2), CL(1), AR(1)
Cluster 3 (4): LATAM Retail/Telecom - MX(3), CL(1)
Cluster 4 (3): JP/GB Mixed - JP(2), GB(1)

QUALITY METRICS:
- Geography Purity: 81.3%
- Sector Purity: 78.5%
- Balance Score: 62.1%

Provide analysis covering:
1. MISSING Pattern Summary (geography + sector breakdown with counts and percentages)
2. Key Findings (3-4 bullet points with specific evidence)
3. Cluster Interpretation (what each cluster represents and why it matters)
4. Vendor Coverage Assessment (gaps and strengths)

Keep it under 400 words. Use specific numbers from the data.`;

export const JSON_OUTPUT_PROMPT = `Convert this MISSING analysis into JSON:

DATA:
- Requested: 59, Received: 42, Missing: 17 (28.8%)
- Geography: BR(6), MX(4), CL(3), AR(2), JP(2), GB(1)
- Sectors: Banking(6), Energy(5), Retail(3), Telecom(2), Automotive(1)
- Clusters: LATAM Banking(6), LATAM Energy(4), LATAM Retail/Telecom(4), JP/GB Mixed(3)
- Metrics: Geo Purity=0.813, Sector Purity=0.785, Balance=0.621

JSON SCHEMA:
{
  "summary": { "requested": N, "received": N, "missing": N, "missingRate": N.N },
  "missingByGeography": { "BR": N, "MX": N, "CL": N, "AR": N, "JP": N, "GB": N },
  "missingBySector": { "Banking": N, "Energy": N, "Retail": N, "Telecom": N, "Automotive": N },
  "clusters": [
    { "id": N, "size": N, "dominantGeography": "string", "dominantSector": "string", "interpretation": "string" }
  ],
  "qualityMetrics": { "geographyPurity": N.N, "sectorPurity": N.N, "balanceScore": N.N, "crossRegionClusters": N },
  "insights": ["string", "string", "string", "string"],
  "recommendation": "string"
}

Output ONLY valid JSON. No markdown fences. No explanation.`;

export const MODEL_REGISTRY: Record<string, ModelProfile> = {
    'qwen2.5:1.5b': {
        name: 'qwen2.5:1.5b',
        type: 'chat',
        dimensions: 2048,
        contextWindow: 32768,
        temperature: 0.1,
        maxTokens: 500,
        fallbackModel: 'qwen2.5:3b',
        idealPrompts: {
            guidedAnalysis: GUIDED_PROMPT_QWEN25_1_5B,
            jsonOutput: JSON_OUTPUT_PROMPT,
        },
    },
    'qwen2.5:3b': {
        name: 'qwen2.5:3b',
        type: 'chat',
        dimensions: 2048,
        contextWindow: 32768,
        temperature: 0.2,
        maxTokens: 800,
        idealPrompts: {
            standardAnalysis: STANDARD_PROMPT_QWEN25_3B,
            jsonOutput: JSON_OUTPUT_PROMPT,
        },
    },
    'qwen2.5:7b': {
        name: 'qwen2.5:7b',
        type: 'chat',
        dimensions: 2048,
        contextWindow: 32768,
        temperature: 0.3,
        maxTokens: 1000,
        idealPrompts: {
            standardAnalysis: STANDARD_PROMPT_QWEN25_3B,
            jsonOutput: JSON_OUTPUT_PROMPT,
        },
    },
    'qwen3.5:0.8b': {
        name: 'qwen3.5:0.8b',
        type: 'chat',
        dimensions: 2048,
        contextWindow: 32768,
        temperature: 0.1,
        maxTokens: 500,
        fallbackModel: 'qwen2.5:1.5b',
        idealPrompts: {
            guidedAnalysis: GUIDED_PROMPT_QWEN25_1_5B,
            jsonOutput: JSON_OUTPUT_PROMPT,
        },
    },
    'nomic-embed-text:latest': {
        name: 'nomic-embed-text:latest',
        type: 'embedding',
        dimensions: 768,
        contextWindow: 8192,
        temperature: 0,
        maxTokens: 0,
    },
    'all-minilm:latest': {
        name: 'all-minilm:latest',
        type: 'embedding',
        dimensions: 384,
        contextWindow: 256,
        temperature: 0,
        maxTokens: 0,
    },
};

export const CANDIDATE_MODELS = [
    { model: 'qwen2.5:1.5b', status: 'recommended', embedding: true, notes: 'Best balance of speed and quality' },
    { model: 'qwen3.5:0.8b', status: 'recommended', embedding: true, notes: 'Smallest Qwen3.5, fast and efficient' },
    { model: 'nomic-embed-text:latest', status: 'embedding', embedding: true, notes: 'Dedicated embedding model' },
    { model: 'qwen2.5:3b', status: 'candidate', embedding: true, notes: 'Better reasoning' },
    { model: 'qwen2.5:7b', status: 'candidate', embedding: true, notes: 'Strongest analysis' },
    { model: 'all-minilm:latest', status: 'candidate', embedding: true, notes: 'Fastest embeddings' },
    { model: 'gemma3:4b', status: 'candidate', embedding: true, notes: 'Alternative option' },
    { model: 'phi4:3.8b', status: 'candidate', embedding: false, notes: 'Good reasoning, needs external embeddings' },
];

export function getModelProfile(modelName: string): ModelProfile | undefined {
    return MODEL_REGISTRY[modelName];
}

export function listCandidateModels(): typeof CANDIDATE_MODELS {
    return CANDIDATE_MODELS;
}
