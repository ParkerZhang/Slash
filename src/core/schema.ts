export interface JsonSchema {
    type?: string | string[];
    properties?: Record<string, JsonSchema>;
    items?: JsonSchema | JsonSchema[];
    required?: string[];
    [key: string]: unknown;
}

function asSchema(value: unknown): JsonSchema | null {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
        return null;
    }
    return value as JsonSchema;
}

export function extractSubSchemas(schema: JsonSchema): Record<string, JsonSchema> {
    const subSchemas: Record<string, JsonSchema> = {};

    const visit = (node: JsonSchema, path: string) => {
        if (path) {
            subSchemas[path] = node;
        }

        const properties = node.properties ?? {};
        for (const [key, value] of Object.entries(properties)) {
            const child = asSchema(value);
            if (child) {
                visit(child, path ? `${path}.${key}` : key);
            }
        }

        if (node.items) {
            if (Array.isArray(node.items)) {
                node.items.forEach((item, index) => {
                    const child = asSchema(item);
                    if (child) {
                        visit(child, path ? `${path}[]${index}` : `[]${index}`);
                    }
                });
            } else {
                const child = asSchema(node.items);
                if (child) {
                    visit(child, path ? `${path}[]` : '[]');
                }
            }
        }
    };

    visit(schema, '');
    return subSchemas;
}

export function buildSelectedSchema(
    schema: JsonSchema,
    subSchemas: Record<string, JsonSchema>,
    selectedPaths: string[],
): JsonSchema {
    if (selectedPaths.length === 0) {
        return schema;
    }

    return {
        type: 'object',
        title: 'SelectedSubSchemas',
        properties: Object.fromEntries(
            selectedPaths
                .filter((path) => subSchemas[path])
                .sort((a, b) => a.localeCompare(b))
                .map((path) => [path, subSchemas[path]]),
        ),
        selectedPaths,
    };
}
