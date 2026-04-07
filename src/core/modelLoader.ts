import * as fs from 'fs';
import * as path from 'path';

const CHUNK_SIZE = 25 * 1024 * 1024; // 25MB chunks

/**
 * Splits a file into smaller chunks for git compatibility
 */
export function splitFile(filePath: string): string[] {
    if (!fs.existsSync(filePath)) {
        throw new Error(`File not found: ${filePath}`);
    }

    const stats = fs.statSync(filePath);
    const fileName = path.basename(filePath);
    const dir = path.dirname(filePath);
    const buffer = fs.readFileSync(filePath);
    
    const chunkPaths: string[] = [];
    let offset = 0;
    let chunkIndex = 0;

    while (offset < buffer.length) {
        const chunk = buffer.subarray(offset, offset + CHUNK_SIZE);
        const chunkPath = path.join(dir, `${fileName}.part${chunkIndex}`);
        fs.writeFileSync(chunkPath, chunk);
        chunkPaths.push(chunkPath);
        
        offset += CHUNK_SIZE;
        chunkIndex++;
    }

    console.log(`Split ${fileName} (${(stats.size / 1024 / 1024).toFixed(2)}MB) into ${chunkIndex} parts.`);
    return chunkPaths;
}

/**
 * Recombines chunks into the original file if the original is missing
 */
export function ensureModelFile(filePath: string): void {
    if (fs.existsSync(filePath)) {
        return; // Already exists
    }

    const dir = path.dirname(filePath);
    const fileName = path.basename(filePath);
    
    // Check if parts exist
    const parts: string[] = [];
    let i = 0;
    while (true) {
        const partPath = path.join(dir, `${fileName}.part${i}`);
        if (fs.existsSync(partPath)) {
            parts.push(partPath);
            i++;
        } else {
            break;
        }
    }

    if (parts.length === 0) {
        return; // No parts to recombine, maybe it will be downloaded by the library
    }

    console.log(`Recombining ${fileName} from ${parts.length} parts...`);
    const writeStream = fs.createWriteStream(filePath);
    
    for (const partPath of parts) {
        const data = fs.readFileSync(partPath);
        writeStream.write(data);
    }
    
    writeStream.end();
    console.log(`Recombined ${fileName} successfully.`);
}

/**
 * CLI utility to split large models in the workspace
 */
export function splitModelsInWorkspace() {
    const root = process.cwd();
    const modelOnnx = path.join(root, 'modelFiles', 'Xenova', 'all-MiniLM-L6-v2', 'onnx', 'model.onnx');
    
    if (fs.existsSync(modelOnnx)) {
        splitFile(modelOnnx);
    } else {
        console.log("Model file not found at expected path.");
    }
}

const isMain = import.meta.url === `file://${process.argv[1]}` || process.argv[1]?.endsWith('modelLoader.ts');
if (isMain) {
    splitModelsInWorkspace();
}
