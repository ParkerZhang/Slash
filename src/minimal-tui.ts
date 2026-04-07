import * as readline from 'node:readline';
import { createRegisteredCommandRegistry } from './commands/index.js';
import { CliWorkspace } from './cli.js';

async function runMinimalTui() {
    const registry = createRegisteredCommandRegistry();
    const workspace = new CliWorkspace();
    
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        prompt: 'slash > '
    });

    console.log('--- TUI Slash Minimal (REPL Mode) ---');
    console.log('Type /help for commands, /exit to quit.');
    
    rl.prompt();

    rl.on('line', async (line) => {
        const input = line.trim();
        
        if (!input) {
            rl.prompt();
            return;
        }

        try {
            const result = await registry.execute(input, workspace);
            
            if (result.output) {
                console.log(result.output);
            }
            
            if (result.action === 'EXIT' || workspace.exited) {
                rl.close();
                return;
            }
            
            if (result.action === 'CLEAR') {
                console.clear();
            }
        } catch (err) {
            console.error('Error:', err instanceof Error ? err.message : String(err));
        }

        rl.prompt();
    });

    rl.on('close', () => {
        console.log('\nGoodbye!');
        process.exit(0);
    });
}

runMinimalTui().catch(err => {
    console.error('Fatal Error:', err);
    process.exit(1);
});
