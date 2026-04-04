import { Workspace, CommandResult } from '../core/types.js';

export interface SlashCommand {
    name: string;
    description: string;
    execute: (args: string, workspace: Workspace, registry: CommandRegistry) => CommandResult | Promise<CommandResult>;
    // Optional: for better suggestions/auto-complete
    suggestArgs?: (input: string, workspace: Workspace) => string[];
}

export class CommandRegistry {
    private commands: Map<string, SlashCommand> = new Map();

    register(command: SlashCommand): void {
        this.commands.set(command.name.toLowerCase(), command);
    }

    clear(): void {
        this.commands.clear();
    }

    getCommand(name: string): SlashCommand | undefined {
        return this.commands.get(name.toLowerCase());
    }

    getAllCommands(): SlashCommand[] {
        return Array.from(this.commands.values());
    }

    async execute(input: string, workspace: Workspace): Promise<CommandResult> {
        const parts = input.trim().split(/\s+/);
        if (parts.length === 0) return { output: '' };

        const cmdName = parts[0].toLowerCase();
        const args = parts.slice(1).join(' ');

        const cmd = this.getCommand(cmdName);
        if (!cmd) {
            return { output: `Unknown command: ${cmdName}` };
        }

        try {
            return await cmd.execute(args, workspace, this);
        } catch (error) {
            return { output: `Error executing ${cmdName}: ${error instanceof Error ? error.message : String(error)}` };
        }
    }
}

export const globalRegistry = new CommandRegistry();
