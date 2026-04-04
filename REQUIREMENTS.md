# Requirements

## System Requirements
- **Node.js**: >= 18.0.0
- **npm**: >= 9.0.0
- **Ollama**: installed locally for AI schema features

## Runtime Dependencies
| Package        | Version    | Purpose                              |
| -------------- | ---------- | ------------------------------------ |
| alasql         | ^4.17.1    | SQL querying for CSV data            |
| ink            | ^6.8.0     | React-based TUI framework            |
| ink-text-input | ^6.0.0     | Text input component for Ink         |
| react          | ^19.2.4    | UI library                           |
| react-dom      | ^19.2.4    | React DOM renderer                   |
| ts-node        | ^10.9.2    | TypeScript execution                 |
| typescript     | ^6.0.2     | TypeScript compiler                  |

## Development Dependencies
| Package        | Version    | Purpose                              |
| -------------- | ---------- | ------------------------------------ |
| @types/node    | ^25.5.1    | Node.js type definitions             |
| @types/react   | ^19.2.14   | React type definitions               |
| @types/react-dom | ^19.2.3 | React DOM type definitions           |
| tsx            | ^4.21.0    | Fast TypeScript runner               |
| vitest         | ^4.1.2     | Unit testing framework               |

## Installation
```bash
npm install
```

## Development
```bash
npm start          # Run the application
npm test           # Run tests
npm run test:watch # Run tests in watch mode
```

## V2 Feature Requirements

- **AI features** require a local Ollama model and a working `ollama` CLI on `PATH`.
- **Refresh on the fly** is intended for command/core changes and does not fully hot-reload the Ink UI tree.
- **Schema workflow** depends on generating or attaching schema metadata to a loaded file, typically via `/ai <id>`.
