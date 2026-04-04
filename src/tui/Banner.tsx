import React from 'react';
import { Text, Box } from 'ink';
import { BannerStyle } from './banners.js';

interface BannerProps {
    banner: BannerStyle;
    onCycle: () => void;
}

export const BannerComponent: React.FC<BannerProps> = ({ banner }) => {
    return (
        <Box flexDirection="column" borderStyle="round" borderDimColor paddingX={1} paddingY={0}>
            <Box flexDirection="row">
                <Text bold color="#00FF00">{banner.title}</Text>
                <Text dimColor> (Ctrl+B to cycle)</Text>
            </Box>
            {banner.lines.map((line: string, i: number) => (
                <Text key={`line-${i}`} color="#00FF00">
                    {'  '}{line}
                </Text>
            ))}
        </Box>
    );
};
