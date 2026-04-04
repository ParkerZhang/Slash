export interface BannerStyle {
    name: string;
    title: string;
    lines: string[];
}

export const BANNERS: BannerStyle[] = [
    {
        name: 'haiku-1',
        title: '/slash · TUI Data Workspace',
        lines: [
            'Data flows silent',
            'Patterns emerge from chaos —',
            'Clusters reveal the truth.',
        ],
    },
    {
        name: 'haiku-2',
        title: '/slash · TUI Data Workspace',
        lines: [
            'Bytes dance in the dark,',
            'Algorithms find the hidden,',
            'Order from the noise.',
        ],
    },
    {
        name: 'haiku-3',
        title: '/slash · TUI Data Workspace',
        lines: [
            'Columns hold the past,',
            'Rows whisper forgotten stories,',
            'Truth wakes with query.',
        ],
    },
    {
        name: 'haiku-4',
        title: '/slash · TUI Data Workspace',
        lines: [
            'Silent data streams,',
            'Patterns bloom like morning flowers,',
            'Insight breaks the dawn.',
        ],
    },
    {
        name: 'haiku-5',
        title: '/slash · TUI Data Workspace',
        lines: [
            'Code weaves through the void,',
            'Numbers sing their ancient song,',
            'Meaning finds its form.',
        ],
    },
    {
        name: 'haiku-6',
        title: '/slash · TUI Data Workspace',
        lines: [
            'Fields of ones and zeros,',
            'Harvest yields a golden truth,',
            'Wisdom in the shell.',
        ],
    },
    {
        name: 'haiku-7',
        title: '/slash · TUI Data Workspace',
        lines: [
            'Terminal glows bright,',
            'Commands echo through the void,',
            'Data speaks at last.',
        ],
    },
    {
        name: 'haiku-8',
        title: '/slash · TUI Data Workspace',
        lines: [
            'Screens light up the dark,',
            'Queries pierce the silent data,',
            'Answers bloom like spring.',
        ],
    },
    {
        name: 'qin',
        title: '/slash 峄石铭 · 秦篆',
        lines: [
            '皇帝立国，维初在昔，嗣世称王',
            '讨伐乱逆，威动四极，武义直方',
            '戎臣奉诏，经时不久，灭六暴强',
            '廿有六年，上荐高号，孝道显明',
            '一、山岳刻石工程',
            '1. 山岳名录 - 搜罗天下名山',
            '2. 篆文编纂 - 以李斯笔法',
            '3. 碑文镌刻 - 程序生成篆文',
        ],
    },
    {
        name: 'churchill',
        title: '/slash · We Shall Fight',
        lines: [
            'We shall fight on the beaches,',
            'We shall fight on the landing grounds,',
            'We shall fight in the fields and in the streets,',
            'We shall fight in the hills;',
            'We shall never surrender.',
            '— Winston Churchill, 1940',
        ],
    },
];

export const DEFAULT_BANNER = 'haiku-1';

export function getBannerStyle(name?: string): BannerStyle {
    const banner = BANNERS.find(b => b.name === name);
    return banner || BANNERS.find(b => b.name === DEFAULT_BANNER)!;
}
