#!/usr/bin/env python3
"""
Convert Unlimited-OCR JSON to LaTeX (.tex) v3.
Fixes: oneside (no blank pages), filters front matter, skips OCR'd TOC.
"""

import argparse
import json
import os
import re

# ── Helpers ────────────────────────────────────────────────────────────────

def clean_spaces(s: str) -> str:
    return re.sub(r'(?<=\d)\s+(?=\d)', '', s)


def html_to_latex_table(html: str) -> str:
    rows = re.findall(r'<tr>(.*?)</tr>', html, re.DOTALL)
    if not rows:
        return '% [table parse failed]'
    parsed = []
    for row in rows:
        cells = re.findall(r'<td>(.*?)</td>', row, re.DOTALL)
        if cells:
            parsed.append(cells)
    if not parsed:
        return '% [empty table]'
    cols = max(len(r) for r in parsed)
    col_spec = '|' + 'c|' * cols
    lines = ['\\begin{tabular}{' + col_spec + '}', '\\hline']
    for cells in parsed:
        line = ' & '.join(clean_spaces(c.strip()) for c in cells) + ' \\\\'
        lines.append(line)
        lines.append('\\hline')
    lines.append('\\end{tabular}')
    return '\n'.join(lines)


def extract_content(line: str):
    m = re.match(r'<\|det\|>(\w+)\s+\[[\d,\s]+\]\s*<\|/det\|>\s*(.*)', line)
    if m:
        return m.group(1), m.group(2).strip()
    return None, None


SKIP = frozenset({
    'page_number', 'P', 'E', 'F', 'W', 'v', 'y', 'g', 'Y', 'U',
    'a', 'N', 'G', 'R', 't', 'u', 'H', 'p', 'aside_text', 'image',
    'footer',
})


# ── Content filtering ─────────────────────────────────────────────────────

# Front matter headers to suppress completely
FRONT_MATTER = frozenset({
    '中国证券监督管理委员会', '"十二五"普通高等教育本科国家级规划教材',
    '第三版前言', '第二版前言', '第一版前言', '目录', '图书在版编目(CIP)数据',
    '内容提要', '概率论与', '谨以此书纪念魏宗舒教授',
})


def is_chapter_head(text: str):
    """Detect '第一章 ...', '第二章 ...' etc."""
    return bool(re.match(r'^第[一二三四五六七八九十百千零]+章\s', text))


# ── Document element stream ───────────────────────────────────────────────

STATS = {'chapters': 0, 'sections': 0, 'text_blocks': 0, 'equations': 0, 'tables': 0}

def page_to_elements(pages: dict, page_nums: list):
    eq_counter = [0]
    current_chapter = None
    seen_sections = set()
    chapter_started = False
    _emitted_chapters = set()  # first "第X章" header triggers mainmatter

    for pn in page_nums:
        key = f'p{pn}'
        if key not in pages:
            continue
        raw = pages[key]['text']
        lines = raw.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line:
                continue
            tag, content = extract_content(line)
            if tag is None or tag in SKIP:
                continue

            # ── Header ──
            if tag == 'header':
                ch_text = clean_spaces(content)
                # Skip front matter
                if ch_text in FRONT_MATTER:
                    continue
                # Skip "第一章" standalone (TOC page artifact)
                if ch_text.strip() in ('第一章', '第二章', '第三章', '第四章',
                                        '第五章', '第六章', '第七章', '第八章'):
                    continue
                if is_chapter_head(ch_text):
                    if ch_text != current_chapter:
                        current_chapter = ch_text
                        chapter_name = re.sub(r'^第[一二三四五六七八九十百千零]+章\s*', '', ch_text)
                        # Skip chapter name-only header that's clearly front matter
                        if chapter_name in ('目录',):
                            continue
                        if not chapter_started:
                            yield ('mainmatter', None)
                            chapter_started = True
                        seen_sections.clear()
                        STATS['chapters'] += 1
                        yield ('chapter', chapter_name)

            # ── Title (sections) ──
            elif tag == 'title':
                t = clean_spaces(content)
                t = t.replace('|||', '').strip()
                if not t:
                    continue
                if not chapter_started:
                    continue  # skip titles before first chapter

                # Extract section number
                m = re.match(r'^§?\s*([\d]+(?:\.[\d]+)*)\s+(.*)', t)
                if m:
                    num = m.group(1)
                    name = m.group(2)
                    depth = len(num.split('.'))
                    dedup_key = f'{current_chapter}|{num}'
                    if dedup_key in seen_sections:
                        continue
                    seen_sections.add(dedup_key)

                    # Auto-detect chapter from section number
                    ch_num = num.split('.')[0]
                    CN_NUMS = ['〇', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
                    CHAPTER_NAMES = ['', '随机事件与概率', '随机变量及其分布',
                                     '多维随机变量及其分布', '大数定律与中心极限定理',
                                     '统计量及其分布', '参数估计', '假设检验',
                                     '方差分析与回归分析']
                    ch_idx = int(ch_num)
                    if ch_idx < len(CHAPTER_NAMES):
                        ch_cn = CN_NUMS[ch_idx]
                        expected_ch = f'第{ch_cn}章 {CHAPTER_NAMES[ch_idx]}'
                    else:
                        expected_ch = None
                    if expected_ch and expected_ch != current_chapter and ch_idx not in _emitted_chapters:
                        # Auto-emit the chapter that this section belongs to
                        _emitted_chapters.add(ch_idx)
                        current_chapter = expected_ch
                        yield ('chapter', CHAPTER_NAMES[ch_idx])
                        seen_sections.clear()
                        seen_sections.add(dedup_key)

                    STATS['sections'] += 1
                    if depth >= 3:
                        yield ('subsubsection', f'{num} {name}')
                    elif depth == 2:
                        yield ('subsection', f'{num} {name}')
                    else:
                        yield ('section', f'{num} {name}')
                elif t.startswith('习题') or t.startswith('练习'):
                    yield ('section*', t)
                elif re.match(r'^[一二三四五六七八九十]、', t):
                    yield ('subsubsection*', t)

            # ── Text ──
            elif tag == 'text':
                if not chapter_started:
                    continue
                STATS['text_blocks'] += 1
                yield ('text', clean_spaces(content))

            # ── Equation ──
            elif tag == 'equation':
                if not chapter_started:
                    continue
                eq_parts = [content]
                found_end = '\\]' in content
                while not found_end and i < len(lines):
                    nl = lines[i].strip()
                    i += 1
                    eq_parts.append(nl)
                    if '\\]' in nl:
                        found_end = True
                eq_text = ' '.join(eq_parts)
                eq_text = re.sub(r'^\\\[\s*', '', eq_text)
                eq_text = re.sub(r'\s*\\\]$', '', eq_text)
                eq_text = clean_spaces(eq_text.strip())
                if eq_text:
                    eq_counter[0] += 1
                    STATS['equations'] += 1
                    yield ('equation', (eq_text, eq_counter[0]))

            # ── Table ──
            elif tag == 'table':
                if not chapter_started:
                    continue
                table_parts = [content]
                found_close = '</table>' in content
                while not found_close and i < len(lines):
                    nl = lines[i].strip()
                    i += 1
                    table_parts.append(nl)
                    if '</table>' in nl:
                        found_close = True
                html = ' '.join(table_parts)
                m = re.search(r'(<table>.*?</table>)', html, re.DOTALL)
                if m:
                    STATS['tables'] += 1
                    yield ('table', html_to_latex_table(m.group(1)))

            # ── Image caption ──
            elif tag == 'image_caption':
                if not chapter_started:
                    continue
                yield ('caption', clean_spaces(content))


# ── LaTeX builder ─────────────────────────────────────────────────────────

def build_latex(pages: dict, page_nums: list) -> str:
    parts = []
    buf = []

    def flush():
        if buf:
            parts.append('  ' + ' '.join(buf) + '\n\n')
            buf.clear()

    for elem in page_to_elements(pages, page_nums):
        t, c = elem

        if t == 'mainmatter':
            flush()
            parts.append('\\mainmatter\n\n')

        elif t == 'chapter':
            flush()
            parts.append(f'\\chapter{{{c}}}\n\n')

        elif t == 'section':
            flush()
            parts.append(f'  \\section{{{c}}}\n\n')

        elif t == 'section*':
            flush()
            parts.append(f'  \\section*{{{c}}}\n\n')

        elif t == 'subsection':
            flush()
            parts.append(f'  \\subsection{{{c}}}\n\n')

        elif t == 'subsection*':
            flush()
            parts.append(f'  \\subsection*{{{c}}}\n\n')

        elif t == 'subsubsection':
            flush()
            parts.append(f'  \\subsubsection{{{c}}}\n\n')

        elif t == 'subsubsection*':
            flush()
            parts.append(f'  \\subsubsection*{{{c}}}\n\n')

        elif t == 'text':
            buf.append(c)

        elif t == 'equation':
            flush()
            eq_text, num = c
            parts.append(f'  \\begin{{equation}}\n    {eq_text}\n    \\label{{eq:auto{num}}}\n  \\end{{equation}}\n\n')

        elif t == 'table':
            flush()
            parts.append(f'  \\[\n    {c}\n  \\]\n\n')

        elif t == 'caption':
            flush()
            parts.append(f'  \\begin{{center}} {{\\small {c}}} \\end{{center}}\n\n')

    flush()
    return ''.join(parts)


def latex_document(body: str, title: str = '概率论与数理统计教程（OCR版）') -> str:
    return r'''\documentclass[UTF8,a4paper,11pt,oneside]{ctexbook}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{geometry}
\geometry{top=2.5cm,bottom=2.5cm,left=3cm,right=3cm}
\usepackage{hyperref}
\usepackage{array}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\thepage}
\fancyhead[LO]{\leftmark}
\fancyhead[RE]{\rightmark}
\usepackage{indentfirst}
\setlength{\parindent}{2em}
\setlength{\headheight}{13.6pt}
\maxdeadcycles=10000

\begin{document}

\title{\textbf{''' + title + r'''}}
\author{茆诗松 等 著 \\ OCR 识别整理版}
\date{\today}
\maketitle
\tableofcontents
\newpage

''' + body + r'''

\end{document}
'''


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='OCR JSON file')
    parser.add_argument('-o', '--output', default=None, help='Output .tex file')
    args = parser.parse_args()

    with open(args.input, encoding='utf-8') as f:
        data = json.load(f)

    nums = sorted(int(k[1:]) for k in data if k.startswith('p') and k[1:].isdigit())
    print(f'Loaded {len(nums)} pages (p{nums[0]}–p{nums[-1]})')

    body = build_latex(data, nums)
    doc = latex_document(body)

    out = args.output or (os.path.splitext(args.input)[0] + '.tex')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(doc)

    print(f'LaTeX: {out}  ({len(doc)} chars)')
    print(f'Stats: {STATS["chapters"]} chapters, {STATS["sections"]} sections, '
          f'{STATS["equations"]} equations, {STATS["tables"]} tables')
    print(f'       {STATS["text_blocks"]} text blocks')


if __name__ == '__main__':
    main()
