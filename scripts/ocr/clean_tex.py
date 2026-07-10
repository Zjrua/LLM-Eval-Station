#!/usr/bin/env python3
"""Clean up common OCR artifacts in the LaTeX file for compilation."""

import re, sys

def clean_latex(path_in: str, path_out: str = None):
    with open(path_in, encoding='utf-8') as f:
        content = f.read()

    changes = []

    # 1. Fix HTML entities
    # &#x27; → '
    c1 = content.count('&#x27;')
    if c1:
        content = content.replace('&#x27;', "'")
        changes.append(f'Fixed {c1} &#x27; → apostrophe')

    # &#xA0; → space
    c2 = content.count('&#xA0;')
    if c2:
        content = content.replace('&#xA0;', ' ')
        changes.append(f'Fixed {c2} &#xA0; → space')

    # &lt; → <
    c3 = content.count('&lt;')
    if c3:
        content = content.replace('&lt;', '<')
        changes.append(f'Fixed {c3} &lt; → <')

    # &gt; → >
    c4 = content.count('&gt;')
    if c4:
        content = content.replace('&gt;', '>')
        changes.append(f'Fixed {c4} &gt; → >')

    # &amp; → &
    c5 = content.count('&amp;')
    if c5:
        content = content.replace('&amp;', '\\&')
        changes.append(f'Fixed {c5} &amp; → \\&')

    # 2. Fix multiple \tag in one equation
    # Pattern: \tag{...} \label{...} or \label{...} \tag{...}
    # Replace with just \tag{...} (keep the tag, remove the label)
    old_count = content.count('\\tag{')
    # Remove \label that comes right before or after \tag
    content = re.sub(r'\\label\{[^}]*\}\s*\\tag\{', '\\tag{', content)
    content = re.sub(r'\\tag\{[^}]*\}\s*\\label\{[^}]*\}', lambda m: m.group(0).replace('\\label', '% \\label'), content)
    new_count = content.count('\\tag{')
    if old_count != new_count:
        changes.append(f'Fixed multiple \\tag issues')

    # 3. Fix \\ at end of text (LaTeX line break outside of math/tabular)
    # This happens in some edge cases but is hard to fix generically

    # 4. Fix any stray # that isn't in a LaTeX command
    # Outside of math mode, # is special
    # This is hard to fix generically, skip for now

    # 5. Add headheight fix
    if '\\setlength{\\headheight}' not in content:
        content = content.replace(
            '\\begin{document}',
            '\\setlength{\\headheight}{13.6pt}\n\\begin{document}'
        )
        changes.append('Added \\headheight fix')

    if path_out is None:
        path_out = path_in

    with open(path_out, 'w', encoding='utf-8') as f:
        f.write(content)

    if changes:
        print('\n'.join(changes))
    else:
        print('No changes needed')

    return content


if __name__ == '__main__':
    if len(sys.argv) > 1:
        clean_latex(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        clean_latex('/Users/zjrua/tmp/概率论与数理统计教程_OCR.tex')
