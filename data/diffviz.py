"""
Code from https://github.com/kilink/ghdiff
"""
import difflib
import six
import html

def escape(text):
    return html.escape(text)


def diff(a, b, n=4):
    if isinstance(a, six.string_types):
        a = a.splitlines()
    if isinstance(b, six.string_types):
        b = b.splitlines()
    return colorize(list(difflib.unified_diff(a, b, n=n)))


def colorize(diff):
    return "\n".join(_colorize(diff))


def _colorize(diff):
    if isinstance(diff, six.string_types):
        lines = diff.splitlines()
    else:
        lines = diff
    lines.reverse()
    while lines and not lines[-1].startswith("@@"):
        lines.pop()
    if len(lines) > 0: lines.pop()  # Remove top of hunk. Lines not meaningful for us.
    yield '<div class="diff">'
    while lines:
        line = lines.pop()
        klass = ""
        if line.startswith("@@"):
            klass = "control"
        elif line.startswith("-"):
            klass = "delete"
            if lines:
                _next = []
                while lines and len(_next) < 2:
                    _next.append(lines.pop())
                if _next[0].startswith("+") and (
                        len(_next) == 1 or _next[1][0] not in ("+", "-")):
                    aline, bline = _line_diff(line[1:], _next.pop(0)[1:])
                    yield '<div class="delete">-%s</div>' % (aline,)
                    yield '<div class="insert">+%s</div>' % (bline,)
                    if _next:
                        lines.append(_next.pop())
                    continue
                lines.extend(reversed(_next))
        elif line.startswith("+"):
            klass = "insert"
        yield '<div class="%s">%s</div>' % (klass, escape(line),)
    yield "</div>"


def _line_diff(a, b):
    aline = []
    bline = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        if tag == 'equal':
            aline.append(escape(a[i1:i2]))
            bline.append(escape(b[j1:j2]))
            continue
        aline.append('<span class="highlight">%s</span>' % (escape(a[i1:i2]),))
        bline.append('<span class="highlight">%s</span>' % (escape(b[j1:j2]),))
    return "".join(aline), "".join(bline)

