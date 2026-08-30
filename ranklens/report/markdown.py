"""A tiny, dependency-free Markdown -> HTML converter.

Just enough to render the AI narrative (`AnalyzeReport.ai_narrative` /
`CompareReport.ai_narrative`) inside the HTML report without pulling in a
Markdown library. HTML is escaped FIRST, so model output can never inject
markup. Supports: ``#``/``##``/``###`` headings, ``**bold**``, ``*italic*``,
`` `code` ``, ``-``/``*`` unordered lists, ``1.`` ordered lists (with proper
indentation-based nesting), paragraphs and hard line breaks. Inline LaTeX math
(``$\\rightarrow$``, ``$r = -0.45$``, ``\\(...\\)``, ``$$...$$``) is converted
to plain Unicode text, since the report has no math renderer. Robust to
``None``/empty input.
"""
from __future__ import annotations

import html as _html
import re

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
_CODE = re.compile(r"`([^`]+?)`")
_UL = re.compile(r"^(\s*)[-*]\s+(.*)$")
_OL = re.compile(r"^(\s*)\d+[.)]\s+(.*)$")
_H = re.compile(r"^(#{1,3})\s+(.*)$")

# LaTeX command -> Unicode. The report has no MathJax/KaTeX, so the model's
# occasional ``$\rightarrow$`` would otherwise render as literal source text.
_MATH_COMMANDS = {
    r"\rightarrow": "→", r"\Rightarrow": "⇒", r"\longrightarrow": "→",
    r"\leftarrow": "←", r"\Leftarrow": "⇐", r"\leftrightarrow": "↔",
    r"\to": "→", r"\gets": "←",
    r"\times": "×", r"\cdot": "·", r"\div": "÷", r"\pm": "±", r"\mp": "∓",
    r"\approx": "≈", r"\neq": "≠", r"\equiv": "≡",
    r"\leq": "≤", r"\geq": "≥", r"\le": "≤", r"\ge": "≥",
    r"\ll": "≪", r"\gg": "≫", r"\sim": "~", r"\propto": "∝",
    r"\infty": "∞", r"\sum": "∑", r"\prod": "∏", r"\sqrt": "√",
    r"\Delta": "Δ", r"\Sigma": "Σ", r"\Omega": "Ω",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\theta": "θ", r"\lambda": "λ", r"\mu": "μ",
    r"\pi": "π", r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ", r"\phi": "φ",
    r"\%": "%", r"\&": "&", r"\#": "#", r"\_": "_", r"\$": "$",
}
# Longest-first so e.g. ``\leq`` wins over ``\le``.
_MATH_COMMANDS_SORTED = sorted(_MATH_COMMANDS.items(), key=lambda kv: -len(kv[0]))


def _replace_commands(s: str) -> str:
    """Swap known LaTeX commands for Unicode, respecting command boundaries."""
    for cmd, repl in _MATH_COMMANDS_SORTED:
        s = re.sub(re.escape(cmd) + r"(?![a-zA-Z])", repl, s)
    return s


def _demath(inner: str) -> str:
    """Turn the inside of a math span into plain text."""
    s = _replace_commands(inner)
    # \text{...}, \mathrm{...}, \operatorname{...} -> their contents
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|mathsf|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)            # drop any leftover \command
    s = s.replace("{", "").replace("}", "")      # drop grouping braces
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def _maybe_inline_math(m: "re.Match[str]") -> str:
    """Convert ``$...$`` only when it looks like math — leave currency alone."""
    inner = m.group(1)
    if "\\" in inner or re.search(r"[=<>^]", inner) or "_{" in inner:
        return _demath(inner)
    return m.group(0)


def _strip_math(text: str) -> str:
    """Replace LaTeX math spans with plain Unicode text."""
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: _demath(m.group(1)), text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: _demath(m.group(1)), text, flags=re.DOTALL)
    text = re.sub(r"\\\((.+?)\\\)", lambda m: _demath(m.group(1)), text, flags=re.DOTALL)
    text = re.sub(r"\$([^$\n]{1,80}?)\$", _maybe_inline_math, text)
    return _replace_commands(text)  # catch any bare, unwrapped commands


def _inline(text: str) -> str:
    """Apply inline spans to an already HTML-escaped line."""
    text = _CODE.sub(r"<code>\1</code>", text)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    text = _ITALIC.sub(r"<em>\1</em>", text)
    return text


def md_to_html(text: str | None) -> str:
    """Convert a small Markdown subset to safe HTML. Empty input -> ""."""
    if not text or not str(text).strip():
        return ""
    src = _strip_math(str(text)).replace("\r\n", "\n")
    lines = _html.escape(src).split("\n")

    out: list[str] = []
    # Each level: {"indent": int, "tag": "ul"/"ol", "open_li": bool}, outermost first.
    stack: list[dict] = []
    para: list[str] = []

    def close_list() -> None:
        lvl = stack.pop()
        if lvl["open_li"]:
            out.append("</li>")
        out.append(f"</{lvl['tag']}>")

    def close_all_lists() -> None:
        while stack:
            close_list()

    def flush_para() -> None:
        if para:
            out.append("<p>" + "<br>".join(_inline(p) for p in para) + "</p>")
            para.clear()

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            # A blank line ends a paragraph but NOT a list: a "loose" list keeps
            # blank lines between items. The list is closed when a heading or a
            # non-list paragraph actually follows (or at EOF).
            flush_para()
            continue

        h = _H.match(line)
        if h:
            flush_para()
            close_all_lists()
            level = len(h.group(1))
            out.append(f"<h{level}>{_inline(h.group(2).strip())}</h{level}>")
            continue

        ul = _UL.match(line)
        ol = _OL.match(line)
        if ul or ol:
            flush_para()
            tag = "ul" if ul else "ol"
            indent = len((ul or ol).group(1).expandtabs(4))
            item = (ul or ol).group(2).strip()

            # Drop any list levels deeper than this line.
            while stack and stack[-1]["indent"] > indent:
                close_list()

            if stack and stack[-1]["indent"] == indent:
                if stack[-1]["tag"] != tag:
                    close_list()
                    out.append(f"<{tag}>")
                    stack.append({"indent": indent, "tag": tag, "open_li": False})
                elif stack[-1]["open_li"]:
                    out.append("</li>")
                    stack[-1]["open_li"] = False
            else:
                # Deeper than (or first of) the current level -> open a nested list.
                out.append(f"<{tag}>")
                stack.append({"indent": indent, "tag": tag, "open_li": False})

            out.append(f"<li>{_inline(item)}")
            stack[-1]["open_li"] = True
            continue

        close_all_lists()
        para.append(line.strip())

    flush_para()
    close_all_lists()
    return "\n".join(out)
