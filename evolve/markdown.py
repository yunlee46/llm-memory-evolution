"""Split markdown into heading-delimited sections and reassemble. Used by crossover and mutation."""
from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.M)


@dataclass
class Section:
    title: str          # heading text without '#'; "" for the preamble
    level: int          # 0 for preamble
    body: str           # text under the heading, without the heading line

    @property
    def key(self) -> str:
        return re.sub(r"[^a-z0-9]+", " ", self.title.lower()).strip()

    def render(self) -> str:
        head = f"{'#' * self.level} {self.title}\n" if self.level else ""
        return head + self.body.strip("\n") + "\n"


def split_sections(md: str) -> list[Section]:
    """Sections at level 1-2 headings; level-3 headings stay inside their parent body."""
    sections: list[Section] = []
    pos = 0
    cur = Section("", 0, "")
    for m in _HEADING.finditer(md):
        level = len(m.group(1))
        if level > 2:
            continue
        cur.body = md[pos:m.start()]
        if cur.title or cur.body.strip():
            sections.append(cur)
        cur = Section(m.group(2), level, "")
        pos = m.end()
    cur.body = md[pos:]
    if cur.title or cur.body.strip():
        sections.append(cur)
    if not sections:
        sections.append(Section("", 0, md))
    return sections


def join_sections(sections: list[Section]) -> str:
    return "\n".join(s.render() for s in sections).strip() + "\n"


def word_count(md: str) -> int:
    return len(re.findall(r"\S+", md))
