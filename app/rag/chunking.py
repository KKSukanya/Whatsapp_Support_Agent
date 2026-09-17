"""
Chunking strategy: split by markdown headings first (## sections are
semantically coherent policy answers), then hard-wrap anything still too
long. This matters more than it sounds — naive fixed-size chunking on
policy docs regularly splits a rule from its exception clause, which is a
classic "why did retrieval return a misleading half-answer" bug.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    tenant_id: str
    heading: str


def chunk_markdown(text: str, source: str, tenant_id: str, max_chars: int = 600) -> list[Chunk]:
    # Split on markdown headings (# / ## / **bold-as-heading** used in FAQs)
    parts = re.split(r"\n(?=#{1,3}\s)|\n(?=\*\*.+\*\*\n)", text)
    chunks: list[Chunk] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        heading_match = re.match(r"#{1,3}\s*(.+)", part) or re.match(r"\*\*(.+?)\*\*", part)
        heading = heading_match.group(1).strip() if heading_match else source

        if len(part) <= max_chars:
            chunks.append(Chunk(text=part, source=source, tenant_id=tenant_id, heading=heading))
            continue

        # Hard-wrap oversized sections on paragraph boundaries
        buf = ""
        for para in part.split("\n\n"):
            if len(buf) + len(para) > max_chars and buf:
                chunks.append(Chunk(text=buf.strip(), source=source, tenant_id=tenant_id, heading=heading))
                buf = ""
            buf += para + "\n\n"
        if buf.strip():
            chunks.append(Chunk(text=buf.strip(), source=source, tenant_id=tenant_id, heading=heading))

    return chunks
