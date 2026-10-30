"""Shared helper for section-scoped updates to ``experiments/results/dataset_stats.md``."""
from __future__ import annotations

from pathlib import Path


def update_stats_section(
    stats_path: Path,
    section_header: str,
    section_md: str,
) -> None:
    """Append or replace only the ``section_header`` section in ``stats_path``.

    Replaces from ``section_header`` up to (not including) the next ``## ``
    header or EOF, preserving all preceding and following sections byte-for-byte.
    """
    stats_path = Path(stats_path).expanduser().resolve()
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    if stats_path.exists():
        existing = stats_path.read_text(encoding="utf-8")
        start_idx = existing.find(section_header)
        if start_idx != -1:
            prefix = existing[:start_idx]
            rest = existing[start_idx + len(section_header) :]
            next_rel = rest.find("\n## ")
            if next_rel != -1:
                suffix = rest[next_rel + 1 :]
                updated = prefix + section_md.rstrip("\n") + "\n\n" + suffix
            else:
                updated = prefix + section_md
        else:
            updated = existing.rstrip() + "\n\n" + section_md
    else:
        updated = section_md

    stats_path.write_text(updated, encoding="utf-8")
