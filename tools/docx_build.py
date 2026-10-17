#!/usr/bin/env python3
"""用法：uv run python tools/docx_build.py <合并稿.md> <输出.docx> <标题>。"""

from s2s.docx_build import main


if __name__ == "__main__":
    raise SystemExit(main())
