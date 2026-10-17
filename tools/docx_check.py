#!/usr/bin/env python3
"""用法：uv run python tools/docx_check.py <docx> <总集数> [--缺失集 3,5]。"""

from s2s.docx_check import main


if __name__ == "__main__":
    raise SystemExit(main())
