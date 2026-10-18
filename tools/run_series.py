#!/usr/bin/env python3
"""从连续集号视频运行全剧生产并交付 Markdown 与 Word。"""
from s2s.dispatch.controllers import series_main

if __name__ == '__main__':
    raise SystemExit(series_main())
