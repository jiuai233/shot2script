#!/usr/bin/env python3
"""从视频运行单集生产流程，或记录主控步骤完成。"""
from s2s.dispatch.controllers import episode_main

if __name__ == '__main__':
    raise SystemExit(episode_main())
