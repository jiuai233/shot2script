"""主流程与独立 PyTorch 子进程之间的文件接口。"""

import json
from pathlib import Path
import subprocess
import time


def run_asd(video, audio, tracks, shots, fps, folder, python, weights, timeout=1200, result_source=None):
    ends = {s["id"]: s["止"] for s in shots["镜头"]}
    from ..timefmt import to_sec

    manifest = {"video": str(Path(video).resolve()), "audio": str(Path(audio).resolve()), "fps": fps,
                "tracks": [{"id": t["id"], "shot_end": to_sec(ends[t["shot"]]),
                            "samples": [{"time": s["time"], "box": s["box"]} for s in t["samples"] if s["valid"]]}
                           for t in tracks if any(s["valid"] for s in t["samples"])]}
    source, target = folder / "asd输入.json", folder / "asd输出.json"
    if result_source is not None:
        existing = Path(result_source)
        if json.loads(existing.with_name("asd输入.json").read_text()) != manifest:
            raise ValueError("复用 ASD 结果的输入与当前轨迹不一致")
        result = json.loads(existing.read_text())
        if result.get("weights") != str(Path(weights).resolve()):
            raise ValueError("复用 ASD 结果的权重路径不一致")
        result["reused_from"] = str(existing)
        result["original_subprocess_seconds"] = result["subprocess_seconds"]
        result["subprocess_seconds"] = 0.
        source.write_text(json.dumps(manifest))
        target.write_text(json.dumps(result, ensure_ascii=False))
        return result
    source.write_text(json.dumps(manifest))
    started = time.perf_counter()
    command = [str(python), str(Path(__file__).with_name("asd_worker.py")), str(source), str(target), "--weights", str(weights)]
    try:
        process = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Light-ASD 子进程超过 {timeout} 秒，已停止；未自动重试") from exc
    (folder / "asd运行.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError(f"Light-ASD 子进程退出 {process.returncode}，详见 {folder / 'asd运行.log'}")
    result = json.loads(target.read_text())
    result["subprocess_seconds"] = time.perf_counter() - started
    target.write_text(json.dumps(result, ensure_ascii=False))
    return result
