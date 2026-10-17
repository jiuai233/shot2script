import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from s2s.diarize.clustering import agglomerate, assign
from s2s.diarize.embeddings import unit, window_embedding
from s2s.diarize.pipeline import analyze, load_words, validate_records, word_switch, write_back


def record(i=1, start="00:00.0", end="00:01.0", **kwargs):
    return {"id": f"E01-S{i:03d}", "起": start, "止": end, "文本": "原句。",
            "声线": "E01-V9", "听辨": "清楚", **kwargs}


def embed(samples):
    return np.array([1, 0], dtype=np.float32) if np.mean(samples) > 0 else np.array([0, 1], dtype=np.float32)


def test_cluster_without_known_count():
    vectors = [unit(v) for v in [[1, 0, 0], [0.98, 0.02, 0], [0, 1, 0], [0, 0, 1]]]
    assert agglomerate(vectors, 0.8) == [[0, 1], [2], [3]]
    assert agglomerate([], 0.8) == []
    assert agglomerate(vectors[:1], 0.8) == [[0]]


def test_average_link_not_single_link():
    vectors = [np.array([np.cos(a), np.sin(a)]) for a in [0, 0.5, 1.0]]
    assert len(agglomerate(vectors, 0.8)) == 2


def test_short_reassigned_but_not_in_clustering():
    vectors = [unit(v) for v in [[1, 0], [0.98, 0.02], [0, 1]]]
    labels, scores, matrix = assign(vectors, [True, False, False], 0.8)
    assert labels == {0: 0, 1: 0}
    assert scores[1] > 0.99
    assert scores[2] == 0
    assert matrix == [[1.0]]


def test_empty_and_missing_vectors():
    assert assign([None], [True], 0.6) == ({}, [None], [])
    assert unit([0, 0]) is None
    assert unit([float("nan"), 1]) is None


def test_window_average():
    sizes = []
    def extractor(samples):
        sizes.append(len(samples))
        return embed(samples)
    result = window_embedding(np.ones(16000 * 7), extractor, 3)
    assert len(sizes) == 3
    assert min(sizes) > 16000 * 2
    np.testing.assert_allclose(result, [1, 0])


def test_short_and_silent_stay_pending():
    rows = [record(end="00:00.2")]
    result = analyze(rows, np.ones(3200), embed)
    assert result["句列表"][0]["待定"]
    assert result["句列表"][0]["声线"] == "待定"
    result = analyze(rows, np.zeros(3200), lambda x: None)
    assert result["句列表"][0]["相似度"] is None


def test_half_difference_flags_overlap():
    result = analyze([record(end="00:02.0")], np.r_[np.ones(16000), -np.ones(16000)], embed)
    row = result["句列表"][0]
    assert row["疑似重叠"]
    assert row["前后半句相似度"] == 0


def test_original_overlap_preserved():
    result = analyze([record(听辨="重叠")], np.ones(16000), embed)
    assert result["句列表"][0]["重叠依据"] == ["原记录重叠"]


def test_half_second_is_eligible():
    result = analyze([record(end="00:00.5")], np.ones(8000), embed)
    assert result["句列表"][0]["参与聚类"]
    assert not result["句列表"][0]["待定"]


def test_matrix_counts_and_unknown():
    rows = [record(end="00:01.0"), record(2, "00:01.0", "00:02.0"),
            record(3, "00:02.0", "00:02.2")]
    result = analyze(rows, np.r_[np.ones(16000), -np.ones(16000), np.ones(3200)], embed)
    assert [r["句数"] for r in result["类别"]] == [2, 1]
    assert result["类间相似度矩阵"] == [[1, 0], [0, 1]]
    assert not result["句列表"][2]["参与聚类"]
    assert not result["句列表"][2]["待定"]


@pytest.mark.parametrize("rows", [[record(), record()], [record(end="00:00.0")],
                                  [record(), record(2, id="E02-S002")]])
def test_invalid_records(rows):
    with pytest.raises(ValueError):
        validate_records(rows)


@pytest.mark.parametrize("kwargs", [{"threshold": float("nan")}, {"threshold": 2},
                                    {"threshold": -0.1}, {"overlap_threshold": -2}, {"window_seconds": 0.1}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        analyze([record()], np.ones(16000), embed, **kwargs)


def test_out_of_audio_rejected():
    with pytest.raises(ValueError, match="超出音频"):
        analyze([record(end="00:02.0")], np.ones(16000), embed)


def test_load_words_offsets_and_ignores_punctuation(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps({"window": [10, 12], "response": {"words": [
        {"text": "啊", "start": 0.1, "end": 0.3, "speaker": 1},
        {"text": "！", "start": 0.3, "end": 0.3, "speaker": 2},
        {"text": "你", "start": 0.4, "end": 0.6, "speaker": 2}]}}))
    words = load_words([path])
    assert len(words) == 2
    assert words[0]["start"] == 10.1
    assert word_switch(words, 10, 11)
    assert not word_switch(words, 10.35, 11)
    row = record(start="00:10.0", end="00:11.0")
    assert analyze([row], np.ones(176000), embed, words=words)["句列表"][0]["疑似重叠"]


def test_writeback_backup_and_no_text_time_changes(tmp_path):
    rows = [record()]
    target = tmp_path / "声音记录.json"
    raw = json.dumps(rows, ensure_ascii=False).encode()
    target.write_bytes(raw)
    result = analyze(rows, np.ones(16000), embed)
    result["句列表"][0]["疑似重叠"] = True
    info = write_back(target, rows, result)
    assert Path(info["备份"]).read_bytes() == raw
    updated = json.loads(target.read_text())
    assert updated == [{**rows[0], "声线": "E01-V1", "听辨": "重叠"}]
    assert rows[0]["声线"] == "E01-V9"


def test_writeback_pending_keeps_schema_valid_label(tmp_path):
    rows = [record(end="00:00.2")]
    target = tmp_path / "声音记录.json"
    target.write_text(json.dumps(rows))
    result = analyze(rows, np.ones(3200), embed)
    info = write_back(target, rows, result)
    assert json.loads(target.read_text()) == rows
    assert info["保留原声线的待定句"] == ["E01-S001"]


def test_writeback_conflict_no_backup(tmp_path):
    target = tmp_path / "声音记录.json"
    target.write_text("[]")
    with pytest.raises(ValueError, match="不同"):
        write_back(target, [record()], {})
    assert list(tmp_path.iterdir()) == [target]


def load_cli():
    path = Path(__file__).parents[1] / "tools" / "diarize.py"
    spec = importlib.util.spec_from_file_location("t3d_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_success_and_existing_output(tmp_path, monkeypatch):
    cli = load_cli()
    source = tmp_path / "records.json"
    source.write_text(json.dumps([record()]))
    output = tmp_path / "out.json"
    monkeypatch.setattr(cli, "run", lambda *a, **k: {"类别": [], "句列表": [], "运行": {}})
    assert cli.main(["sample.mp4", str(source), str(output)]) == 0
    assert json.loads(output.read_text())["类别"] == []
    assert cli.main(["sample.mp4", str(source), str(output)]) == 1


def test_word_switch_ignores_null_and_boundary():
    words = [{"start": 0, "end": 0.5, "speaker": 1},
             {"start": 0.5, "end": 1, "speaker": None},
             {"start": 1, "end": 2, "speaker": 2}]
    assert not word_switch(words, 0, 1)
    assert word_switch(words, 0, 2)


def test_writeback_bad_results_no_backup(tmp_path):
    target = tmp_path / "声音记录.json"
    rows = [record()]
    target.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="句编号"):
        write_back(target, rows, {"句列表": []})
    assert list(tmp_path.iterdir()) == [target]


def test_cli_failed_run_does_not_write(tmp_path, monkeypatch):
    cli = load_cli()
    source = tmp_path / "records.json"
    source.write_text("[]")
    output = tmp_path / "out.json"
    def failed(*args, **kwargs):
        raise FileNotFoundError("缺少声纹模型")
    monkeypatch.setattr(cli, "run", failed)
    assert cli.main(["sample.mp4", str(source), str(output)]) == 1
    assert not output.exists()


def test_decode_real_pcm(tmp_path):
    import wave
    from s2s.diarize.embeddings import decode_audio
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(np.full(8000, 1000, dtype="<i2").tobytes())
    decoded = decode_audio(path)
    assert decoded.shape == (8000, 1)
    np.testing.assert_allclose(decoded[:, 0], 1000 / 32768)
