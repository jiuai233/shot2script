import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from s2s.lipsync import Parameters
from s2s.lipsync.core import combine_scores, group_tracks, score_samples
from s2s.lipsync.detector import Supplement, add_pixel_motion, iou
from s2s.lipsync.pipeline import review_sentence, render_html


@pytest.mark.parametrize('kwargs', [{'model_weight': -1}, {'model_weight': 1.1}, {'conflict_gap': 0}, {'pixel_motion_scale': 0}])
def test_new_parameters(kwargs):
    with pytest.raises(ValueError):
        Parameters(**kwargs)


def test_fusion_weights_conflict_model_only_and_missing_scores():
    p = Parameters()
    assert combine_scores(.2, '未开口', .98, 25, 'fused', p)[1:] == ('存疑', True)
    assert combine_scores(.65, '存疑', .95, 25, 'fused', p) == (.845, '开口', False)
    assert combine_scores(.2, '未开口', .95, 25, 'model', p) == (.95, '开口', False)
    assert combine_scores(.7, '开口', None, 0, 'fused', p)[1] == '存疑'
    assert combine_scores(.7, '看不到', .99, 25, 'fused', p)[1] == '看不到'
    assert combine_scores(.7, '开口', .99, 3, 'fused', p)[1] == '存疑'
    assert combine_scores(.7, '开口', .99, 25, 'fused', Parameters(model_weight=0))[0] == .7
    assert combine_scores(.7, '开口', .99, 25, 'fused', Parameters(model_weight=1))[0] == .99


def test_yunet_supplement_on_both_real_profile_frames():
    pytest.importorskip('mediapipe')
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    root = Path(__file__).resolve().parents[1]
    if not (root / 'tmp/T11/probe_55.jpg').exists():
        pytest.skip('无 E01 探测帧')
    d = Supplement(root / 'models/lipsync/face_detection_yunet_2023mar.onnx')
    opts = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(root / 'models/lipsync/face_landmarker.task')),
                                output_facial_transformation_matrixes=True,
                                min_face_detection_confidence=.35, min_face_presence_confidence=.35)
    with FaceLandmarker.create_from_options(opts) as landmarker:
        for t in [21.5, 55]:
            image = cv2.imread(str(root / f'tmp/T11/probe_{t}.jpg'))
            original = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            assert not original.face_landmarks
            faces = d.detect(image, [], landmarker, Parameters())
            assert faces and all(f['source'].startswith('YuNet') for f in faces)
            assert any(f['valid'] for f in faces)
            for face in faces:
                json.dumps({k: face.get(k) for k in ("box", "mar", "mouth_box", "source")})
            count = len(faces)
            assert len(d.detect(image, faces, landmarker, Parameters())) == count


def test_pixel_motion_is_not_fabricated_lip_ratio_or_identity():
    samples = []
    for i in range(12):
        samples.append({'time': i / 12, 'source': 'YuNet嘴部像素', 'valid': True, 'mar': None,
                        'feature': None, '_mouth': np.full((40, 64), .2 * (i % 2)),
                        '_reference': np.zeros((40, 64))})
    tracks = [{'samples': samples, 'shot': 'K1'}]
    add_pixel_motion(tracks)
    assert samples[0]['mouth_motion'] is None
    assert samples[1]['mouth_motion'] == pytest.approx(.2)
    assert '_mouth' not in samples[0]
    score, status, evidence = score_samples(samples, [], lambda _: 0, Parameters())
    assert score == .65 and status == '存疑'
    assert '非张合比' in evidence['测量方式']
    group_tracks(tracks, Parameters())
    assert tracks[0]['group'] is None
    assert all(s['mar'] is None for s in samples)


def test_structured_review_does_not_modify_code_result():
    row = {'句': 'E01-S007', '起': '00:10.0', '止': '00:11.0', '文本': '你好', '结论': '存疑'}
    original = copy.deepcopy(row)
    calls = []
    def ask(images, prompt, **kwargs):
        calls.append(kwargs)
        return '```json\n{"画面结论":"画内","依据":"右侧嘴动"}\n```'
    result = review_sentence(row, [str(i) for i in range(6)], ask)
    assert result['与代码一致'] is False and result['画面结论'] == '画内'
    assert row == original and calls == [{}]


def test_four_version_html_escapes_comparison(tmp_path):
    image = tmp_path / 'frame.jpg'
    cv2.imwrite(str(image), np.zeros((100, 100, 3), dtype=np.uint8))
    row = {'句': 'E01-S001', '结论': '存疑', '起': '00:00.0', '止': '00:01.0', '声线': None,
           '文本': '你好', '代表帧': str(image), '代表时间秒': 0, '轨迹': [], '复核': None,
           '版本结论': {'T11原版': '看不到', '加侧脸检测': '未开口', '加主动说话人模型': '画内', '融合后': '存疑'}}
    html = render_html({'句': [row]}, {})
    assert all(label in html for label in row['版本结论'])
    assert html.count('data:image/jpeg;base64,') == 1
    assert iou([0, 0, 1, 1], [.5, 0, .5, 1]) == .5


def test_worker_real_cpu_fixture(tmp_path):
    import subprocess
    root = Path(__file__).resolve().parents[1]
    python = root / 'envs/asd/bin/python'
    manifest = root / 'tmp/T11b/真实CPU探测输入.json'
    if not python.exists() or not manifest.exists():
        pytest.skip('无独立 ASD 环境或真实输入')
    output = tmp_path / 'result.json'
    process = subprocess.run([str(python), str(root / 'tools/s2s/lipsync/asd_worker.py'), str(manifest), str(output),
                              '--weights', str(root / 'models/lipsync/Light-ASD/pretrain_AVA_CVPR.model')],
                             text=True, capture_output=True, timeout=30)
    assert process.returncode == 0, process.stderr
    record = json.loads(output.read_text())
    frames = next(iter(record['tracks'].values()))['frames']
    assert len(frames) >= 20 and all(0 <= f['probability'] <= 1 for f in frames)
    assert np.diff([f['time'] for f in frames]) == pytest.approx(.04)
    assert record['peak_mib'] > 0


def test_review_existing_checks_every_doubtful_sentence_and_saves_comparison(tmp_path, monkeypatch):
    from s2s import vision
    from s2s.lipsync.pipeline import review_existing
    evidence = tmp_path / 'evidence'
    (evidence / 'frames').mkdir(parents=True)
    frames = []
    for i in range(8):
        name = f'frame{i}.jpg'
        cv2.imwrite(str(evidence / 'frames' / name), np.zeros((64, 64, 3), dtype=np.uint8))
        frames.append({'文件': name, '时间': f'00:0{i // 4}.{i % 4 * 2}'})
    (evidence / 'K1_帧.json').write_text(json.dumps({'帧': frames}))
    (evidence / '轨迹.json').write_text('[]')
    rows = [{'句': f'E01-S00{i+1}', '起': f'00:0{i}.0', '止': f'00:0{i+1}.0',
             '文本': '测试', '结论': '存疑', '声线': None, '轨迹': [], '复核': None,
             '代表帧': str(evidence / 'frames' / frames[i * 4]['文件']), '代表时间秒': i} for i in range(2)]
    source = tmp_path / 'source.json'
    record = {'句': rows, '证据目录': str(evidence), '统计': {}}
    source.write_text(json.dumps(record))
    calls = []
    def ask(images, prompt, **kwargs):
        assert kwargs == {} and len(images) == 4
        calls.append(images)
        return json.dumps({'画面结论': '存疑' if len(calls) == 1 else '画内', '依据': '测试'})
    monkeypatch.setattr(vision, 'ask', ask)
    target = tmp_path / 'review' / '口型记录.json'
    result = review_existing(source, target)
    assert result['复核统计']['一致'] == result['复核统计']['不一致'] == 1
    assert result['复核统计']['已复核'] == 2 and result['复核统计']['无法比较'] == 0
    assert [r['结论'] for r in result['句']] == ['存疑', '存疑']
    assert json.loads(source.read_text()) == record
    assert target.with_name('口型核对.html').is_file()


def test_shared_decoding_matches_seek_on_real_tracks(tmp_path):
    import subprocess
    root = Path(__file__).resolve().parents[1]
    python = root / 'envs/asd/bin/python'
    source = root / 'tmp/T11b/profile/输入.json'
    if not python.exists() or not source.exists():
        pytest.skip('无独立环境或分段输入')
    results = []
    for mode in ['seek', 'shared']:
        target = tmp_path / f'{mode}.json'
        run = subprocess.run([str(python), str(root / 'tools/s2s/lipsync/asd_worker.py'), str(source), str(target),
                              '--weights', str(root / 'models/lipsync/Light-ASD/pretrain_AVA_CVPR.model'),
                              '--decode-mode', mode], capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, run.stderr
        results.append(json.loads(target.read_text()))
    assert results[0]['tracks'] == results[1]['tracks']
    assert len(results[1]['timings']) == 3
    assert results[1]['decoding']['requested_frames'] == 110


def test_asd_reuse_rejects_mismatched_input(tmp_path):
    from s2s.lipsync.asd import run_asd
    old, new = tmp_path / 'old', tmp_path / 'new'
    old.mkdir(); new.mkdir()
    (old / 'asd输入.json').write_text('{}')
    (old / 'asd输出.json').write_text('{}')
    with pytest.raises(ValueError, match='输入与当前轨迹不一致'):
        run_asd('video', 'audio', [], {'镜头': []}, 12, new, 'unused', 'weights', result_source=old / 'asd输出.json')


def test_rescore_reuses_rule_evidence_and_preserves_source(tmp_path, monkeypatch):
    from dataclasses import asdict
    from s2s.lipsync.pipeline import rescore_existing
    from s2s.lipsync import asd
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    (evidence / '轨迹.json').write_text(json.dumps([{'id': 'K1-F1', '镜头': 'K1', '样本': [
        {'时间': f'00:00.{i}00', '框': [0, 0, 1, 1], '有效': True} for i in range(4)]}]))
    (evidence / '镜头表.json').write_text(json.dumps({'镜头': [{'id': 'K1', '止': '00:01.0'}]}))
    image = evidence / 'image.jpg'
    cv2.imwrite(str(image), np.zeros((64, 64, 3), np.uint8))
    source = tmp_path / 'side.json'
    row = {'句': 'S1', '起': '00:00.0', '止': '00:01.0', '文本': '测试', '结论': '未开口', '声线': None,
           '代表帧': str(image), '代表时间秒': 0, '复核': None, '轨迹': [
        {'id': 'K1-F1', '规则分数': .1, '说话分数': .1, '轨迹组': None, '状态': '未开口',
         '评分依据': {'规则状态': '未开口'}}]}
    source.write_text(json.dumps({'版本': 'side', '参数': {**asdict(Parameters()), '阈值': .6},
        '句': [row], '证据目录': str(evidence), '统计': {'耗时秒': 10, '进程内存峰值MiB': 1}}))
    original = source.read_bytes()
    monkeypatch.setattr(asd, 'run_asd', lambda *a, **k: {'tracks': {'K1-F1': {'frames': [
        {'time': i / 25, 'probability': .99} for i in range(25)]}}, 'subprocess_seconds': 1, 'peak_mib': 2})
    result = rescore_existing('unused', source, tmp_path / 'result' / 'record.json', 'fused', 'python', 'weights')
    assert result['句'][0]['结论'] == '存疑'
    assert result['句'][0]['轨迹'][0]['评分依据']['评分冲突'] is True
    assert result['句'][0]['轨迹'][0]['规则分数'] == .1
    assert result['统计']['耗时秒'] >= 10
    assert source.read_bytes() == original


def test_review_parses_fenced_json_with_surrounding_prose():
    from s2s.lipsync.pipeline import parse_review_answer
    answer = '观察说明\n```json\n{"画面结论":"存疑","依据":"看不清"}\n```\n后续建议'
    assert parse_review_answer(answer, '存疑') == {'画面结论': '存疑', '与代码一致': True}
    assert parse_review_answer('无法确认', '存疑') == {}


def test_review_resume_only_calls_missing_short_sentence(tmp_path, monkeypatch):
    from s2s import vision
    from s2s.lipsync.pipeline import review_existing
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    (evidence / '轨迹.json').write_text('[]')
    video = tmp_path / 'short.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 64))
    for i in range(30):
        writer.write(np.full((64, 64, 3), i * 6, np.uint8))
    writer.release()
    image = evidence / 'representative.jpg'
    cv2.imwrite(str(image), np.zeros((64, 64, 3), np.uint8))
    rows = [{'句': f'S{i}', '起': '00:00.0', '止': '00:00.2', '文本': '短句', '结论': '存疑',
             '声线': None, '轨迹': [], '代表帧': str(image), '代表时间秒': 0,
             '复核': {'状态': '已复核', '结果': '```json\n{"画面结论":"存疑"}\n```\n补充'} if i == 0 else {'状态': '未调用'}} for i in range(2)]
    source = tmp_path / 'source.json'
    source.write_text(json.dumps({'句': rows, '统计': {}, '证据目录': str(evidence), '复核统计': {'耗时秒': 10}}))
    calls = []
    def ask(images, prompt, **kwargs):
        assert kwargs == {} and len(images) == 6 and len(set(images)) == 6
        calls.append(images)
        return '{"画面结论":"画内"}'
    monkeypatch.setattr(vision, 'ask', ask)
    result = review_existing(source, tmp_path / 'output' / 'record.json', video)
    assert len(calls) == 1
    assert result['复核统计']['已复核'] == 2
    assert result['复核统计']['一致'] == result['复核统计']['不一致'] == 1
    assert result['复核统计']['本次调用'] == result['复核统计']['复用已调用'] == 1
    assert result['复核统计']['耗时秒'] >= 10
    assert Path(result['句'][1]['复核']['补帧索引']).exists()
