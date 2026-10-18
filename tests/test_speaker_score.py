"""已有标注区间子集评分；不替代正式整集 evaluate。"""
import copy
import json
from pathlib import Path

import pytest

from s2s.evaluate import validate_annotation
from s2s.speaker_score import main, score_speaker


def gold(*rows):
    return {'集': 1, '视频': 'sample.mp4', '校对完成': False,
            '人物': [{'id': h, '名称': h, '外观': h, '出现': []} for h in ('H01', 'H02')],
            '台词': [dict(id=f'L{i:03}', 起=a, 止=b, 文本='原话', 说话人=h, 形态=f, **extra)
                   for i, (a, b, h, f, extra) in enumerate(rows, 1)], '关键动作': []}


def sounds(*spans):
    return [dict(id=f'E01-S{i:03}', 起=a, 止=b, 文本='机器原话', 声线=f'E01-V{i}', 听辨='清楚')
            for i, (a, b) in enumerate(spans, 1)]


def result(*people):
    return {'对白归属': [dict(句=f'E01-S{i:03}', 说话人=p, 形态='画内', 确定度='确认')
                       for i, p in enumerate(people, 1)], '临时人物映射': []}


def metric(g, s, r, mapping=None, field='说话人'):
    return score_speaker(g, s, r, mapping)['指标'][field]


def test_same_label_split_invariance_and_duplicate_gold_union():
    g = gold(('00:00.0', '00:04.0', 'H01', '画内', {}),
             ('00:01.0', '00:03.0', 'H01', '画内', {}))
    whole = metric(g, sounds(('00:00.0', '00:04.0')), result('C001'), {'H01': 'C001'})
    split = metric(g, sounds(('00:00.0', '00:02.0'), ('00:02.0', '00:04.0')),
                   result('C001', 'C001'), {'C001': 'H01'})
    for m in (whole, split):
        assert m['分母秒数'] == m['正确秒数'] == 4
        assert m['有效正确覆盖率'] == m['确认精度'] == 1
        assert m['有效标注条数'] == 2


def test_one_machine_sentence_crosses_people_not_excluded():
    g = gold(('00:00.0', '00:02.0', 'H01', '画内', {}),
             ('00:02.0', '00:04.0', 'H02', '画内', {}))
    m = metric(g, sounds(('00:00.0', '00:04.0')), result('C001'), {'C001': 'H01'})
    assert (m['分母秒数'], m['正确秒数'], m['错误秒数']) == (4, 2, 2)
    assert m['确认精度'] == m['错归率'] == .5
    assert not m['冲突区间']


def test_one_gold_sentence_all_predictions_count_including_wrong_tail():
    g = gold(('00:00.0', '00:04.0', 'H01', '画内', {}))
    m = metric(g, sounds(('00:00.0', '00:01.0'), ('00:01.0', '00:04.0')),
               result('C001', 'C002'), {'C001': 'H01', 'C002': 'H02'})
    assert (m['正确秒数'], m['错误秒数'], m['有效正确覆盖率']) == (1, 3, .25)


def test_fields_missing_independently_and_null_speaker_does_not_hide_form():
    g = gold(('00:00.0', '00:01.0', None, '画内', {}),
             ('00:01.0', '00:03.0', 'H01', None, {}))
    s = sounds(('00:00.0', '00:01.0'), ('00:01.0', '00:03.0'))
    r = result(None, 'C001')
    scored = score_speaker(g, s, r, {'C001': 'H01'})['指标']
    assert scored['说话人']['分母秒数'] == 2
    assert scored['形态']['分母秒数'] == scored['形态']['正确秒数'] == 1
    assert scored['说话人']['排除条目'] == [{'来源': 'L001', '原因': ['空说话人']}]
    assert scored['形态']['排除条目'] == [{'来源': 'L002', '原因': ['空形态']}]


def test_zero_pending_conflicts_and_same_value_overlap_deduplicated():
    g = gold(('00:00.0', '00:04.0', 'H01', '画内', {}),
             ('00:01.0', '00:02.0', 'H02', '画内', {}),
             ('00:04.0', '00:04.0', 'H01', '画内', {}),
             ('00:05.0', '00:06.0', 'H01', '画内', {'待核对': '听不清'}))
    scored = score_speaker(g, sounds(('00:00.0', '00:06.0')), result('H01'), {'H01': 'H01'})['指标']
    m = scored['说话人']
    assert m['分母秒数'] == 3
    assert m['冲突秒数'] == 1
    assert m['冲突区间'][0]['来源'] == ['L001', 'L002']
    assert m['排除条目'] == [{'来源': 'L003', '原因': ['零时长']}, {'来源': 'L004', '原因': ['待核对']}]
    assert scored['形态']['分母秒数'] == 4
    assert scored['形态']['冲突秒数'] == 0


def test_uncovered_missing_attribution_and_doubt_keep_denominator():
    g = gold(('00:00.0', '00:04.0', 'H01', '画内', {}))
    s = sounds(('00:01.0', '00:02.0'), ('00:02.0', '00:03.0'), ('00:03.0', '00:04.0'))
    r = result('C001', 'C001', 'C001')
    r['对白归属'].pop(0)
    r['对白归属'][0]['确定度'] = '存疑'
    m = metric(g, s, r, {'C001': 'H01'})
    assert (m['分母秒数'], m['未确认秒数'], m['未覆盖秒数']) == (4, 3, 1)
    assert m['确认覆盖率'] == m['有效正确覆盖率'] == .25
    assert {d['原因'] for d in m['逐区间'] if d['状态'] == '未确认'} == {'缺声音句', '缺归属', '归属存疑'}


@pytest.mark.parametrize('field', ['说话人', '形态'])
def test_error_rate_uses_confirmed_time_not_diluted_by_doubt(field):
    g = gold(('00:00.0', '00:04.0', 'H01', '画内', {}))
    s = sounds(('00:00.0', '00:01.0'), ('00:01.0', '00:04.0'))
    r = result('C002', 'C001')
    r['对白归属'][1]['确定度'] = '存疑'
    r['对白归属'][0]['形态'] = '画外'
    m = metric(g, s, r, {'C001': 'H01', 'C002': 'H02'} if field == '说话人' else None, field)
    assert (m['错误秒数'], m['确认秒数'], m['未确认秒数']) == (1, 1, 3)
    assert m['错归率'] == 1
    assert m['确认覆盖率'] == .25
    assert m['确认精度'] == m['有效正确覆盖率'] == 0


@pytest.mark.parametrize('field', ['说话人', '形态'])
def test_no_confirmed_time_has_null_precision_and_error_rate(field):
    g = gold(('00:00.0', '00:04.0', 'H01', '画内', {}))
    r = result('C001')
    r['对白归属'][0]['确定度'] = '存疑'
    m = metric(g, sounds(('00:00.0', '00:04.0')), r, {'C001': 'H01'}, field)
    assert m['确认秒数'] == 0
    assert m['确认精度'] is m['错归率'] is None
    assert m['确认覆盖率'] == m['有效正确覆盖率'] == 0


@pytest.mark.parametrize('mapping,missing', [(None, 4), ({}, 4), ({'C001': 'H01'}, 2)])
def test_absent_partial_mapping_never_fabricates_overall_accuracy(mapping, missing):
    g = gold(('00:00.0', '00:04.0', 'H01', '画内', {}))
    m = metric(g, sounds(('00:00.0', '00:02.0'), ('00:02.0', '00:04.0')),
               result('C001', 'C002'), mapping)
    assert m['无法映射秒数'] == missing
    assert m['确认覆盖率'] == 1
    assert m['未确认秒数'] == 0
    assert m['确认精度'] is m['有效正确覆盖率'] is m['错归率'] is None
    assert m['正确秒数'] == 4 - missing


def test_form_is_independent_of_identity_mapping():
    g = gold(('00:00.0', '00:02.0', 'H01', '画内', {}))
    scored = score_speaker(g, sounds(('00:00.0', '00:02.0')), result('C001'))
    assert scored['指标']['形态']['有效正确覆盖率'] == 1
    assert scored['指标']['说话人']['有效正确覆盖率'] is None
    assert scored['人物映射']['缺失编号'] == ['C001']


def test_alias_chain_self_loop_and_real_loop():
    g = gold(('00:00.0', '00:02.0', 'H01', '画内', {}))
    s = sounds(('00:00.0', '00:02.0'))
    r = result('E01-P01-T1')
    r['临时人物映射'] = [dict(临时='E01-P01-T1', 映射='E01-P02-T1', 确定度='确认'),
                      dict(临时='E01-P02-T1', 映射='E01-P02-T1', 确定度='确认')]
    assert metric(g, s, r, {'E01-P02-T1': 'H01'})['正确秒数'] == 2
    r['临时人物映射'][1]['映射'] = 'E01-P01-T1'
    with pytest.raises(ValueError, match='循环'):
        score_speaker(g, s, r)


@pytest.mark.parametrize('confidences', [('存疑',), ('存疑', '确认', '确认'),
                                       ('确认', '存疑', '确认'), ('确认', '确认', '存疑')])
def test_independent_mapping_cannot_transfer_through_doubtful_alias_chain(confidences):
    g = gold(('00:00.0', '00:02.0', 'H01', '画内', {}))
    s = sounds(('00:00.0', '00:02.0'))
    r = result('C001')
    chain = [f'E01-P01-T{i + 1}' for i in range(len(confidences))] + ['C001']
    r['临时人物映射'] = [dict(临时=left, 映射=right, 确定度=confidence)
                      for left, right, confidence in zip(chain, chain[1:], confidences)]
    with pytest.raises(ValueError, match='人物映射.*存疑.*直接提供终端编号.*C001.*独立映射'):
        score_speaker(g, s, r, {chain[0]: 'H01'})
    assert metric(g, s, r, {'C001': 'H01'})['正确秒数'] == 2


@pytest.mark.parametrize('confidence', ['确认', '存疑'])
def test_independent_mapping_self_alias_still_allowed(confidence):
    g = gold(('00:00.0', '00:02.0', 'H01', '画内', {}))
    r = result('C001')
    r['临时人物映射'] = [dict(临时='C001', 映射='C001', 确定度=confidence)]
    scored = score_speaker(g, sounds(('00:00.0', '00:02.0')), r, {'C001': 'H01'})
    assert scored['人物映射']['归一化预测到人工'] == {'C001': 'H01'}
    assert scored['指标']['说话人']['正确秒数'] == (2 if confidence == '确认' else 0)


def test_independent_mapping_confirmed_chain_normalizes_and_rejects_conflicts():
    g = gold(('00:00.0', '00:02.0', 'H01', '画内', {}))
    s = sounds(('00:00.0', '00:02.0'))
    r = result('C001')
    r['临时人物映射'] = [dict(临时='E01-P01-T1', 映射='E01-P01-T2', 确定度='确认'),
                      dict(临时='E01-P01-T2', 映射='C001', 确定度='确认')]
    assert metric(g, s, r, {'E01-P01-T1': 'H01'})['正确秒数'] == 2
    with pytest.raises(ValueError, match='人物映射终端冲突'):
        score_speaker(g, s, r, {'E01-P01-T1': 'H01', 'C001': 'H02'})


def test_doubtful_alias_not_forced_and_null_is_unconfirmed():
    g = gold(('00:00.0', '00:02.0', 'H01', '画内', {}))
    s = sounds(('00:00.0', '00:02.0'))
    r = result('E01-P01-T1')
    r['临时人物映射'] = [dict(临时='E01-P01-T1', 映射='C001', 确定度='存疑')]
    assert metric(g, s, r, {'C001': 'H01'})['未确认秒数'] == 2
    assert metric(g, s, result(None), {'C001': 'H01'})['未确认秒数'] == 2


def test_only_same_sentence_blocker_affects_confirmation():
    g = gold(('00:00.0', '00:02.0', 'H01', '画内', {}))
    s = sounds(('00:00.0', '00:01.0'), ('00:01.0', '00:02.0'))
    r = result('C001', 'C001')
    r['待核对'] = [{'位置': {'镜头': 'E01-K001'}, '级别': '阻断'},
                 {'位置': {'句': 'E01-S001'}, '级别': '阻断'}]
    m = metric(g, s, r, {'C001': 'H01'})
    assert (m['未确认秒数'], m['正确秒数']) == (1, 1)


def test_form_conflict_does_not_remove_speaker_time():
    g = gold(('00:00.0', '00:03.0', 'H01', '画内', {}),
             ('00:01.0', '00:02.0', 'H01', '画外', {}))
    scored = score_speaker(g, sounds(('00:00.0', '00:03.0')), result('C001'), {'C001': 'H01'})
    assert scored['指标']['说话人']['分母秒数'] == 3
    assert scored['指标']['形态']['分母秒数'] == 2
    assert scored['指标']['形态']['冲突秒数'] == 1


def test_legacy_terminal_registration_gap_is_explicit():
    g = gold(('00:00.0', '00:02.0', 'H01', '画内', {}))
    scored = score_speaker(g, sounds(('00:00.0', '00:02.0')), result('E01-P01-T1'))
    assert scored['指标']['说话人']['无法映射秒数'] == 2
    assert '登记未验证' in scored['口径']['旧登记边界']
    assert scored['人物映射']['缺失编号'] == ['E01-P01-T1']


def test_unused_alias_cycle_is_rejected():
    r = result()
    r['临时人物映射'] = [dict(临时='E01-P01-T1', 映射='E01-P02-T1', 确定度='确认'),
                      dict(临时='E01-P02-T1', 映射='E01-P01-T1', 确定度='确认')]
    with pytest.raises(ValueError, match='循环'):
        score_speaker(gold(), [], r)


def test_two_stage_n_registered_and_same_episode():
    g = gold(('00:00.0', '00:02.0', 'H01', '画内', {}))
    s = sounds(('00:00.0', '00:02.0'))
    r = result('E01-N001')
    r.update(结果版本=2, 本集人物={'人物': [{'id': 'E01-N001'}]})
    assert metric(g, s, r, {'E01-N001': 'H01'})['正确秒数'] == 2
    r['本集人物']['人物'] = []
    with pytest.raises(ValueError, match='未登记'):
        score_speaker(g, s, r)


def test_two_stage_t_alias_must_reach_registered_terminal():
    g = gold(('00:00.0', '00:02.0', 'H01', '画内', {}))
    s = sounds(('00:00.0', '00:02.0'))
    r = result('E01-P01-T1')
    r.update(结果版本=2, 本集人物={'人物': [{'id': 'E01-N001'}]})
    with pytest.raises(ValueError, match='终端T'):
        score_speaker(g, s, r)
    r['临时人物映射'] = [dict(临时='E01-P01-T1', 映射='E01-N001', 确定度='确认')]
    assert metric(g, s, r, {'E01-N001': 'H01'})['正确秒数'] == 2
    r['本集人物']['人物'][0]['id'] = 'E02-N001'
    with pytest.raises(ValueError, match='非本集'):
        score_speaker(g, s, r)


@pytest.mark.parametrize('mutate', [
    lambda s, r: s.append(copy.deepcopy(s[0])),
    lambda s, r: r['对白归属'].append(copy.deepcopy(r['对白归属'][0])),
    lambda s, r: r['对白归属'][0].update(句='E01-S999'),
    lambda s, r: r['对白归属'][0].update(说话人='随意姓名'),
    lambda s, r: r['对白归属'][0].update(说话人='H99'),
    lambda s, r: r['对白归属'][0].update(说话人='E02-P01-T1'),
    lambda s, r: r['对白归属'][0].update(形态='猜测'),
    lambda s, r: r['临时人物映射'].extend([dict(临时='E01-P01-T1', 映射='C001'), dict(临时='E01-P01-T1', 映射='C002')]),
    lambda s, r: r['临时人物映射'].append(dict(临时='E01-P01-T1', 映射='未知')),
    lambda s, r: s.append({**s[0], 'id': 'E01-S002', '起': '00:01.0'}),
    lambda s, r: s[0].update(止='00:00.0'),
    lambda s, r: r['临时人物映射'].append(dict(临时='E01-P01-T1', 映射='C001')),
    lambda s, r: r.update(待核对=[{'位置': {'句': 'E01-S999'}, '级别': '阻断'}]),
])
def test_illegal_predictions_rejected_not_excluded(mutate):
    g = gold(('00:00.0', '00:02.0', 'H01', '画内', {}))
    s, r = sounds(('00:00.0', '00:02.0')), result('C001')
    mutate(s, r)
    with pytest.raises(ValueError):
        score_speaker(g, s, r)


@pytest.mark.parametrize('mapping', [{'C001': 'H99'}, {'H01': 'C001', 'H02': 'C001'}, {'姓名': 'H01'}])
def test_bad_mapping_rejected(mapping):
    with pytest.raises(ValueError, match='映射'):
        metric(gold(), [], result(), mapping)


def test_empty_denominator_null_rates():
    m = metric(gold(), [], result())
    assert m['分母秒数'] == 0
    assert all(m[k] is None for k in ('确认覆盖率', '确认精度', '有效正确覆盖率', '错归率'))


def test_explicit_subset_accepts_false_formal_still_rejects_no_mutation():
    g = gold(('00:00.0', '00:02.0', 'H01', '画内', {}))
    before = copy.deepcopy(g)
    scored = score_speaker(g, sounds(('00:00.0', '00:02.0')), result('C001'))
    assert scored['评测范围'] == '已有标注子集（非正式整集验收）'
    assert g == before
    with pytest.raises(ValueError, match='尚未人工校对'):
        validate_annotation(g, complete=True)


def test_cli_exclusive_output_and_source_unchanged(tmp_path, capsys):
    values = [sounds(('00:00.0', '00:02.0')), result('C001'),
              gold(('00:00.0', '00:02.0', 'H01', '画内', {})), {'C001': 'H01'}]
    paths = [tmp_path / name for name in ('声音.json', '结果.json', '标注.json', '映射.json')]
    for p, v in zip(paths, values):
        p.write_text(json.dumps(v, ensure_ascii=False), encoding='utf-8')
    before = [p.read_bytes() for p in paths]
    out = tmp_path / 'new' / '评分.json'
    args = [*map(str, paths[:3]), str(out), '--人物映射', str(paths[3])]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)['指标']['说话人']['正确秒数'] == 2
    report = json.loads(out.read_text())
    assert report['来源'] == {name: {'路径': str(path.resolve())}
                            for name, path in zip(('声音记录', '归属结果', '标注', '人物映射'), paths)}
    assert report['人物映射']['提供内容'] == values[3]
    assert [p.read_bytes() for p in paths] == before
    assert main(args) == 1
    assert '已存在' in capsys.readouterr().err


def test_cli_rejects_duplicate_json_keys_without_output(tmp_path, capsys):
    source = tmp_path / '重复.json'
    source.write_text('{"C001":"H01","C001":"H02"}', encoding='utf-8')
    out = tmp_path / '评分.json'
    assert main([str(source), str(source), str(source), str(out)]) == 1
    assert 'JSON字段重复' in capsys.readouterr().err
    assert not out.exists()


def test_real_source_remains_unchanged_and_empty_mapping_not_gold():
    root = Path(__file__).resolve().parents[1]
    p = root / 'eval/样本/E01/标注.json'
    before = p.read_bytes()
    annotation = json.loads(before)
    scored = score_speaker(annotation, [], result(), {})
    assert scored['指标']['说话人']['分母秒数'] > 0
    assert scored['指标']['说话人']['有效正确覆盖率'] is None
    assert p.read_bytes() == before
    assert annotation['校对完成'] is False
