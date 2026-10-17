import copy
import json

import pytest

from s2s.board import initialize_board
from s2s.dispatch.roles import dispatch


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')


@pytest.fixture
def episode(tmp_path):
    initialize_board('测试剧', 1, root=tmp_path)
    ep = tmp_path / '测试剧/E01'
    shots, segments = [], []
    for n in (1, 2):
        shot = f'E01-K{n:03d}'
        filename = f'K{n:03d}_01.jpg'
        shots.append({'id': shot, '起': f'00:0{n-1}.0', '止': f'00:0{n}.0',
                      '帧': [{'文件': filename, '时间': f'00:0{n-1}.1'}]})
        segments.append({'id': f'E01-P{n:02d}', '起': shots[-1]['起'], '止': shots[-1]['止'],
                         '镜头': [shot], '衔接前': 'E01-K001' if n == 2 else None,
                         '衔接后': 'E01-K002' if n == 1 else None})
        (ep / '素材/frames' / filename).write_bytes(b'mock-image')
        people = [{'人物': 'E01-P01-T1', '外观': '紫衣双马尾', '依据帧': [filename]}] if n == 1 else []
        save(ep / f'派发结果/动作识别/E01-P{n:02d}.json',
             {'片段': f'E01-P{n:02d}', '镜头': [{'镜头': shot, '画面人物': people,
              '动作': [], '神态': [], '口型': [], '说明': '动作叙述隔离哨兵'}]})
    save(ep / '镜头表.json', {'集': 1, '视频': 'mock.mp4', '时长': '00:02.0', '镜头': shots})
    save(ep / '片段表.json', {'集': 1, '片段': segments})
    save(ep / '临时人物.json', [{'id': 'E01-P01-T1', '片段': 'E01-P01', '外观': '紫衣双马尾',
          '出现镜头': ['E01-K001'], '映射': None, '映射依据': ''}])
    save(ep / '声音记录.json', [{'id': 'E01-S001', '起': '00:00.5', '止': '00:01.5',
          '文本': '声音原文哨兵', '声线': 'E01-V1', '听辨': '清楚'}])
    save(ep / '字幕记录.json', [{'起': '00:00.0', '止': '00:02.0', '文字': '台词字幕隔离哨兵', '类型': '台词字幕'}])
    return ep


def merge_result():
    return {'分组': [{'成员': ['E01-P01-T1'], '已有角色': None, '名称': '穿紫衣的双马尾女孩',
            '名称类型': '描述名', '外观': '紫衣双马尾', '依据': '代表原帧可见紫衣双马尾',
            '确定度': '确认', '待核对': []}]}


def attribution_result(speaker='E01-N001', uncertain=False):
    return {'对白归属': [{'句': 'E01-S001', '说话人': speaker, '形态': '画内',
                         '依据': '整句画面嘴部持续开合', '确定度': '存疑' if uncertain else '确认'}],
            '补充建议': [], '待核对': [{'位置': {'句': 'E01-S001'}, '问题': '身份待核实', '级别': '阻断'}] if uncertain else []}


def board_bytes(ep):
    return {str(p.relative_to(ep.parent)): p.read_bytes() for p in [*ep.glob('*.json'), ep.parent / '角色卡.json']
            if p.name != '派发日志.jsonl'}


def test_single_new_person_cross_segment_sentence(episode, tmp_path):
    before = board_bytes(episode)
    calls = []

    def call(role, prompt, **kwargs):
        calls.append((prompt, kwargs))
        assert '台词字幕隔离哨兵' not in prompt
        assert '动作叙述隔离哨兵' not in prompt
        if '本次阶段：人物合并' in prompt:
            assert '声音原文哨兵' not in prompt
            assert len(kwargs['images']) == 1
            return json.dumps(merge_result(), ensure_ascii=False)
        assert '声音原文哨兵' in prompt and 'E01-N001' in prompt
        assert {p.name for p in kwargs['images']} == {'K001_01.jpg', 'K002_01.jpg'}
        return json.dumps(attribution_result(), ensure_ascii=False)

    output = tmp_path / 'result.json'
    assert dispatch('角色卡', episode, {}, output, mode='建立', call=call)['通过']
    result = json.loads(output.read_text())
    assert len(calls) == 2
    assert result['结果版本'] == 2
    assert result['本集人物']['人物'][0]['id'] == 'E01-N001'
    assert result['临时人物映射'][0]['映射'] == 'E01-N001'
    assert result['对白归属'][0]['说话人'] == 'E01-N001'
    assert board_bytes(episode) == before


@pytest.mark.parametrize('failed_stage', ['人物合并', '对白归属'])
def test_stage_failure_preserves_previous_output_and_board(episode, tmp_path, failed_stage):
    before = board_bytes(episode)
    output = tmp_path / 'result.json'
    output.write_text('old-result', encoding='utf-8')

    def call(role, prompt, **kwargs):
        if f'本次阶段：{failed_stage}' in prompt:
            return '{}'
        return json.dumps(merge_result())

    with pytest.raises((ValueError, RuntimeError)):
        dispatch('角色卡', episode, {}, output, mode='建立', call=call)
    assert output.read_text() == 'old-result'
    assert board_bytes(episode) == before


def confirmed_role(ident='C007'):
    return {'id': ident, '名称': '穿紫衣的双马尾女孩', '名称类型': '描述名', '状态': '已确认',
            '别名称呼': [], '外观': [], '声线': [], '关系': [], '身份依据': ['已审核'],
            '性格总结': [], '首次出场': 'E01-K001', '待核对': []}


def clear_people(ep):
    save(ep / '临时人物.json', [])
    path = ep / '派发结果/动作识别/E01-P01.json'
    result = json.loads(path.read_text())
    result['镜头'][0]['画面人物'] = []
    save(path, result)


@pytest.mark.parametrize('with_sound', [False, True])
def test_zero_people_skips_merge_existing_c_can_speak(episode, tmp_path, with_sound):
    clear_people(episode)
    save(episode.parent / '角色卡.json', [confirmed_role()])
    if not with_sound:
        save(episode / '声音记录.json', [])
    calls = []

    def call(role, prompt, **kwargs):
        calls.append(prompt)
        assert '本次阶段：对白归属' in prompt
        assert 'C007' in prompt
        return json.dumps(attribution_result('C007'))

    output = tmp_path / 'result.json'
    dispatch('角色卡', episode, {}, output, mode='建立', call=call)
    assert len(calls) == int(with_sound)
    result = json.loads(output.read_text())
    assert result['本集人物']['人物'] == [] and result['合并结果'] == {'分组': []}


@pytest.mark.parametrize('uncertain', [False, True])
def test_existing_c_match_only_adopted_when_unambiguous(episode, tmp_path, uncertain):
    save(episode.parent / '角色卡.json', [confirmed_role()])
    merge = merge_result()
    merge['分组'][0].update({'已有角色': 'C007', '确定度': '存疑' if uncertain else '确认',
                            '待核对': ['需复看'] if uncertain else []})
    output = tmp_path / 'result.json'
    dispatch('角色卡', episode, {}, output, call=lambda role, prompt, **kw: json.dumps(
        merge if '本次阶段：人物合并' in prompt else attribution_result('E01-N001' if uncertain else 'C007')))
    result = json.loads(output.read_text())
    assert result['临时人物映射'][0]['映射'] == ('E01-N001' if uncertain else 'C007')
    assert len(result['本集人物']['人物']) == int(uncertain)


def test_unknown_speaker_null_with_same_sentence_blocker(episode, tmp_path):
    output = tmp_path / 'result.json'
    dispatch('角色卡', episode, {}, output, call=lambda role, prompt, **kw: json.dumps(
        merge_result() if '本次阶段：人物合并' in prompt else attribution_result(None, True)))
    assert json.loads(output.read_text())['对白归属'][0]['说话人'] is None


@pytest.mark.parametrize('invalid', ['missing_member', 'duplicate_member', 'invented_c', 'invented_n',
                                    'missing_sentence', 'duplicate_sentence', 'extra_sentence',
                                    'temporary_speaker', 'cross_episode_n', 'blank_basis', 'null_confirmed', 'no_blocker'])
def test_illegal_stage_results_never_publish(episode, tmp_path, invalid):
    merge, attribution = merge_result(), attribution_result()
    if invalid == 'missing_member':
        merge['分组'] = []
    elif invalid == 'duplicate_member':
        merge['分组'].append(copy.deepcopy(merge['分组'][0]))
    elif invalid == 'invented_c':
        merge['分组'][0]['已有角色'] = 'C999'
    elif invalid == 'invented_n':
        merge['分组'][0]['id'] = 'E01-N999'
    elif invalid == 'missing_sentence':
        attribution['对白归属'] = []
    elif invalid == 'duplicate_sentence':
        attribution['对白归属'] *= 2
    elif invalid == 'extra_sentence':
        attribution['对白归属'].append({**attribution['对白归属'][0], '句': 'E01-S002'})
    elif invalid == 'temporary_speaker':
        attribution['对白归属'][0]['说话人'] = 'E01-P01-T1'
    elif invalid == 'cross_episode_n':
        attribution['对白归属'][0]['说话人'] = 'E02-N001'
    elif invalid == 'blank_basis':
        attribution['对白归属'][0]['依据'] = '  '
    elif invalid == 'null_confirmed':
        attribution['对白归属'][0]['说话人'] = None
    else:
        attribution['对白归属'][0]['确定度'] = '存疑'
    before = board_bytes(episode)
    output = tmp_path / 'result.json'
    with pytest.raises((ValueError, RuntimeError)):
        dispatch('角色卡', episode, {}, output, call=lambda role, prompt, **kw: json.dumps(
                 merge if '本次阶段：人物合并' in prompt else attribution))
    assert not output.exists() and board_bytes(episode) == before


@pytest.mark.parametrize('invalid', ['missing_candidate', 'missing_file', 'missing_declaration', 'appearance_mismatch'])
def test_input_preflight_before_any_model(episode, tmp_path, invalid):
    path = episode / '派发结果/动作识别/E01-P01.json'
    result = json.loads(path.read_text())
    if invalid == 'missing_candidate':
        result['镜头'][0]['画面人物'][0]['依据帧'] = []
    elif invalid == 'missing_file':
        (episode / '素材/frames/K001_01.jpg').unlink()
    elif invalid == 'missing_declaration':
        result['镜头'][0]['画面人物'] = []
    else:
        result['镜头'][0]['画面人物'][0]['外观'] = '不一致'
    save(path, result)
    calls = []
    with pytest.raises(ValueError):
        dispatch('角色卡', episode, {}, tmp_path / 'result.json', call=lambda *a, **kw: calls.append(a))
    assert calls == []


@pytest.mark.parametrize('invalid', ['cross_episode', 'missing_segment', 'connection_shot'])
def test_declaration_segment_preflight_rejects_before_model_and_keeps_old_output(episode, tmp_path, invalid):
    path = episode / '派发结果/动作识别/E01-P01.json'
    result = json.loads(path.read_text())
    people = json.loads((episode / '临时人物.json').read_text())
    if invalid == 'connection_shot':
        result['镜头'][0]['镜头'] = 'E01-K002'
        result['镜头'][0]['画面人物'][0]['依据帧'] = ['K002_01.jpg']
        people[0]['出现镜头'] = ['E01-K002']
    else:
        segment = 'E02-P01' if invalid == 'cross_episode' else 'E01-P99'
        result['片段'] = segment
        result['镜头'][0]['画面人物'][0]['人物'] = segment + '-T1'
        people[0].update({'id': segment + '-T1', '片段': segment})
    save(path, result)
    save(episode / '临时人物.json', people)
    before = board_bytes(episode)
    output = tmp_path / 'result.json'
    output.write_text('old-result')
    calls = []

    def call(role, prompt, **kwargs):
        calls.append(prompt)
        merge = merge_result()
        merge['分组'][0]['成员'] = [people[0]['id']]
        return json.dumps(merge if '本次阶段：人物合并' in prompt else attribution_result())

    with pytest.raises(ValueError, match='片段|镜头'):
        dispatch('角色卡', episode, {}, output, call=call)
    assert calls == []
    assert output.read_text() == 'old-result'
    assert board_bytes(episode) == before


def test_shared_picture_keeps_all_people_and_numbering_is_deterministic(episode, tmp_path):
    people = json.loads((episode / '临时人物.json').read_text())
    people.append({**people[0], 'id': 'E01-P01-T2', '外观': '蓝衣短发'})
    save(episode / '临时人物.json', people)
    path = episode / '派发结果/动作识别/E01-P01.json'
    actions = json.loads(path.read_text())
    actions['镜头'][0]['画面人物'].append({'人物': 'E01-P01-T2', '外观': '蓝衣短发', '依据帧': ['K001_01.jpg']})
    save(path, actions)
    merge = merge_result()
    merge['分组'].append({**merge['分组'][0], '成员': ['E01-P01-T2'], '名称': '穿蓝衣的短发女孩'})
    outputs = []
    for groups in (merge['分组'], list(reversed(merge['分组']))):
        def call(role, prompt, **kwargs):
            if '本次阶段：人物合并' in prompt:
                assert len(kwargs['images']) == 1
                assert all(p['id'] in kwargs['labels'][0] for p in people)
                return json.dumps({'分组': groups})
            return json.dumps(attribution_result())
        output = tmp_path / f'result{len(outputs)}.json'
        dispatch('角色卡', episode, {}, output, call=call)
        outputs.append(json.loads(output.read_text()))
    assert outputs[0]['临时人物映射'] == outputs[1]['临时人物映射']


@pytest.mark.parametrize('count', [64, 65])
def test_production_image_limit_does_not_truncate_or_batch(episode, tmp_path, count):
    people, declarations = [], []
    table = json.loads((episode / '镜头表.json').read_text())
    frames = []
    for n in range(1, count + 1):
        filename = f'K001_{n:02d}.jpg'
        (episode / '素材/frames' / filename).write_bytes(b'mock')
        frames.append({'文件': filename, '时间': '00:00.1'})
        people.append({'id': f'E01-P01-T{n}', '片段': 'E01-P01', '外观': f'紫衣{n}',
                       '出现镜头': ['E01-K001'], '映射': None, '映射依据': ''})
        declarations.append({'人物': people[-1]['id'], '外观': people[-1]['外观'], '依据帧': [filename]})
    table['镜头'][0]['帧'] = frames
    save(episode / '镜头表.json', table)
    save(episode / '临时人物.json', people)
    path = episode / '派发结果/动作识别/E01-P01.json'
    actions = json.loads(path.read_text())
    actions['镜头'][0]['画面人物'] = declarations
    save(path, actions)
    save(episode / '声音记录.json', [])
    calls = []
    def call(role, prompt, **kwargs):
        calls.append(kwargs)
        return json.dumps({'分组': [{**merge_result()['分组'][0], '成员': [p['id'] for p in people]}]})
    if count == 65:
        with pytest.raises(ValueError, match='上限'):
            dispatch('角色卡', episode, {}, tmp_path / 'result.json', call=call)
        assert not calls
    else:
        dispatch('角色卡', episode, {}, tmp_path / 'result.json', call=call)
        assert len(calls) == 1 and len(calls[0]['images']) == 64


def test_batches_finish_out_of_order_but_publish_original_sound_order(episode, tmp_path):
    import time
    sounds = json.loads((episode / '声音记录.json').read_text())
    sounds[0]['止'] = '00:00.9'
    sounds.append({**sounds[0], 'id': 'E01-S002', '起': '00:01.0', '止': '00:01.9'})
    save(episode / '声音记录.json', sounds)
    finished = []
    def call(role, prompt, **kwargs):
        if '本次阶段：人物合并' in prompt:
            return json.dumps(merge_result())
        data = json.loads(prompt.split('## 本次阶段：对白归属\n')[1])
        sentence = data['负责完整句'][0]['id']
        if sentence == 'E01-S001':
            time.sleep(0.03)
        assert len(data['相邻句仅参考不得输出']) == 1
        assert '缺失' in data['证据状态']['口型记录']
        finished.append(sentence)
        result = attribution_result()
        result['对白归属'][0]['句'] = sentence
        return json.dumps(result)
    output = tmp_path / 'result.json'
    dispatch('角色卡', episode, {}, output, call=call)
    assert finished == ['E01-S002', 'E01-S001']
    assert [r['句'] for r in json.loads(output.read_text())['对白归属']] == ['E01-S001', 'E01-S002']


def test_changed_input_before_final_publish_rejects_old_approval(episode, tmp_path):
    output = tmp_path / 'result.json'
    output.write_text('old-result')
    old_table = (episode / '本集人物.json').read_bytes()
    def call(role, prompt, **kwargs):
        if '本次阶段：人物合并' in prompt:
            return json.dumps(merge_result())
        sounds = json.loads((episode / '声音记录.json').read_text())
        sounds[0]['文本'] = '外部更新'
        save(episode / '声音记录.json', sounds)
        return json.dumps(attribution_result())
    with pytest.raises(ValueError, match='版本过期'):
        dispatch('角色卡', episode, {}, output, call=call)
    assert output.read_text() == 'old-result'
    assert (episode / '本集人物.json').read_bytes() == old_table


def test_attribution_image_limit_prechecked_and_retry_uses_same_freeze(episode, tmp_path, monkeypatch):
    from s2s import config
    actual = config.get_vision
    monkeypatch.setattr(config, 'get_vision', lambda: {**actual(), 'max_images_per_call': 1})
    calls = []
    def call(role, prompt, **kwargs):
        calls.append(prompt)
        return json.dumps(merge_result())
    with pytest.raises(ValueError, match='对白归属图片超过配置上限'):
        dispatch('角色卡', episode, {}, tmp_path / 'limit.json', call=call)
    assert len(calls) == 1  # 合并一张图可调用，跨段归属两张图拒绝。
    monkeypatch.setattr(config, 'get_vision', actual)
    attempts = []
    def retry(role, prompt, **kwargs):
        if '本次阶段：人物合并' in prompt:
            return json.dumps(merge_result())
        attempts.append((prompt.split('上次返回：')[0].rstrip(), kwargs['images'], kwargs['labels']))
        assert '过期' in prompt  # 旧证据不能被当作否定事实。
        return '{}' if len(attempts) == 1 else json.dumps(attribution_result())
    save(episode / '口型记录.json', {'句': []})
    dispatch('角色卡', episode, {}, tmp_path / 'retry.json', call=retry)
    assert len(attempts) == 2 and attempts[0] == attempts[1]


@pytest.mark.parametrize('edge', ['gap', 'overlap', 'out_of_bounds'])
def test_sentence_owner_rejects_invalid_boundaries_before_call(episode, tmp_path, edge):
    if edge == 'out_of_bounds':
        sounds = json.loads((episode / '声音记录.json').read_text())
        sounds[0]['起'], sounds[0]['止'] = '00:02.0', '00:02.1'
        save(episode / '声音记录.json', sounds)
    else:
        segments = json.loads((episode / '片段表.json').read_text())
        segments['片段'][1]['起'] = '00:01.1' if edge == 'gap' else '00:00.9'
        save(episode / '片段表.json', segments)
    calls = []
    with pytest.raises(ValueError):
        dispatch('角色卡', episode, {}, tmp_path / 'result.json', call=lambda *a, **kw: calls.append(a))
    assert calls == []
