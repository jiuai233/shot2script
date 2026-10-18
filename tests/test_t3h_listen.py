"""裁定严格迁移，快照只绑定四个字段，回听展示不影响自动排序。"""

from copy import deepcopy
import json
import subprocess
from pathlib import Path

from lxml import html
import pytest

from s2s.asr.crosscheck import update_pending
from s2s.listen.apply import apply_results
from s2s.listen.migrate import migrate
from s2s.listen.sheet import build_sheet
from s2s.listen.snapshot import summary
from s2s.listen.storage import read, write
from test_t3f_apply import setup_board
from test_t3f_crosscheck import sound
from test_t3f_sheet import board


def decision(row, text='人工确定'):
    return {'句号': row['id'], '决定': '手改', '最终文本': text, '听辨': '清楚', '备注': '原备注'}


def checked(text, a, b, i=1):
    row = sound(text, a, b, i)
    row['核对'] = {'状态':'一致','主路':text,'副路':text,'字幕':'','优先级':2}
    return row


def test_exact_migration_preserves_final_text_only_with_identical_coverage():
    old = sound('旧识别', 1, 2)
    new = checked('旧识别', 1, 2)
    rows, mapping = migrate([new], {'声音快照':[old], '结果':[decision(old)]})
    assert rows[0]['文本'] == '人工确定' and rows[0]['核对']['状态'] == '人工确认'
    assert mapping[0]['状态'] == '已沿用'
    assert new['文本'] == '旧识别'


@pytest.mark.parametrize('kind', ['split','merge','boundary'])
def test_changed_coverage_never_auto_confirms(kind):
    old = [sound('原句', 1, 3)]
    new = [checked('新句', 1, 3.1)]
    if kind == 'split': new = [checked('原', 1, 2), checked('句', 2, 3, 2)]
    if kind == 'merge':
        old.append(sound('另句', 3, 4, 2)); new = [checked('合并句', 1, 4)]
    rows, mapping = migrate(new, {'声音快照':old,'结果':[decision(r) for r in old]})
    assert all(r['核对']['状态'] != '人工确认' for r in rows)
    assert all(r['状态'] == '待确认' for r in mapping)
    pending = update_pending(rows, [], 'E01', mapping)
    assert len(pending) == len(rows) and all('原裁定' in r['依据'] for r in pending)


def test_deleted_old_sentence_needs_confirmation_and_unmatched_preserved():
    old = [sound('原句', 1, 2), sound('另一句', 4, 5, 2)]
    choices = [decision(old[0]), decision(old[1])]
    choices[0].update({'决定':'删除','最终文本':''})
    rows, mapping = migrate([checked('现句',1,2)], {'声音快照':old,'结果':choices})
    assert mapping[0]['状态'] == '待确认' and '删除' in mapping[0]['原裁定']
    assert mapping[1]['状态'] == '未匹配' and mapping[1]['新句号'] is None


@pytest.mark.parametrize('field,value', [('声线','E01-V9'), ('听辨','听不清'), ('核对',{'状态':'一致','主路':'主路1','副路':'新补充','字幕':'','优先级':2})])
def test_apply_ignores_but_reports_noncore_changes(tmp_path, field, value):
    episode, snapshot = setup_board(tmp_path)
    current = deepcopy(snapshot); current[0][field] = value
    write(episode/'声音记录.json', current)
    result = apply_results(episode, {'声音快照':snapshot,'声音摘要':summary(snapshot),'结果':[decision(snapshot[0])]})
    assert field in result['非摘要字段变化'][0]['字段']
    if field == '声线': assert read(episode/'声音记录.json')[0]['声线'] == 'E01-V9'


@pytest.mark.parametrize('field,value', [('id','E01-S099'), ('文本','已修改'), ('起','00:00.9'), ('止','00:01.6')])
def test_apply_rejects_each_core_change(tmp_path, field, value):
    episode, snapshot = setup_board(tmp_path)
    current = deepcopy(snapshot); current[0][field] = value
    write(episode/'声音记录.json', current)
    with pytest.raises(ValueError, match='声音记录已变化'):
        apply_results(episode, {'声音快照':snapshot,'结果':[decision(snapshot[0])]})
    assert read(episode/'声音记录.json') == current


def test_subtitle_reference_from_page_is_adoptable(tmp_path):
    episode, snapshot = setup_board(tmp_path)
    value = decision(snapshot[0], '人工采用的参考字幕'); value['决定'] = '字幕'
    result = apply_results(episode, {'声音快照':snapshot, '字幕参考':{snapshot[0]['id']:value['最终文本']},'结果':[value]})
    assert result['处理句数'] == 1
    assert read(episode/'声音记录.json')[0]['文本'] == value['最终文本']


def test_page_context_uses_neighbors_in_chronological_order_and_prior_display(board, tmp_path):
    sounds = read(board/'声音记录.json')
    write(board/'素材/回听/裁定迁移.json', [{'新句号':sounds[0]['id'],'旧句号':['E01-S008'],'状态':'待确认','原裁定':'亲耳听到<旧文>'}])
    write(board/'字幕记录.json', [{'起':'00:00.0','止':'00:03.0','文字':'完全错误的字幕','类型':'台词字幕'}])
    result = build_sheet(board,tmp_path/'page.html')
    tree = html.fromstring((tmp_path/'page.html').read_text())
    data = json.loads(tree.xpath('//script[@id="listen-data"]')[0].text)
    item = next(r for r in data['items'] if r['句号']=='E01-S002')
    assert item['本句片段'] == [.8, 2.2]
    assert item['上下文片段'] == [0, 3]
    assert result['句数'] == 3
    assert data['items'][0]['句号'] == 'E01-S003'
    assert '原裁定：亲耳听到<旧文>' in tree.text_content()
    assert '参考，可能不准' in tree.text_content()
    assert data['summary'] == summary(sounds)
    assert read(board/'声音记录.json') == sounds


def test_js_summary_storage_retains_voice_changes(tmp_path):
    js = Path('tools/s2s/listen/sheet.js').resolve()
    script = f"""
    const assert=require('node:assert/strict'), api=require({json.dumps(str(js))});
    const snapshot=[{{id:'E01-S001',文本:'甲',起:'00:00.0',止:'00:01.0',声线:'E01-V1'}}];
    const item={{句号:'E01-S001',主路:'甲',副路:'乙',字幕:'丙'}};
    const choice={{句号:'E01-S001',决定:'手改',最终文本:'人工',听辨:'清楚',备注:''}};
    let value; const storage={{setItem:(k,v)=>value=v,getItem:()=>value}};
    api.persist(storage,'key',snapshot,{{'E01-S001':choice}});
    assert.equal(Object.keys(api.restore(storage,'key',[{{...snapshot[0],声线:'E01-V2'}}],[item]).decisions).length,1);
    assert.equal(Object.keys(api.restore(storage,'key',[{{...snapshot[0],起:'00:00.1'}}],[item]).decisions).length,0);
    const exported=api.exportResults(snapshot,[item],{{'E01-S001':choice}});
    assert.deepEqual(exported.声音摘要,[{{id:'E01-S001',文本:'甲',起:'00:00.0',止:'00:01.0'}}]);
    assert.equal(exported.字幕参考['E01-S001'],'丙');
    """
    result = subprocess.run(['node','-e',script], capture_output=True,text=True,timeout=10)
    assert result.returncode == 0, result.stderr


def test_confirmed_migration_does_not_reappear_as_pending(board, tmp_path):
    rows = read(board/'声音记录.json')
    for r in rows: r['核对']['状态']='人工确认'
    write(board/'声音记录.json', rows)
    write(board/'素材/回听/裁定迁移.json', [{'新句号':rows[0]['id'],'旧句号':['E01-S008'],'状态':'待确认','原裁定':'旧文本'}])
    assert build_sheet(board,tmp_path/'confirmed.html')['句数']==0


def test_truncated_or_abnormal_agreement_still_requires_listening():
    from s2s.asr.crosscheck import needs_listen
    row = checked('同文',0,1)
    row['截断'] = True
    assert needs_listen(row)
    assert len(update_pending([row],[],'E01')) == 1
    row.pop('截断'); row['止']='00:00.2'
    assert needs_listen(row)
    row['核对']['状态']='人工确认'
    assert not needs_listen(row)
