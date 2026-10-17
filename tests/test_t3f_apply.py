"""人工来源、删除重编号、旧页面拒绝和写回校验。"""
from copy import deepcopy
from pathlib import Path

import pytest
from s2s.asr.crosscheck import merge, update_pending
from s2s.listen.apply import apply_results
from s2s.listen.storage import read, write
from s2s.validate import validate_file


def setup_board(tmp_path):
    episode=tmp_path/'E01'; episode.mkdir()
    primary=[{'id':f'E01-S{i:03d}','起':f'00:0{i}.0','止':f'00:0{i}.5',
              '文本':f'主路{i}','声线':'E01-V1','听辨':'清楚'} for i in range(1,6)]
    secondary=[{**r,'文本':f'副路{i}'} for i,r in enumerate(primary,1)]
    sounds=merge(primary,secondary)
    for r in sounds: r['核对']['字幕']='字幕'
    write(episode/'声音记录.json',sounds)
    write(episode/'待核对.json',update_pending(sounds,[],'E01'))
    return episode,sounds


def test_all_sources_delete_and_renumber(tmp_path):
    episode,sounds=setup_board(tmp_path)
    decisions=[]
    for row,source in zip(sounds,['主路','删除','副路','字幕','手改']):
        text=row['核对'][source] if source in ('主路','副路','字幕') else '亲耳听到的文本' if source=='手改' else ''
        decisions.append({'句号':row['id'],'决定':source,'最终文本':text,'听辨':'听不清','备注':'回听备注'})
    result=apply_results(episode,{'声音快照':sounds,'结果':decisions})
    rows=read(episode/'声音记录.json')
    assert [r['文本'] for r in rows]==['主路1','副路3','字幕','亲耳听到的文本']
    assert all(r['核对']['状态']=='人工确认' and r['听辨']=='听不清' for r in rows)
    assert [r['id'] for r in rows]==[f'E01-S{i:03d}' for i in range(1,5)]
    pending=read(episode/'待核对.json')
    assert all(r['状态']=='已裁定' for r in pending)
    assert '句' not in pending[1]['位置']
    assert pending[2]['位置']['句']=='E01-S002'
    assert read(result['句号对照'])['E01-S002'] is None
    assert read(Path(result['备份'])/'声音记录.json')==sounds
    assert all(validate_file(episode/name)['通过'] for name in ('声音记录.json','待核对.json'))


@pytest.mark.parametrize('change', ['stale','unknown','duplicate','badsource','empty','badclarity','mismatch'])
def test_invalid_results_leave_board_untouched(tmp_path,change):
    episode,sounds=setup_board(tmp_path)
    decision={'句号':'E01-S001','决定':'主路','最终文本':'主路1','听辨':'清楚','备注':''}
    payload={'声音快照':deepcopy(sounds),'结果':[decision]}
    if change=='stale': payload['声音快照'][0]['文本']='旧'
    if change=='unknown': decision['句号']='E01-S099'
    if change=='duplicate': payload['结果'].append(decision)
    if change=='badsource': decision['决定']='猜的'
    if change=='empty': decision.update({'决定':'手改','最终文本':''})
    if change=='badclarity': decision['听辨']='未知'
    if change=='mismatch': decision['最终文本']='别的文本'
    with pytest.raises(ValueError): apply_results(episode,payload)
    assert read(episode/'声音记录.json')==sounds
    assert not (episode/'素材'/'回听'/'备份').exists()


def test_validation_failure_does_not_write(tmp_path):
    episode,sounds=setup_board(tmp_path)
    pending=read(episode/'待核对.json'); pending[0]['级别']='无效'
    write(episode/'待核对.json',pending)
    with pytest.raises(ValueError,match='校验失败'):
        apply_results(episode,{'声音快照':sounds,'结果':[]})
    assert read(episode/'声音记录.json')==sounds


def test_delete_all(tmp_path):
    episode,sounds=setup_board(tmp_path)
    decisions=[{'句号':r['id'],'决定':'删除','最终文本':'','听辨':'清楚','备注':''} for r in sounds]
    result=apply_results(episode,{'声音快照':sounds,'结果':decisions})
    assert result['剩余句数']==0
    assert validate_file(episode/'待核对.json')['通过']
