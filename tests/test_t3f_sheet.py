"""离线页资源与 Node 纯逻辑测试。"""
import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import unquote

from lxml import html
import pytest

from s2s.listen import sheet
from s2s.listen.storage import write

ROOT = Path(__file__).resolve().parents[1]


def test_node_export_and_storage():
    result=subprocess.run(['node',str(ROOT/'tests/fixtures/t3f/listen_logic.cjs'),
        str(ROOT/'tools/s2s/listen/sheet.js')],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert '通过' in result.stdout


@pytest.fixture
def board(tmp_path):
    episode=tmp_path/'E01'; episode.mkdir()
    video=tmp_path/'source.wav'
    subprocess.run(['/opt/homebrew/bin/ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=500:duration=4',str(video)],check=True,timeout=15)
    frames=episode/'素材/frames'; frames.mkdir(parents=True)
    from PIL import Image
    Image.new('RGB',(40,60)).save(frames/'K001_01.jpg')
    write(episode/'镜头表.json',{'集':1,'视频':str(video),'时长':'00:04.0','镜头':[
        {'id':'E01-K001','起':'00:00.0','止':'00:04.0','帧':[{'文件':'K001_01.jpg','时间':'00:01.0'}]}]})
    rows=[]
    for i,status,priority in [(1,'一致',2),(2,'不一致',3),(3,'仅副路',1)]:
        rows.append({'id':f'E01-S{i:03d}','起':f'00:0{i-1}.0','止':f'00:0{i}.0','文本':'主路',
                     '声线':'E01-V1','听辨':'清楚','核对':{'状态':status,'主路':'</script><img onerror=alert(1)>甲',
                     '副路':'乙','字幕':'甲','优先级':priority}})
    write(episode/'声音记录.json',rows)
    return episode


def test_real_audio_relative_assets_html_escaping_and_order(board,tmp_path):
    output=tmp_path/'pages/回听.html'
    result=sheet.build_sheet(board,output)
    assert result['句数']==2
    tree=html.fromstring(output.read_text())
    assert len(tree.xpath('//article'))==2
    assert 'E01-S003' in tree.xpath('//article/h2')[0].text_content()
    assert not tree.xpath('//img[@onerror]')
    for src in tree.xpath('//audio/@src|//img/@src'):
        assert not src.startswith(('http:', 'https:', 'file:','/'))
        assert (output.parent/unquote(src)).is_file()
    assert len(tree.xpath('//audio'))==4
    assert len(tree.xpath('//mark'))>0
    assert not tree.xpath('//script[@src]|//link[@href]')
    payload=json.loads(tree.xpath('//script[@id="listen-data"]')[0].text)
    assert payload['items'][1]['主路'].startswith('</script>')
    assert sheet.build_sheet(board,tmp_path/'all.html',True)['句数']==3
    # 最后一句上下文从前一句起点 1 秒到本句止点 3 秒。
    assert abs(sheet.audio.duration(board/'素材/回听/E01-S003_1.0-3.0_上下文_v2.m4a')-2)<.2


def test_missing_frame_fails_clearly(board,tmp_path):
    (board/'素材/frames/K001_01.jpg').unlink()
    with pytest.raises(ValueError,match='代表帧不存在'):
        sheet.build_sheet(board,tmp_path/'page.html')


def test_no_pending_is_usable(board,tmp_path):
    from s2s.listen.storage import read
    rows=read(board/'声音记录.json')
    for r in rows: r['核对']['状态']='人工确认'
    write(board/'声音记录.json',rows)
    result=sheet.build_sheet(board,tmp_path/'empty.html')
    assert result['句数']==0
    assert '没有待回听句子' in (tmp_path/'empty.html').read_text()
