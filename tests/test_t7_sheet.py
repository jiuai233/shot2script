import importlib.util
import json
from pathlib import Path
import subprocess
from urllib.parse import unquote

from jsonschema import Draft202012Validator
from lxml import html
import pytest

from s2s.evaluate import SCHEMA, validate_annotation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('t7_annotate_sheet', ROOT / 'tools' / 'annotate_sheet.py')
sheet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sheet)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


@pytest.fixture
def inputs(tmp_path):
    episode = tmp_path / '本集'
    shots = {'集': 1, '视频': 'videos/sample/E01.mp4', '时长': '00:04.0', '镜头': [
        {'id': 'E01-K001', '起': '00:00.0', '止': '00:02.0', '帧': [
            {'文件': f'K001_{i:02d}.jpg', '时间': f'00:0{i-1}.0'} for i in range(1, 4)]},
        {'id': 'E01-K002', '起': '00:02.0', '止': '00:04.0', '帧': [
            {'文件': 'K002_01.jpg', '时间': '00:02.1'}]}]}
    save(episode / '镜头表.json', shots)
    subtitles = [
        {'起': '00:00.1', '止': '00:01.1', '文字': '原话</script><b>不得执行</b>', '类型': '台词字幕'},
        {'起': '00:01.0', '止': '00:03.0', '文字': '跨镜头原话', '类型': '台词字幕'},
        {'起': '00:02.0', '止': '00:03.0', '文字': '地点文字', '类型': '地点字幕'}]
    save(episode / '字幕记录.json', subtitles)
    for shot in shots['镜头']:
        for frame in shot['帧']:
            path = episode / '素材' / 'frames' / frame['文件']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture')
    return episode, tmp_path / '输出' / '标注.json', tmp_path / '输出' / '工作表.html'


def test_prefill_and_html_payload(inputs):
    episode, annotation, output = inputs
    assert sheet.main(list(map(str, inputs))) == 0
    data = json.loads(annotation.read_text())
    assert len(data['台词']) == 2
    assert not data['人物'] and not data['关键动作'] and not data['校对完成']
    assert all(row['说话人'] is None and row['形态'] is None for row in data['台词'])
    validate_annotation(data)
    tree = html.fromstring(output.read_text())
    payload = json.loads(tree.get_element_by_id('sheet-data').text)
    assert len(payload['shots']) == 2
    assert len(payload['annotation']['台词']) == 2
    for shot in payload['shots']:
        for frame in shot['frames']:
            assert (output.parent / unquote(frame['src'])).exists()
            assert not frame['src'].startswith(('/', 'http'))
    assert not tree.xpath('//script[@src] | //link[@rel="stylesheet"]')
    assert len(tree.xpath('//script')) == 2
    assert '原话</script>' in payload['annotation']['台词'][0]['文本']


def test_existing_annotation_is_preserved(inputs):
    episode, annotation, output = inputs
    data = sheet.prefill(episode)
    data['台词'][0]['文本'] = '人工校正的原话'
    save(annotation, data)
    before = annotation.read_bytes()
    assert sheet.main(list(map(str, inputs))) == 0
    assert annotation.read_bytes() == before
    tree = html.fromstring(output.read_text())
    assert json.loads(tree.get_element_by_id('sheet-data').text)['annotation'] == data


def node_export(payload, script):
    result = subprocess.run(['node', '-e', script], input=json.dumps(payload, ensure_ascii=False),
                            text=True, capture_output=True, timeout=10, check=True)
    return json.loads(result.stdout)


def test_node_export_is_valid_schema(inputs):
    episode, _, output = inputs
    data = sheet.prefill(episode)
    payload = {'state': data, 'schema': json.loads(SCHEMA.read_text())}
    javascript = f'''
const {{buildAnnotation}} = require({json.dumps(str(ROOT / 'eval' / 'sheet.js'))});
let input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
input.state.人物.push({{id:'H01', 名称:'穿红衣的奶奶', 外观:'白发、红花衬衫', 出现:['00:00.0-00:04.0']}});
input.state.台词.forEach(row => {{row.说话人 = 'H01'; row.形态 = '画内';}});
input.state.关键动作.push({{id:'M001', 起:'00:01.0', 止:'00:01.5', 执行者:'H01', 动作:'举起文件', 对象:'文件', 物件:'文件', 结果:null, 类别:4}});
input.state.校对完成 = true;
console.log(JSON.stringify(buildAnnotation(input.state, input.schema)));
'''
    exported = node_export(payload, javascript)
    Draft202012Validator(payload['schema']).validate(exported)
    validate_annotation(exported, complete=True)
    assert exported['人物'][0]['id'] == 'H01'
    assert exported['关键动作'][0]['动作'] == '举起文件'


@pytest.mark.parametrize('change', [
    "input.state.台词[0].起 = 'bad'",
    "input.state.台词[0].止 = '00:00.0'",
    "input.state.台词[0].说话人 = 'H99'",
    "input.state.台词.push({...input.state.台词[0]})",
    "input.state.校对完成 = true",
])
def test_node_export_rejects_invalid_state(inputs, change):
    episode, _, _ = inputs
    payload = {'state': sheet.prefill(episode), 'schema': json.loads(SCHEMA.read_text())}
    script = f'''
const {{buildAnnotation}} = require({json.dumps(str(ROOT / 'eval' / 'sheet.js'))});
let input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
{change};
try {{buildAnnotation(input.state, input.schema); console.log('false');}}
catch (error) {{console.log('true');}}
'''
    assert node_export(payload, script) is True


def test_render_rejects_wrong_episode(inputs):
    episode, _, output = inputs
    data = sheet.prefill(episode)
    data['集'] = 2
    with pytest.raises(ValueError, match='不一致'):
        sheet.render_sheet(episode, data, output)
