"""显式离线夹具：不读取视频、不调用模型，不能生成正式验收授权。"""

from copy import deepcopy
import json
from pathlib import Path
import threading

from .run_record import digest, load, save


class OfflineEpisode:
    def __init__(self, path):
        self.data = load(path)
        if not isinstance(self.data, dict) or self.data.get('版本') != 1 or self.data.get('仅离线模拟') is not True:
            raise ValueError('离线夹具必须显式声明版本1和仅离线模拟')
        if not all(isinstance(self.data.get(key), dict) for key in ('固定产物', '模型返回')):
            raise ValueError('离线夹具须包含固定产物和模型返回对象')
        self.lock = threading.Lock()
        self.counts = {}

    def prepare(self, video, episode):
        allowed = {'镜头表.json', '片段表.json', '声音记录.json', '字幕记录.json', '待核对.json',
                   '口型记录.json', '声纹聚类.json', '声纹候选.json', '素材/人声段.json',
                   '素材/回听/疑似漏段.json'}
        for name, source in self.data['固定产物'].items():
            if name not in allowed:
                raise ValueError('离线固定产物名越界：' + name)
            value = deepcopy(source)
            if name == '镜头表.json':
                value['视频'] = str(video)
            save(episode / name, value)
        for shot in load(episode / '镜头表.json')['镜头']:
            for frame in shot['帧']:
                path = (episode / '素材/frames' / frame['文件']).resolve()
                if not path.is_relative_to((episode / '素材/frames').resolve()) or path.suffix != '.jpg':
                    raise ValueError('离线帧路径越界')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'OFFLINE FIXTURE: NOT A REAL VIDEO FRAME\n')
        (episode / '回听.html').write_text('离线合同夹具，没有真实可回听音频', encoding='utf-8')
        save(episode / '候选证据状态.json', {'口型': '缺失；不能作为否定证据', '模拟运行': True})
        return {'通过': True, '模拟预处理': True}

    def call(self, role, prompt, **kwargs):
        scope = kwargs['request_budget']
        parts = scope.stage.split('/')
        if role == '动作识别':
            key = '动作识别/' + parts[2]
        elif role == '角色卡':
            key = parts[1] + ('/' + parts[2] if parts[1] == '对白归属' else '')
        elif role == '验收':
            key = '验收画面/' + parts[2] if kwargs.get('images') else '验收文本'
        else:
            key = role
        ticket = scope.begin({'模型': kwargs['model'], '图片数': len(kwargs.get('images') or []),
                              '夹具键': key, '请求SHA256': digest(prompt)})
        error = None
        try:
            with self.lock:
                index = self.counts.get(key, 0)
                self.counts[key] = index + 1
            if key not in self.data['模型返回']:
                raise ValueError('离线夹具缺少本阶段返回：' + key)
            source = self.data['模型返回'][key]
            response = source[min(index, len(source) - 1)] if isinstance(source, list) else source
            if isinstance(response, dict) and '模拟错误' in response:
                raise RuntimeError(response['模拟错误'])
            return response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            scope.finish(ticket, usage=None, error=error)
