"""固定单集的内容冻结、阶段断点和可恢复多文件发布；不做语义判断。"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

RUNTIME_ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze_inputs(video, *, fixture=None, roles=None):
    from . import config
    project = RUNTIME_ROOT
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    code = {str(p.relative_to(project)): file_hash(p) for p in sorted((project / 'tools').rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.json', '.ts', '.swift')}
    dependencies = {name: file_hash(project / name) if (project / name).is_file() else None
                    for name in ('pyproject.toml', 'uv.lock')}
    prompts = {p.name: file_hash(p) for p in sorted((config.PROJECT_ROOT / 'prompts').glob('*.md'))}
    settings = config.load_config()
    # 配置只冻结散列和模型标识，不把环境、密钥或任意自定义配置正文写入运行目录。
    models = {role: {k: v for k, v in config.get_agent(role).items()
                     if k in ('model', 'runner', 'thinking', 'max_images_per_call')}
              for role in ('动作识别', '角色卡', '成稿', '验收')}
    local_models = {str(p.relative_to(config.PROJECT_ROOT)): file_hash(p)
                    for p in sorted((config.PROJECT_ROOT / 'models').rglob('*'))
                    if p.is_file() and p.suffix in ('.onnx', '.task', '.safetensors')}
    return {'版本': 1, '调度': '固定单集-E0-v1', '代码提交': commit, '代码SHA256': code,
            '依赖文件SHA256': dependencies,
            '提示词SHA256': prompts, '配置SHA256': digest(settings), '模型': models,
            'ASR模型': config.get_asr_backend()['model'],
            '本地模型文件SHA256': local_models or None,
            '模型版本说明': '远程权重版本不可观测；本地目录无权重时为null，不冒称已冻结模型文件',
            '固定文本路径': '成稿与验收文本复用既有DeepSeek直连；两者均无工具输入',
            '视频': {'路径': str(Path(video).resolve()), 'SHA256': file_hash(video)},
            '继承角色卡SHA256': file_hash(roles) if roles and Path(roles).is_file() else digest([]),
            '离线夹具': {'路径': str(Path(fixture).resolve()), 'SHA256': file_hash(fixture)} if fixture else None,
            '模拟运行': fixture is not None}


class RunRecord:
    """完成阶段不重跑；发布前写前后散列，崩溃后只接续已知版本。"""

    def __init__(self, episode, frozen):
        self.episode = Path(episode).resolve()
        self.folder = self.episode / '流水线'
        self.folder.mkdir(exist_ok=True)
        self.path = self.folder / '阶段账本.json'
        manifest = self.folder / '冻结.json'
        if manifest.exists():
            if load(manifest) != frozen:
                raise ValueError('输入、配置、提示词或代码版本已变化；请使用新的隔离运行目录')
        else:
            save(manifest, frozen)
        self.data = load(self.path) if self.path.exists() else {
            '版本': 1, '开始': now(), '冻结SHA256': digest(frozen), '阶段': {}, '文件SHA256': self._files(), '活动阶段': None}
        if (not isinstance(self.data, dict) or self.data.get('版本') != 1
                or not isinstance(self.data.get('阶段'), dict) or not isinstance(self.data.get('文件SHA256'), dict)):
            raise ValueError('阶段账本合同无效，拒绝恢复')
        self._recover()
        self.verify()

    def _files(self):
        paths = [self.episode.parent / '角色卡.json']
        paths += [p for p in self.episode.rglob('*') if p.is_file()
                  and not any(part in ('流水线', '校验失败', '.validation') for part in p.relative_to(self.episode).parts)
                  and p.name != '派发日志.jsonl' and not p.name.endswith(('.tmp', '.new'))]
        if any(p.is_symlink() for p in paths):
            raise ValueError('隔离运行产物不接受符号链接，避免读取运行目录外资料')
        return {str(p): file_hash(p) for p in sorted(paths) if p.is_file()}

    def verify(self):
        for filename, expected in self.data['文件SHA256'].items():
            path = Path(filename)
            if not path.is_file() or file_hash(path) != expected:
                raise ValueError('运行产物被外部修改或缺失，拒绝复用：' + filename)
        unexpected = set(self._files()) - self.data['文件SHA256'].keys()
        if unexpected:
            raise ValueError('运行目录出现未记账产物，拒绝复用：' + ', '.join(sorted(unexpected)))

    def _flush(self):
        save(self.path, self.data)

    def _finish(self, name, result, elapsed):
        row = self.data['阶段'][name]
        row.update(状态='完成', 结束=now(), 耗时秒=round(elapsed, 6) if elapsed is not None else None, 结果=result)
        row.pop('发布事务', None)
        self.data.update(活动阶段=None, 文件SHA256=self._files())
        self.data['最近结束'] = now()
        self.data['总墙钟秒'] = round((datetime.fromisoformat(self.data['最近结束'])
                                       - datetime.fromisoformat(self.data['开始'])).total_seconds(), 6)
        if name == '终态发布':
            self.data['结束'] = self.data['最近结束']
        self._flush()

    def _recover(self):
        name = self.data.get('活动阶段')
        if not name:
            return
        row = self.data['阶段'][name]
        transaction = row.get('发布事务')
        if transaction:
            targets = {item['目标'] for item in transaction['文件']}
            actual = self._files()
            expected = self.data['文件SHA256']
            if (set(actual) - expected.keys() - targets
                    or any(actual.get(path) != value for path, value in expected.items() if path not in targets)):
                raise ValueError('发布中断期间出现其他外部修改，拒绝恢复')
            resumed = time.monotonic()
            self._apply_transaction(transaction)
            row['中断恢复'] = True
            row['恢复耗时秒'] = round(time.monotonic() - resumed, 6)
            elapsed = row['耗时秒'] + row['恢复耗时秒'] if row.get('耗时秒') is not None else None
            self._finish(name, transaction['结果'], elapsed)
        else:
            # 纯派发各自原子保存结果；显式标记中断，靠其签名缓存继续，不静默重跑全链。
            folders = {'角色卡两步': '角色卡', '成稿与排布': '成稿', '独立全文验收': '验收'}
            allowed = self.episode / '派发结果' / ('动作识别' if name.startswith('动作-') else folders.get(name, '.none'))
            preprocessing = {'镜头表.json', '片段表.json', '声音记录.json', '字幕记录.json', '待核对.json',
                             '口型记录.json', '声纹聚类.json', '声纹候选.json', '回听.html', '生产任务.json', '候选证据状态.json'}
            for path, value in self._files().items():
                target = Path(path)
                pending_material = name == '固定预处理' and (target.parent == self.episode and target.name in preprocessing
                                                            or target.is_relative_to(self.episode / '素材'))
                if pending_material or path not in self.data['文件SHA256'] and target.is_relative_to(allowed):
                    self.data['文件SHA256'][path] = value
            row.update(状态='中断', 错误='上次进程在本阶段中断；仅接续当前阶段')
            if name == '固定预处理':
                row['恢复要求'] = '素材尚未发布；按生产任务子阶段签名继续，输入校验前不派发模型'
            self.data['活动阶段'] = None
            self._flush()

    def stage(self, name, operation, *, inputs=()):
        previous = self.data['阶段'].get(name, {})
        if previous.get('状态') == '完成':
            return previous['结果']
        attempt = previous.get('尝试', 0) + 1
        prior = {k: v for k, v in previous.items() if k != '历史'}
        self.data['阶段'][name] = {'状态': '运行中', '尝试': attempt, '开始': now(),
                                    '输入产物': [{'路径': str(Path(p).resolve()), 'SHA256': file_hash(p)}
                                             for p in inputs if Path(p).is_file()],
                                    '历史': [*previous.get('历史', []), *([prior] if previous else [])]}
        self.data['活动阶段'] = name
        self._flush()
        started = time.monotonic()
        try:
            result = operation()
            self._finish(name, result, time.monotonic() - started)
            return result
        except Exception as exc:
            row = self.data['阶段'][name]
            row.update(状态='失败', 结束=now(), 耗时秒=round(time.monotonic() - started, 6),
                       错误=type(exc).__name__ + ': ' + str(exc))
            # 已持久化的发布事务由下次恢复；不能重新生成C/Q编号。
            if '发布事务' not in row:
                self.data.update(活动阶段=None, 文件SHA256=self._files())
            self._flush()
            raise

    def publish(self, writes, result):
        """供现有apply及固定落板步骤使用；不修改或默认批准任何语义决定。"""
        name = self.data.get('活动阶段')
        if not name:
            raise ValueError('发布必须属于一个已记账阶段')
        folder = self.folder / '发布' / (name + '-' + str(self.data['阶段'][name]['尝试']))
        folder.mkdir(parents=True, exist_ok=False)
        rows = []
        for index, (target, value) in enumerate(writes):
            target = Path(target).resolve()
            if not (target.is_relative_to(self.episode) or target == self.episode.parent / '角色卡.json'):
                raise ValueError('固定发布目标越界')
            staged = folder / str(index)
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n'
            staged.write_text(text, encoding='utf-8')
            rows.append({'目标': str(target), '之前': file_hash(target) if target.is_file() else None,
                         '之后': file_hash(staged), '暂存': str(staged)})
        transaction = {'文件': rows, '结果': result}
        self.data['阶段'][name]['发布事务'] = transaction
        self._flush()
        self._apply_transaction(transaction)

    @staticmethod
    def _apply_transaction(transaction):
        # 先整体检查，独立编辑不被恢复覆盖。无须重新跑模型或重新分配人物/问题ID。
        for row in transaction['文件']:
            target = Path(row['目标'])
            current = file_hash(target) if target.is_file() else None
            if current not in (row['之前'], row['之后']):
                raise ValueError('发布中断后目标被独立修改，须人工处理：' + str(target))
            if file_hash(row['暂存']) != row['之后']:
                raise ValueError('发布暂存损坏，拒绝恢复')
        for row in transaction['文件']:
            target = Path(row['目标'])
            if target.is_file() and file_hash(target) == row['之后']:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.new')
            shutil.copyfile(row['暂存'], temporary)
            temporary.replace(target)
