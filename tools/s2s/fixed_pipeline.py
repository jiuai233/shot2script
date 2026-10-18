"""E0最薄固定单集入口；保留独立验收和正式稿的既有安全门禁。"""

from contextlib import contextmanager
import fcntl
from pathlib import Path

from . import config
from .board import initialize_board
from .dispatch.controllers import episode_number
from .dispatch.rolecard import apply_rolecard_result, validate_two_stage_result
from .dispatch.roles import dispatch
from .fixed_steps import finish_episode, merge_actions, save_draft
from .offline_episode import OfflineEpisode
from .production import check_episode_delivery, prepare_episode, require, resolve_video
from .quality import check_script
from .request_ledger import RequestLedger
from .run_record import RunRecord, digest, file_hash, freeze_inputs, load, save
from .validate import validate_file


@contextmanager
def run_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.fixed-episode.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('同一隔离运行正在执行；不要并行启动相同单集') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _validate(episode, kinds):
    for kind in kinds:
        require(validate_file(episode / (kind + '.json'), kind), kind)
    return {'通过': True, '已校验': list(kinds)}


def run_fixed_episode(series, number, *, video=None, run_root=None, fixture=None,
                      approvals=None, timeout=900, max_requests=100, preflight=False, costing=None):
    if not series.strip() or series in ('.', '..') or '/' in series or '\\' in series:
        raise ValueError('剧名须为单个目录名')
    number = episode_number(number)
    original = config.PROJECT_ROOT / '剧目' / series
    source = resolve_video(original, number, video)
    offline = OfflineEpisode(fixture) if fixture else None
    frozen = freeze_inputs(source, fixture=fixture, roles=original / '角色卡.json')
    root = Path(run_root).expanduser().resolve() if run_root else original / '隔离运行' / digest(frozen)[:16]
    episode = root / series / f'E{number:02d}'
    ledger = RequestLedger(episode / '流水线/调用账本.json', simulated=offline is not None,
                           max_requests=max_requests, max_seconds=timeout, costing=costing)
    # 金额不可验证时，在任何本地模型/预处理和远程请求之前明确阻断。
    checked = ledger.preflight()
    if preflight:
        return {**checked, '目录': str(episode), '冻结SHA256': digest(frozen),
                '说明': '离线仅验证合同' if offline else '真实计价预检通过'}
    with run_lock(root):
        fresh = not (episode / '流水线/冻结.json').exists()
        if fresh and (root / series).exists() and any((root / series).rglob('*')):
            raise ValueError('隔离目录已有未冻结产物，拒绝采纳或覆盖；请选择新的运行目录')
        initialize_board(series, number, root=root)
        if fresh:
            inherited = (offline.data.get('初始角色卡', []) if offline
                         else load(original / '角色卡.json') if (original / '角色卡.json').is_file() else [])
            save(episode.parent / '角色卡.json', inherited)
        record = RunRecord(episode, frozen)
        call = offline.call if offline else None
        output = episode / '派发结果'

        def role(name, variables, path, **kwargs):
            return dispatch(name, episode, variables, path, call=call, request_budget=ledger,
                            direct_text=name in ('成稿', '验收'), resume=True, timeout=timeout, **kwargs)

        record.stage('固定预处理', lambda: offline.prepare(source, episode) if offline
                     else prepare_episode(source, episode, optional_lipsync=True), inputs=[source])
        record.stage('输入校验', lambda: _validate(episode, ('镜头表', '片段表', '声音记录', '字幕记录', '待核对')))
        segments = load(episode / '片段表.json')['片段']
        if not segments:
            raise ValueError('没有可调度的镜头片段')
        actions = {}
        for segment in segments:
            ident = segment['id']
            path = output / '动作识别' / (ident + '.json')
            record.stage('动作-' + ident, lambda ident=ident, path=path:
                         role('动作识别', {'片段号': ident}, path),
                         inputs=[episode / (kind + '.json') for kind in ('镜头表', '片段表', '声音记录', '字幕记录')])
            actions[ident] = path
        record.stage('动作合并', lambda: merge_actions(episode, actions, record.publish), inputs=actions.values())
        record.stage('动作合并校验', lambda: _validate(episode, ('动作记录', '临时人物', '待核对')))
        rolecard = output / '角色卡/建立.json'

        def establish():
            if rolecard.is_file():
                validate_two_stage_result(episode, load(rolecard))
                return {'通过': True, '结果文件': str(rolecard), '复用中断产物': True}
            return role('角色卡', {}, rolecard, mode='建立')

        record.stage('角色卡两步', establish, inputs=[*actions.values(), episode / '临时人物.json', episode / '声音记录.json',
                                                      episode.parent / '角色卡.json'])
        # 路径明确但文件尚未提供时停在审核前；模板不等于批准。
        if approvals and not Path(approvals).is_file():
            result = load(rolecard)
            template = episode / '流水线/审核模板.json'
            save(template, {'结果版本': 2, '输入版本': result['输入版本'], '批准': []})
            return {'通过': True, '状态': '等待显式人物审核', '正式交付': False,
                    '目录': str(episode), '角色卡结果': str(rolecard), '审核模板': str(template),
                    '说明': '由审核者填写批准条目及依据，并保存到指定审核路径后重跑相同命令'}
        approved_hash = file_hash(approvals) if approvals else None
        previous = record.data['阶段'].get('人物审核apply', {}).get('结果', {})
        if previous and previous.get('审核文件SHA256') != approved_hash:
            raise ValueError('已经apply的审核决策不能覆盖；请保留本次产物并新建定向修订任务')

        def apply():
            def publish(writes, receipt):
                receipt['审核文件SHA256'] = approved_hash
                record.publish(writes, receipt)
            return apply_rolecard_result(episode, rolecard, approvals, publish=publish, defer_unreviewed=approvals is None)

        record.stage('人物审核apply', apply, inputs=[rolecard, *([approvals] if approvals else [])])
        record.stage('归属校验', lambda: _validate(episode, ('本集人物', '临时人物', '归属表', '待核对')))
        draft = output / '成稿/全集.json'
        record.stage('成稿与排布', lambda: role('成稿', {'固定动作排布': True}, draft),
                     inputs=[episode / (kind + '.json') for kind in ('动作记录', '归属表', '声音记录', '待核对')])
        record.stage('成稿落板', lambda: save_draft(episode, draft, record.publish), inputs=[draft])
        mechanical = record.stage('机械检查', lambda: check_script(episode))
        audit = output / '验收/全文.json'
        record.stage('独立全文验收', lambda: role('验收', {}, audit, mode='全文审查'),
                     inputs=[episode / name for name in ('工作稿.md', '对照表.json', '镜头表.json', '声音记录.json', '字幕记录.json')])
        result = record.stage('终态发布', lambda: finish_episode(episode, audit, mechanical,
                             record.publish, simulated=offline is not None))
        # 两个终态都从现有交付入口复查；模拟永远返回待裁定而非正式稿。
        delivery = check_episode_delivery(episode)
        if delivery['正式交付'] != result['正式交付']:
            raise ValueError('阶段终态与正式交付门禁不一致')
        return {**result, '阶段账本': str(record.path), '调用账本': str(ledger.path),
                '冻结': str(record.folder / '冻结.json')}
