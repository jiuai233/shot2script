"""最薄单集调度的确定性落板步骤；事实与排布决定仍来自子角色。"""

from .action_placement import apply_placements
from .quality import check_review, check_script, issue_closed
from .run_record import load


def add_issue(episode, pending, issues, *, source, position, basis, kind='全文完整性与一致性',
              text='', people=(), need='核对原始证据后定向处理', owner='成稿', level='阻断'):
    number = max((int(r['id'].rsplit('Q', 1)[1]) for r in [*pending, *issues]), default=0) + 1
    row = {'id': f'{episode.name}-Q{number:03d}', '来源': source, '位置': position, '原文': text,
           '类型': kind, '级别': level, '相关人物': list(people), '依据': basis, '修正要求': need,
           '处理职责': owner, '状态': '待核对', '复核结果': None}
    pending.append(row)
    return row


def merge_actions(episode, paths, publish):
    known = {r['id'] for r in load(episode.parent / '角色卡.json') if r['状态'] == '已确认'}
    actions, temporary = [], {}
    pending, issues = load(episode / '待核对.json'), load(episode / '问题清单.json')
    segments = load(episode / '片段表.json')['片段']
    if set(paths) != {s['id'] for s in segments}:
        raise ValueError('动作合并必须覆盖本集全部片段')
    for segment in segments:
        result = load(paths[segment['id']])
        if result['片段'] != segment['id'] or [s['镜头'] for s in result['镜头']] != segment['镜头']:
            raise ValueError('动作片段/镜头覆盖不一致')
        for shot in result['镜头']:
            for person in shot['画面人物']:
                ident = person['人物']
                if ident in known:
                    continue
                row = temporary.setdefault(ident, {'id': ident, '片段': segment['id'], '外观': '',
                                                    '出现镜头': [], '映射': None, '映射依据': ''})
                if not row['外观'] and person.get('外观'):
                    row['外观'] = person['外观']
                row['出现镜头'].append(shot['镜头'])
            for action in shot['动作']:
                row = {'id': f'{episode.name}-A{len(actions)+1:04d}', '镜头': shot['镜头'], **action}
                actions.append(row)
                if row['关键'] and row['确定度'] == '存疑':
                    add_issue(episode, pending, issues, source='识别', position={'镜头': shot['镜头']},
                              basis=row['说明'] or '关键动作证据不足', kind='具体动作', people=[row['执行者']], owner='动作识别')
        for row in result['待核对']:
            add_issue(episode, pending, issues, source='识别', position=row['位置'], basis=row['问题'],
                      kind='具体动作', need=row.get('需要') or '核对原始帧', owner='动作识别', level=row['级别'])
        for row in result['加密取帧请求']:
            add_issue(episode, pending, issues, source='识别', position={'镜头': row['镜头']},
                      basis='尚未执行的加密取帧请求：' + row['原因'], kind='具体动作', owner='动作识别')
    receipt = {'通过': True, '动作数': len(actions), '临时人物数': len(temporary)}
    publish([(episode / '动作记录.json', actions), (episode / '临时人物.json', list(temporary.values())),
             (episode / '待核对.json', pending)], receipt)
    return receipt


def save_draft(episode, path, publish):
    draft = load(path)
    actions = apply_placements(load(episode / '动作记录.json'), draft, required=True)
    receipt = {'通过': True, '工作稿': str(episode / '工作稿.md')}
    publish([(episode / '动作记录.json', actions), (episode / '工作稿.md', draft['工作稿']),
             (episode / '对照表.json', draft['对照表'])], receipt)
    return receipt


def finish_episode(episode, review_path, mechanical, publish, *, simulated=False):
    """真实门禁是唯一正式发布条件；离线通过只记录影子分支，不生成正式稿。"""
    review = load(review_path)
    coverage = load(review_path.with_suffix('.coverage.json'))
    pending, issues = load(episode / '待核对.json'), load(episode / '问题清单.json')
    for item in review['问题']:
        row = add_issue(episode, issues, pending, source='验收', position=item['位置'], basis=item['依据'],
                        kind=item['类型'], text=item['原文'], people=item['相关人物'],
                        need=item['修正要求'], owner='成稿', level=item['级别'])
        row['状态'] = '待人工裁定'
    for error in mechanical['错误']:
        add_issue(episode, issues, pending, source='成稿', position={k: v for k, v in error['位置'].items()
                  if k in ('场景', '行', '镜头', '时间', '句') and type(v) in (int, str)},
                  basis=error['检查项'] + '：' + error['问题'])
    unresolved = [r['id'] for r in [*pending, *issues] if not issue_closed(r)]
    shadow = bool(mechanical['通过'] and not unresolved and coverage.get('全文审查完成') is True
                  and coverage.get('输入未变') is True and review['结论'] == '通过' and not review['问题'])
    # 先把问题保存到发布事务中，不让旧空问题文件授权正式稿。
    real_pass = not simulated and shadow and check_review(episode)
    if not real_pass and not unresolved:
        basis = ('模拟独立审查只证明调度合同，不能授权正式稿' if simulated
                 else '缺少当前完整真实独立审查或本轮验收未通过')
        row = add_issue(episode, issues, pending, source='验收', position={}, basis=basis,
                        need='完成真实独立验收，或核对审查覆盖与证据', owner='验收')
        unresolved.append(row['id'])
    state = load(episode / '状态.json')
    state.update(状态='验收通过' if real_pass else '待人工裁定', 未决问题=unresolved)
    state['步骤'].update({str(i): True for i in range(1, 7)})
    state['步骤'].update({'7': False, '8': real_pass, '9': not real_pass})
    report = ('模拟运行，仅验证调度合同。\n' if simulated else '')
    report += '独立审查结论：' + review['结论'] + '\n画面覆盖完整：' + str(coverage.get('全文审查完成'))
    report += '\n机械检查：' + ('通过' if mechanical['通过'] else '未通过') + '\n'
    report += '终态：' + state['状态'] + '\n未决：' + '、'.join(unresolved) + '\n'
    result = {'通过': True, '状态': state['状态'], '正式交付': real_pass, '模拟运行': simulated,
              '离线影子终态': ('正式交付候选' if shadow else '待人工裁定') if simulated else None,
              '目录': str(episode), '草稿': str(episode / '工作稿.md'), '验收报告': str(episode / '验收报告.md'),
              '待核对': str(episode / '待核对.json'), '问题清单': str(episode / '问题清单.json')}
    writes = [(episode / '待核对.json', pending), (episode / '问题清单.json', issues),
              (episode / '验收报告.md', report), (episode / '状态.json', state)]
    if real_pass:
        # 正式文件尚未存在时，针对同一工作稿运行完整正式检查。
        checked = check_script(episode, '工作稿.md', formal=True)
        if not checked['通过']:
            raise ValueError('正式发布门禁失败：' + str(checked['错误']))
        writes.insert(0, (episode / '正式稿.md', (episode / '工作稿.md').read_text(encoding='utf-8')))
        result['Markdown'] = str(episode / '正式稿.md')
    publish(writes, result)
    return result
