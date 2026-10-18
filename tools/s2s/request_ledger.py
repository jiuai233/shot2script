"""每个实际HTTP请求的预算边界与账本；未知用量/金额不伪装成零。"""

from datetime import datetime, timezone
from decimal import Decimal
import math
from pathlib import Path
import threading

from .run_record import load, now, save


class BudgetBlocked(RuntimeError):
    pass


class RequestLedger:
    """计价接口须提供preflight、reserve上界、settle官方结算。

    E0未内置价格表或账单适配器，真实运行预检关闭。测试可注入有依据的
    计价替身证明所有初答、格式重答、网络重试均经过同一边界。
    """

    def __init__(self, path, *, simulated=False, max_requests=100, max_seconds=900,
                 warning_usd='2', stop_usd='5', costing=None):
        if type(max_requests) is not int or max_requests < 1 or not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError('请求数与运行秒数上限须为正数')
        self.path, self.simulated, self.costing = Path(path), simulated, costing
        self.lock = threading.Lock()
        policy = {'最大请求数': max_requests, '最大墙钟秒': max_seconds,
                  '警告美元': str(Decimal(warning_usd)), '停止美元': str(Decimal(stop_usd)), '模拟运行': simulated}
        if not Decimal(policy['停止美元']).is_finite() or not 0 <= Decimal(policy['警告美元']) < Decimal(policy['停止美元']):
            raise ValueError('美元阈值须有限，且0 <= 警告 < 停止')
        self.data = load(self.path) if self.path.exists() else {
            '版本': 1, '开始': now(), '策略': policy, '请求': [], '已结算美元': None,
            '真实远程请求数': 0, '模拟调用数': 0, '金额完整': False}
        if self.data['策略'] != policy:
            raise BudgetBlocked('运行预算策略已变化；请使用新的运行目录')

    def preflight(self):
        if self.simulated:
            return {'通过': True, '模拟运行': True, '真实远程请求数': 0}
        if self.costing is None:
            raise BudgetBlocked('真实调用预检阻断：尚无可验证的单请求费用上界与官方结算适配器；'
                                'E0不承诺美元硬限，不启动预处理或模型。先补齐计价/账单接入后再进入E1')
        self.costing.preflight()
        return {'通过': True, '模拟运行': False}

    def scoped(self, stage):
        return RequestScope(self, stage)

    def begin(self, metadata):
        with self.lock:
            self.preflight()
            rows, policy = self.data['请求'], self.data['策略']
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(self.data['开始'])).total_seconds()
            reason = self.data.get('阻塞', {}).get('原因')
            if not reason and len(rows) >= policy['最大请求数']:
                reason = '已达到最大请求数；停止全部新请求（包括初答与重试）'
            elif not reason and elapsed >= policy['最大墙钟秒']:
                reason = '已达到墙钟预算；停止全部新请求'
            elif not reason and not self.simulated and any(r.get('实际美元') is None for r in rows if r['状态'] != '运行中'):
                reason = '已有请求费用未知；停止全部新请求，等待核账'
            reservation = None
            if not reason and not self.simulated:
                reservation = Decimal(str(self.costing.reserve(metadata)))
                if not reservation.is_finite() or reservation < 0:
                    reason = '单请求费用上界无效'
                else:
                    committed = sum(Decimal(r.get('实际美元') or r['预留美元']) for r in rows)
                    if committed + reservation > Decimal(policy['停止美元']):
                        reason = '本次请求可能越过累计停止金额；停止全部新请求'
            if reason:
                self.data['阻塞'] = {'时间': now(), '原因': reason, '阶段': metadata.get('阶段')}
                save(self.path, self.data)
                raise BudgetBlocked(reason)
            ticket = len(rows)
            rows.append({'序号': ticket + 1, **metadata, '开始': now(), '状态': '运行中',
                         '模拟调用': self.simulated, '预留美元': str(reservation) if reservation is not None else None,
                         '官方usage': None, '实际美元': None})
            key = '模拟调用数' if self.simulated else '真实远程请求数'
            self.data[key] += 1
            save(self.path, self.data)
            return ticket

    def remaining_seconds(self):
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(self.data['开始'])).total_seconds()
        return max(0.001, self.data['策略']['最大墙钟秒'] - elapsed)

    def finish(self, ticket, *, usage=None, error=None):
        with self.lock:
            row = self.data['请求'][ticket]
            if row['状态'] != '运行中':
                raise ValueError('同一请求不得重复结算')
            cost = None if self.simulated else self.costing.settle(usage, error)
            if cost is not None:
                cost = Decimal(str(cost))
                if not cost.is_finite() or cost < 0:
                    cost = None
            row.update(结束=now(), 状态='失败' if error else '完成', 官方usage=usage,
                       实际美元=str(cost) if cost is not None else None, 错误=error)
            row['墙钟秒'] = round((datetime.fromisoformat(row['结束']) - datetime.fromisoformat(row['开始'])).total_seconds(), 6)
            settled = [Decimal(r['实际美元']) for r in self.data['请求'] if r['实际美元'] is not None]
            complete = bool(self.data['请求']) and len(settled) == len(self.data['请求'])
            self.data.update(金额完整=complete, 已结算美元=str(sum(settled)) if complete else None)
            known = sum(settled)
            self.data['已知金额小计美元'] = str(known) if settled else None
            self.data['金额警告'] = known >= Decimal(self.data['策略']['警告美元'])
            if cost is not None and cost > Decimal(row['预留美元']):
                self.data['阻塞'] = {'时间': now(), '原因': '结算超过已声明单请求上界，计价适配器失效'}
            save(self.path, self.data)


class RequestScope:
    def __init__(self, ledger, stage):
        self.ledger, self.stage = ledger, stage

    def begin(self, metadata):
        return self.ledger.begin({**metadata, '阶段': self.stage})

    def finish(self, ticket, **kwargs):
        return self.ledger.finish(ticket, **kwargs)

    def remaining_seconds(self):
        return self.ledger.remaining_seconds()
