"""预算使用明确测试计价替身；没有真实价格、凭据或远程请求。"""

from concurrent.futures import ThreadPoolExecutor
import io
import json
import urllib.error

import pytest

from s2s import runners
from s2s.board import initialize_board
from s2s.dispatch.roles import dispatch
from s2s.offline_episode import OfflineEpisode
from s2s.request_ledger import BudgetBlocked, RequestLedger
from s2s.run_record import load
from test_fixed_episode import FIXTURES, VIDEO


class TestCosting:
    __test__ = False
    def __init__(self, reservation='1', cost='0.5'):
        self.reservation, self.cost = reservation, cost
    def preflight(self):
        pass
    def reserve(self, metadata):
        return self.reservation
    def settle(self, usage, error):
        return '0' if error == 'HTTP 429' else self.cost


@pytest.fixture
def http(monkeypatch):
    calls, answers = [], []
    monkeypatch.setattr(runners, 'get_environment', lambda: {'DEEPSEEK_API_KEY': 'test-only-not-a-real-key'})
    monkeypatch.setattr(runners, 'get_runner_config', lambda name: {'base_url': 'https://example.invalid'})
    monkeypatch.setattr(runners.time, 'sleep', lambda seconds: None)
    def urlopen(request, timeout):
        calls.append(request)
        value = answers.pop(0)
        if isinstance(value, Exception):
            raise value
        return io.BytesIO(json.dumps(value).encode())
    monkeypatch.setattr(runners.urllib.request, 'urlopen', urlopen)
    return calls, answers


def response(text='{}', usage=None):
    value = {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}
    if usage is not None:
        value['usage'] = usage
    return value


def execute(ledger):
    return runners.run('测试', model='deepseek/deepseek-flash', thinking='high',
                       runner='deepseek', events=True, request_budget=ledger.scoped('测试阶段'))


def test_unconfigured_real_pricing_is_closed_at_preflight(tmp_path):
    ledger = RequestLedger(tmp_path / 'ledger.json')
    with pytest.raises(BudgetBlocked, match='真实调用预检阻断'):
        ledger.preflight()
    assert not ledger.path.exists()


def test_unknown_usage_is_not_zero(tmp_path, http):
    calls, answers = http
    answers.append(response())
    ledger = RequestLedger(tmp_path / 'ledger.json', costing=TestCosting(cost=None))
    event = json.loads(execute(ledger))
    assert event['message']['usage'] == {}
    row = load(ledger.path)['请求'][0]
    assert row['官方usage'] is None and row['实际美元'] is None
    assert load(ledger.path)['已结算美元'] is None
    with pytest.raises(BudgetBlocked, match='费用未知'):
        execute(ledger)
    assert len(calls) == 1


def test_raw_official_usage_is_preserved_without_invented_fields(tmp_path, http):
    _, answers = http
    usage = {'prompt_tokens': 11, 'completion_tokens': 4, 'custom_provider_field': 3}
    answers.append(response(usage=usage))
    ledger = RequestLedger(tmp_path / 'ledger.json', costing=TestCosting())
    event = json.loads(execute(ledger))
    assert event['message']['usage'] == {'promptTokens': 11, 'output': 4}
    assert load(ledger.path)['请求'][0]['官方usage'] == usage
    assert load(ledger.path)['已结算美元'] == '0.5'


def test_every_http_retry_is_checked_before_network(tmp_path, http):
    calls, answers = http
    answers.append(urllib.error.HTTPError('https://example.invalid', 429, 'retry', {}, io.BytesIO(b'limited')))
    ledger = RequestLedger(tmp_path / 'ledger.json', max_requests=1, costing=TestCosting())
    with pytest.raises(BudgetBlocked, match='全部新请求'):
        execute(ledger)
    rows = load(ledger.path)['请求']
    assert len(calls) == len(rows) == 1
    assert rows[0]['错误'] == 'HTTP 429' and rows[0]['实际美元'] == '0'


def test_every_format_retry_is_checked_before_network(tmp_path, http):
    calls, answers = http
    answers.append(response('{}', {'total_tokens': 4}))
    initialize_board('测试', 1, root=tmp_path)
    episode = tmp_path / '测试/E01'
    OfflineEpisode(FIXTURES / 'pass.json').prepare(VIDEO, episode)
    ledger = RequestLedger(tmp_path / 'ledger.json', max_requests=1, costing=TestCosting())
    with pytest.raises(BudgetBlocked, match='全部新请求'):
        dispatch('动作识别', episode, {'片段号': 'E01-P01'}, episode / 'action.json', request_budget=ledger)
    assert len(calls) == 1
    assert load(ledger.path)['请求'][0]['阶段'].endswith('格式尝试1')


def test_budget_covers_new_stage_initial_request_not_just_retries(tmp_path):
    ledger = RequestLedger(tmp_path / 'ledger.json', warning_usd='0.5', stop_usd='1', costing=TestCosting('0.6', '0.5'))
    first = ledger.scoped('动作').begin({'模型': 'deepseek/deepseek-flash'})
    ledger.finish(first, usage={'test': True})
    with pytest.raises(BudgetBlocked, match='累计停止金额'):
        ledger.scoped('全新成稿初答').begin({'模型': 'deepseek/deepseek-flash'})
    assert load(ledger.path)['金额警告'] is True
    assert len(load(ledger.path)['请求']) == 1


def test_concurrent_requests_reserve_against_same_remaining_budget(tmp_path):
    ledger = RequestLedger(tmp_path / 'ledger.json', costing=TestCosting('3', '1'))
    def begin(index):
        try:
            return ledger.scoped(str(index)).begin({'模型': 'deepseek/deepseek-flash'})
        except BudgetBlocked:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        tickets = list(pool.map(begin, range(8)))
    assert sum(ticket is not None for ticket in tickets) == 1
    assert load(ledger.path)['真实远程请求数'] == 1


def test_pi_cannot_hide_multiple_requests_behind_one_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(runners.subprocess, 'run', lambda *a, **k: pytest.fail('预算路径不能启动Pi'))
    ledger = RequestLedger(tmp_path / 'ledger.json', costing=TestCosting())
    with pytest.raises(runners.RunnerError, match='逐HTTP请求'):
        runners.run('测试', model='deepseek/deepseek-flash', thinking='high', runner='pi', request_budget=ledger.scoped('不支持'))


def test_simulated_calls_keep_unknown_money_separate(tmp_path):
    ledger = RequestLedger(tmp_path / 'ledger.json', simulated=True)
    ticket = ledger.scoped('模拟').begin({'图片数': 2})
    ledger.finish(ticket)
    data = load(ledger.path)
    assert data['真实远程请求数'] == 0 and data['模拟调用数'] == 1
    assert data['已结算美元'] is None and data['请求'][0]['官方usage'] is None
