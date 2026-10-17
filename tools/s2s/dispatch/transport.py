"""在既有调用接口外封装 pi 工具白名单、会话和 JSON 事件。"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import sys
import tempfile

from s2s import config


@dataclass
class Response:
    text: str
    usage: dict = field(default_factory=dict)
    reads: list[str] = field(default_factory=list)


def decode_response(raw: str) -> Response:
    events = []
    for line in raw.split('\n'):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            return Response(raw)
        if not isinstance(event, dict) or 'type' not in event:
            return Response(raw)
        events.append(event)
    messages, reads, usage, pending_reads = [], [], {}, {}
    for event in events:
        if event.get('type') == 'message_end' and event.get('message', {}).get('role') == 'assistant':
            message = event['message']
            if message.get('stopReason') in ('error', 'aborted'):
                raise RuntimeError(message.get('errorMessage') or 'pi 会话中止')
            text = ''.join(c.get('text', '') for c in message.get('content', []) if c.get('type') == 'text')
            if text:
                messages.append(text)
            for key, value in message.get('usage', {}).items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
        if event.get('type') == 'tool_execution_start' and event.get('toolName') == 'read':
            pending_reads[event.get('toolCallId')] = event.get('args', {}).get('path', '')
        if event.get('type') == 'tool_execution_end' and not event.get('isError'):
            path = pending_reads.pop(event.get('toolCallId'), None)
            if path:
                reads.append(path)
    if not messages:
        raise ValueError('pi 未返回最终文本')
    return Response(messages[-1], usage, reads)


@contextmanager
def pi_options(settings: dict, *, read_paths=(), session_dir: Path | None = None, controller=False):
    """临时可执行适配器直接 exec pi，超时终止的是 pi 本身。"""
    if settings.get('runner', 'pi') != 'pi':
        if controller or read_paths:
            raise ValueError('工具读取和主控会话要求 pi 运行器')
        yield {}
        return
    executable = settings.get('bin') or config.get_runner_config('pi').get('bin', 'pi')
    executable = shutil.which(executable) or executable
    flags = ['--mode', 'json', '--no-extensions', '--no-mcp', '--no-skills',
             '--no-prompt-templates', '--no-themes', '--no-approve']
    if controller:
        flags += ['--tools', 'read,bash,edit,write']
    elif read_paths:
        flags += ['--tools', 'read', '--extension', str(Path(__file__).with_name('read_guard.ts'))]
    else:
        flags += ['--no-tools']
    if session_dir is not None:
        session_dir.mkdir(parents=True, exist_ok=True)
        flags += ['--session-dir', str(session_dir.resolve())]
    with tempfile.TemporaryDirectory(prefix='s2s-pi-') as folder:
        wrapper = Path(folder) / 'pi-adapter'
        wrapper.write_text(
            f'#!{sys.executable}\nimport os,sys\n'
            f'os.environ["S2S_DISPATCH_READ_PATHS"] = {json.dumps([str(Path(p).resolve()) for p in read_paths])!r}\n'
            'args = sys.argv[1:]\n'
            + ('args.remove("--no-session")\n' if session_dir is not None else '')
            + f'args = args[:1] + {flags!r} + args[1:]\n'
            + f'os.execv({str(executable)!r}, [{str(executable)!r}, *args])\n', encoding='utf-8')
        wrapper.chmod(0o700)
        yield {'bin': str(wrapper), 'tools': bool(controller or read_paths)}
