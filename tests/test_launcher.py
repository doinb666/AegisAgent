"""启动端口竞争和浏览器就绪门，使用真实套接字验证隔离边界。"""

import socket
import threading
from types import SimpleNamespace

import pytest

from app.launcher import bind_listener, browser_address, open_when_started


def test_busy_port_is_not_reused_and_opt_in_reserves_another_port():
    with bind_listener("127.0.0.1", 0) as occupied:
        occupied.listen()
        port = occupied.getsockname()[1]
        with pytest.raises(OSError):
            bind_listener("127.0.0.1", port)
        with bind_listener("127.0.0.1", port, auto_port=True) as selected:
            assert selected.getsockname()[1] != port
            selected.listen()
            with socket.create_connection(selected.getsockname(), timeout=1):
                pass


def test_auto_port_does_not_hide_an_invalid_host():
    with pytest.raises(OSError):
        bind_listener("256.256.256.256", 8000, auto_port=True)


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("0.0.0.0", "http://127.0.0.1:8000"),
        ("::", "http://[::1]:8000"),
        ("::1", "http://[::1]:8000"),
        ("127.0.0.1", "http://127.0.0.1:8000"),
    ],
)
def test_browser_addresses(host, expected):
    assert browser_address(host, 8000) == expected


def test_browser_waits_for_own_server_startup(monkeypatch):
    opened = threading.Event()
    server = SimpleNamespace(started=False, should_exit=False)
    stopped = threading.Event()
    monkeypatch.setattr("app.launcher.webbrowser.open", lambda _: opened.set() or True)
    worker = threading.Thread(target=open_when_started, args=(server, "http://test", stopped))
    worker.start()
    try:
        assert not opened.wait(0.15)
        server.started = True
        assert opened.wait(1)
    finally:
        stopped.set()
        worker.join(timeout=1)
    assert not worker.is_alive()


@pytest.mark.parametrize("failed", [False, True])
def test_exit_or_failed_startup_never_opens_browser(monkeypatch, failed):
    stopped = threading.Event()
    if not failed:
        stopped.set()
    monkeypatch.setattr("app.launcher.webbrowser.open", lambda _: pytest.fail("不应打开页面"))
    open_when_started(SimpleNamespace(started=False, should_exit=failed), "http://test", stopped)


def test_startup_timeout_leaves_service_running(monkeypatch, caplog):
    server = SimpleNamespace(started=False, should_exit=False)
    monkeypatch.setattr("app.launcher.webbrowser.open", lambda _: pytest.fail("不应打开页面"))
    open_when_started(server, "http://test", threading.Event(), timeout=0)
    assert not server.should_exit
    assert "手动访问" in caplog.text


@pytest.mark.parametrize("raises", [False, True])
def test_browser_failure_has_actionable_message(monkeypatch, caplog, raises):
    def unavailable(_):
        if raises:
            raise OSError("浏览器不可用")
        return False

    monkeypatch.setattr("app.launcher.webbrowser.open", unavailable)
    open_when_started(
        SimpleNamespace(started=True, should_exit=False), "http://test", threading.Event()
    )
    assert "手动访问 http://test" in caplog.text
