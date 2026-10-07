"""发布下载的断流恢复与完整性负例，不连接外部网络。"""

import hashlib
import sys
import threading
import traceback
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from scripts import verify_release


@pytest.fixture
def release_files(tmp_path, monkeypatch):
    bodies = {
        "AegisAgent-source-v0.2.10.zip": b"source",
        "aegiscode-0.2.10-py3-none-any.whl": b"wheel",
        "AegisCode-portable.zip": b"portable",
        "AegisCode-Setup.exe": b"installer",
    }
    rows = []
    for name, body in bodies.items():
        (tmp_path / name).write_bytes(body)
        rows.append(hashlib.sha256(body).hexdigest() + "  " + name)
    manifest = ("\n".join(rows) + "\n").encode()
    (tmp_path / "SHA256SUMS.txt").write_bytes(manifest)
    monkeypatch.setattr(
        sys, "argv", ["verify_release", "--directory", str(tmp_path), "--tag", "v0.2.10"]
    )
    return {**bodies, "SHA256SUMS.txt": manifest}


def install_transport(monkeypatch, handler):
    client_type = httpx.Client
    monkeypatch.setattr(
        verify_release.httpx,
        "Client",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs),
    )


@pytest.fixture
def retry_delays(monkeypatch):
    delays = []
    monkeypatch.setattr(verify_release.time, "sleep", delays.append)
    return delays


@pytest.mark.parametrize("failing_name", ["SHA256SUMS.txt", "AegisCode-Setup.exe"])
def test_download_recovers_header_disconnect(
    release_files, monkeypatch, capsys, failing_name, retry_delays
):
    attempts = Counter()

    def response(request):
        name = request.url.path.rsplit("/", 1)[-1]
        assert "authorization" not in request.headers
        attempts[name] += 1
        if name == failing_name and attempts[name] < 3:
            raise httpx.RemoteProtocolError("模拟连接中断", request=request)
        return httpx.Response(200, content=release_files[name])

    install_transport(monkeypatch, response)
    verify_release.main()
    output = capsys.readouterr().out
    assert output.count("独立下载通过") == 4
    assert attempts[failing_name] == 3
    assert retry_delays == [1, 2]


class InterruptedBody(httpx.SyncByteStream):
    def __init__(self, body):
        self.body = body
        self.closed = False

    def __iter__(self):
        yield self.body[:2]
        raise httpx.ReadError("模拟正文中断")

    def close(self):
        self.closed = True


def test_partial_body_is_discarded_before_retry(release_files, monkeypatch, capsys):
    name = "AegisAgent-source-v0.2.10.zip"
    interrupted = InterruptedBody(release_files[name])
    attempts = Counter()

    def response(request):
        target = request.url.path.rsplit("/", 1)[-1]
        attempts[target] += 1
        if target == name and attempts[target] == 1:
            return httpx.Response(200, stream=interrupted)
        return httpx.Response(200, content=release_files[target])

    install_transport(monkeypatch, response)
    verify_release.main()
    assert interrupted.closed
    assert attempts[name] == 2
    assert capsys.readouterr().out.count("独立下载通过") == 4


def test_permanent_disconnect_stops_after_three_attempts(release_files, monkeypatch, retry_delays):
    attempts = []

    def response(request):
        attempts.append(request)
        raise httpx.ReadError("模拟持续断流", request=request)

    install_transport(monkeypatch, response)
    with pytest.raises(RuntimeError, match="已尝试3次"):
        verify_release.main()
    assert len(attempts) == 3
    assert retry_delays == [1, 2]


@pytest.mark.parametrize("bad_body", [b"wrong!", b"source-extra", b"sour"])
def test_corrupt_asset_is_never_retried(release_files, monkeypatch, bad_body):
    attempts = Counter()

    def response(request):
        name = request.url.path.rsplit("/", 1)[-1]
        attempts[name] += 1
        body = bad_body if name == "AegisAgent-source-v0.2.10.zip" else release_files[name]
        return httpx.Response(200, content=body)

    install_transport(monkeypatch, response)
    with pytest.raises(ValueError, match="公开文件"):
        verify_release.main()
    assert attempts["AegisAgent-source-v0.2.10.zip"] == 1


def test_http_failure_is_never_retried(release_files, monkeypatch):
    attempts = []

    def response(request):
        attempts.append(request)
        return httpx.Response(403)

    install_transport(monkeypatch, response)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        verify_release.main()
    assert len(attempts) == 1


def test_manifest_mismatch_is_never_retried(release_files, monkeypatch):
    attempts = []

    def response(request):
        attempts.append(request)
        return httpx.Response(200, content=b"bad manifest")

    install_transport(monkeypatch, response)
    with pytest.raises(ValueError, match="公开清单"):
        verify_release.main()
    assert len(attempts) == 1


@pytest.mark.parametrize("failure", ["http", "transport"])
def test_final_error_does_not_expose_signed_redirect(
    release_files, monkeypatch, retry_delays, failure
):
    signed_url = "https://download.example.invalid/asset?sig=synthetic-private-query"

    def response(request):
        if request.url.host == "github.com":
            return httpx.Response(302, headers={"Location": signed_url})
        if failure == "transport":
            raise httpx.ReadError("合成签名地址：" + signed_url, request=request)
        return httpx.Response(403)

    install_transport(monkeypatch, response)
    with pytest.raises(Exception) as caught:
        verify_release.main()
    rendered = "".join(traceback.format_exception(caught.value))
    assert "synthetic-private-query" not in rendered
    assert "download.example.invalid" not in rendered


def test_real_http_partial_response_recovers_with_full_digest():
    body = "真实有界产物".encode()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            assert self.headers.get("Authorization") is None
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[:2] if len(requests) == 1 else body)
            self.wfile.flush()
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            verify_release.verify_download(
                client,
                f"http://127.0.0.1:{server.server_port}/artifact",
                len(body),
                hashlib.sha256(body).hexdigest(),
                "验收产物",
            )
        assert requests == ["/artifact", "/artifact"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
