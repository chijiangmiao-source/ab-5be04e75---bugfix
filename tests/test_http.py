"""HTTP 层测试：在随机端口上线程内启动真实 app，走 urllib 调用。"""

from __future__ import annotations

import json
import random
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from app import AuditHandler

from tests.family import crossing_pairs, sparse_bait_family


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), AuditHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def request(base, method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_health(server):
    status, body = request(server, "GET", "/health")
    assert status == 200
    assert body == {"status": "ready", "service": "track-pair-audit"}


def test_audit_ok(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [
            {"id": "p", "left_endpoint": "h0", "right_endpoint": "h3", "residual": 2},
            {"id": "q", "left_endpoint": "h1", "right_endpoint": "h2", "residual": 1},
        ],
    }
    status, body = request(server, "POST", "/audit", payload)
    assert status == 200
    assert body["optimal_count"] == "1"
    assert [p["id"] for p in body["canonical_pairs"]] == ["p", "q"]
    assert body["unmatched_hits"] == []


def test_audit_error_shape_has_no_audit_fields(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [
            {"id": "x", "left_endpoint": "h0", "right_endpoint": "nope", "residual": 0}
        ],
    }
    status, body = request(server, "POST", "/audit", payload)
    assert status == 400
    assert set(body.keys()) == {"errors"}
    assert body["errors"][0]["field"] == "/candidates/0/right_endpoint"


def test_bad_json(server):
    req = urllib.request.Request(
        server + "/audit", data=b"{not json", method="POST"
    )
    req.add_header("Content-Type", "application/json")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 400


def test_unknown_route(server):
    status, _ = request(server, "GET", "/")
    assert status == 404


def test_sparse_family_over_http_order_independent(server):
    # 50 条稀疏候选族：全局非交叉约束在真实 HTTP 接口上同样成立，
    # 且多次调整候选录入顺序（含逆序）响应逐字节一致。
    payload = sparse_bait_family()
    status, baseline = request(server, "POST", "/audit", payload)
    assert status == 200
    assert baseline["paired_hits"] == 4
    assert baseline["total_residual"] == 10
    assert baseline["optimal_count"] == "2"
    assert [p["id"] for p in baseline["canonical_pairs"]] == ["a-cross", "b-short"]
    assert crossing_pairs(baseline["canonical_pairs"], payload["hits"]) == []
    assert baseline["classification"]["required"] == []
    assert set(baseline["classification"]["optional"]) == {
        "a-cross",
        "z-outer",
        "b-short",
        "y-cross",
    }

    orders = [list(reversed(payload["candidates"]))]
    rng = random.Random(20260924)
    for _ in range(4):
        shuffled = payload["candidates"][:]
        rng.shuffle(shuffled)
        orders.append(shuffled)
    for candidates in orders:
        status, body = request(
            server, "POST", "/audit", {"hits": payload["hits"], "candidates": candidates}
        )
        assert status == 200
        assert body == baseline
