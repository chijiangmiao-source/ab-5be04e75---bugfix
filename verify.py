"""Compose verify 服务的单次复核入口。

依次执行：
1. 代码测试（pytest 单元测试，含随机暴力交叉验证）；
2. 构建检查（语法编译、模块导入、镜像内关键文件齐备）；
3. API/HTTP 冒烟（健康路径 + 嵌套同优、交叉低价诱饵、空候选、非法引用、
   稀疏候选族全局几何与顺序无关性等场景）。

任一步失败即以非零退出码结束，全部通过退出码 0。
"""

from __future__ import annotations

import itertools
import json
import os
import py_compile
import random
import sys
import urllib.error
import urllib.request

API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
REQUIRE_FILES = os.environ.get("REQUIRE_BUILD_FILES", "").split(",")

failures: list[str] = []


def check(condition: bool, name: str, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {name}{(' - ' + detail) if detail and not condition else ''}")
    if not condition:
        failures.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------- 1. 代码测试


def run_unit_tests() -> None:
    section("代码测试 (pytest)")
    import pytest

    rc = pytest.main(["-q", "tests"])
    check(rc == 0, "pytest 单元测试", f"退出码 {rc}")


# ---------------------------------------------------------------- 2. 构建检查


def run_build_checks() -> None:
    section("构建检查")
    for path in ["solver.py", "app.py", "verify.py", os.path.join("tests", "test_solver.py")]:
        try:
            py_compile.compile(path, doraise=True)
            ok = True
        except py_compile.PyCompileError as exc:
            ok = False
            print(exc)
        check(ok, f"语法编译: {path}")

    for fname in REQUIRE_FILES:
        fname = fname.strip()
        if not fname:
            continue
        check(os.path.isfile(fname), f"镜像内文件齐备: {fname}")

    try:
        import solver  # noqa: F401
        import app  # noqa: F401

        ok = True
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(exc)
    check(ok, "模块导入 solver/app")


# ---------------------------------------------------------------- 3. API/HTTP 冒烟


def http_request(method: str, path: str, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        API_BASE + path, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def hits(n):
    return [{"id": f"h{k}", "position": k * 10} for k in range(n)]


def cand(cid, a, b, r):
    return {
        "id": cid,
        "left_endpoint": f"h{a}",
        "right_endpoint": f"h{b}",
        "residual": r,
    }


def sparse_bait_family():
    """72 击中、50 条候选的稀疏族：四条关键弧 + 46 条残差 100 填充弧。"""
    family_hits = [{"id": f"h{k}", "position": k} for k in range(72)]
    candidates = [
        cand("a-cross", 0, 30, 0),
        cand("z-outer", 0, 70, 10),
        cand("b-short", 10, 20, 10),
        cand("y-cross", 10, 49, 0),
    ]
    candidates += [cand(f"f0-{b}", 0, b, 100) for b in range(50, 70)]
    candidates += [
        cand(f"f10-{b}", 10, b, 100) for b in list(range(21, 30)) + list(range(31, 48))
    ]
    return {"hits": family_hits, "candidates": candidates}


def has_crossing(canonical_pairs, family_hits):
    pos = {h["id"]: h["position"] for h in family_hits}
    spans = [(pos[p["left_endpoint"]], pos[p["right_endpoint"]]) for p in canonical_pairs]
    return any(
        a < c < b < d or c < a < d < b
        for (a, b), (c, d) in itertools.combinations(spans, 2)
    )


def run_smoke() -> None:
    section(f"API/HTTP 冒烟 ({API_BASE})")

    status, body = http_request("GET", "/health")
    check(status == 200 and body.get("status") == "ready", "健康路径 /health 报告就绪",
          f"HTTP {status} {body}")

    # 场景 1：嵌套同优 —— 顺序配对与嵌套配对同为 2 对同残差。
    payload = {
        "hits": hits(4),
        "candidates": [
            cand("a_out", 0, 3, 1),
            cand("a_in", 1, 2, 5),
            cand("b_left", 0, 1, 3),
            cand("b_right", 2, 3, 3),
        ],
    }
    status, body = http_request("POST", "/audit", payload)
    ok = (
        status == 200
        and body["optimal_count"] == "2"
        and [p["id"] for p in body["canonical_pairs"]] == ["a_out", "a_in"]
        and set(body["classification"]["optional"])
        == {"a_out", "a_in", "b_left", "b_right"}
    )
    check(ok, "嵌套同优方案并列计数与规范解", f"HTTP {status} {body}")

    # 场景 2：交叉低价诱饵 —— 0 残差的交叉弧不得胜出。
    payload = {
        "hits": hits(6),
        "candidates": [
            cand("bait", 0, 3, 0),
            cand("inner", 1, 2, 10),
            cand("tail", 4, 5, 1),
            cand("seq01", 0, 1, 1),
            cand("seq23", 2, 3, 1),
        ],
    }
    status, body = http_request("POST", "/audit", payload)
    ok = (
        status == 200
        and [p["id"] for p in body["canonical_pairs"]] == ["seq01", "seq23", "tail"]
        and "bait" in body["classification"]["never"]
        and body["total_residual"] == 3
    )
    check(ok, "交叉低价诱饵不被接受", f"HTTP {status} {body}")

    # 场景 3：合法空候选 —— 唯一空方案。
    payload = {"hits": hits(4), "candidates": []}
    status, body = http_request("POST", "/audit", payload)
    ok = (
        status == 200
        and body["optimal_count"] == "1"
        and body["canonical_pairs"] == []
        and len(body["unmatched_hits"]) == 4
        and body["classification"] == {"required": [], "optional": [], "never": []}
    )
    check(ok, "合法空候选返回唯一空方案", f"HTTP {status} {body}")

    # 场景 4：非法引用 —— 错误带字段路径，且不夹带任何审计字段。
    payload = {"hits": hits(4), "candidates": [cand("bad", 0, 99, 0)]}
    # 99 不在 id 中，手工构造以模拟未知端点字符串。
    payload["candidates"][0]["right_endpoint"] = "h99"
    status, body = http_request("POST", "/audit", payload)
    ok = (
        status == 400
        and isinstance(body.get("errors"), list)
        and any(e.get("field") == "/candidates/0/right_endpoint" for e in body["errors"])
        and "optimal_count" not in body
    )
    check(ok, "非法引用返回字段路径错误且无审计结果", f"HTTP {status} {body}")

    # 附加：重复端点对、位置冲突、规模越界。
    status, body = http_request(
        "POST",
        "/audit",
        {"hits": hits(4), "candidates": [cand("a", 0, 1, 0), cand("b", 0, 1, 1)]},
    )
    check(
        status == 400
        and any(e["field"] == "/candidates/1" for e in body["errors"])
        and "optimal_count" not in body,
        "重复端点对被拒绝", f"HTTP {status} {body}",
    )

    conflict = hits(4)
    conflict[2]["position"] = conflict[1]["position"]
    status, body = http_request("POST", "/audit", {"hits": conflict, "candidates": []})
    check(
        status == 400
        and any(e["field"] == "/hits/2/position" for e in body["errors"]),
        "位置冲突被拒绝", f"HTTP {status} {body}",
    )

    status, body = http_request("POST", "/audit", {"hits": hits(3), "candidates": []})
    check(
        status == 400 and any(e["field"] == "/hits" for e in body["errors"]),
        "击中规模越界被拒绝", f"HTTP {status} {body}",
    )

    # 未知路径返回 404。
    status, _ = http_request("GET", "/nope")
    check(status == 404, "未知路径返回 404", f"HTTP {status}")

    # 场景 5：大批稀疏候选族 —— 全局非交叉约束不得因分量划分而丢失。
    # 交叉的 a-cross (0,30) 与 y-cross (10,49) 不得同时入选；
    # 正确结果：4 击中、残差 10、恰 2 个最优方案，规范解 a-cross + b-short。
    payload = sparse_bait_family()
    status, body = http_request("POST", "/audit", payload)
    canonical = body.get("canonical_pairs", [])
    ok = (
        status == 200
        and body["paired_hits"] == 4
        and body["total_residual"] == 10
        and body["optimal_count"] == "2"
        and [p["id"] for p in canonical] == ["a-cross", "b-short"]
        and not has_crossing(canonical, payload["hits"])
        and len(body["unmatched_hits"]) == 68
        and body["classification"]["required"] == []
        and set(body["classification"]["optional"])
        == {"a-cross", "z-outer", "b-short", "y-cross"}
        and len(body["classification"]["never"]) == 46
    )
    check(ok, "稀疏候选族保持全局非交叉约束", f"HTTP {status} {body}")

    # 场景 6：顺序无关性 —— 真实 HTTP 接口上多次打乱候选录入顺序，
    # 响应必须与基准逐字段一致。
    rng = random.Random(99)
    consistent = True
    for _ in range(3):
        shuffled = {"hits": payload["hits"], "candidates": payload["candidates"][:]}
        rng.shuffle(shuffled["candidates"])
        other_status, other = http_request("POST", "/audit", shuffled)
        if other_status != 200 or other != body:
            consistent = False
    check(consistent, "候选录入顺序无关（HTTP 重复检查）")


def main() -> int:
    run_unit_tests()
    run_build_checks()
    run_smoke()

    print("\n=== 汇总 ===")
    if failures:
        print(f"失败 {len(failures)} 项: {failures}")
        return 1
    print("全部复核通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
