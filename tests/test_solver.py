"""solver 的单元测试：固定场景 + 随机暴力交叉验证 + 边界/校验测试。"""

from __future__ import annotations

import itertools
import random

import pytest

from solver import MAX_CANDIDATES, MAX_HITS, MIN_HITS, ValidationError, audit

from tests.brute import brute_solve
from tests.scenarios import (
    N_HITS,
    assert_canonical_pairs_noncrossing,
    assert_family_result,
    make_family_payload,
    result_signature,
)


def make_hits(n, prefix="h"):
    return [{"id": f"{prefix}{k}", "position": k * 10} for k in range(n)]


def cand(cid, left, right, residual):
    return {
        "id": cid,
        "left_endpoint": left,
        "right_endpoint": right,
        "residual": residual,
    }


def to_arc_records(candidates, hit_ids):
    pos = {h: k for k, h in enumerate(hit_ids)}
    return [
        (c["id"], pos[c["left_endpoint"]], pos[c["right_endpoint"]], c["residual"])
        for c in candidates
    ]


# ---------------------------------------------------------------- 固定场景


def test_empty_candidates_single_empty_solution():
    n = MIN_HITS
    res = audit({"hits": make_hits(n), "candidates": []})
    assert res["optimal_count"] == "1"
    assert res["canonical_pairs"] == []
    assert res["unmatched_hits"] == [f"h{k}" for k in range(n)]
    assert res["classification"] == {"required": [], "optional": [], "never": []}
    assert res["paired_hits"] == 0
    assert res["total_residual"] == 0


def test_nested_pairs_preferred_and_ids_lexicographic():
    # 4 个击中：并列对 (0,1)+(2,3) 与嵌套对 (0,3)+(1,2) 都是 2 对且同残差。
    hits = make_hits(4)
    candidates = [
        cand("a_out", "h0", "h3", 1),
        cand("a_in", "h1", "h2", 5),
        cand("s_left", "h0", "h1", 3),
        cand("s_right", "h2", "h3", 3),
    ]
    res = audit({"hits": hits, "candidates": candidates})
    assert res["optimal_count"] == "2"
    # 两个方案残差均为 6，嵌套方案 id 序列 [a_out, a_in] 更小。
    assert [p["id"] for p in res["canonical_pairs"]] == ["a_out", "a_in"]
    assert res["unmatched_hits"] == []
    assert res["classification"]["required"] == []
    assert set(res["classification"]["optional"]) == {
        "a_out",
        "a_in",
        "s_left",
        "s_right",
    }


def test_crossing_pairs_excluded_cheap_bait():
    # 交叉低价诱饵：bait=(0,3) 残差 0，看似该选，但其 3 对方案被迫搭配
    # 高价内弧 (1,2)；真正的最优是顺序相邻配对。
    hits = make_hits(6)
    candidates = [
        cand("bait", "h0", "h3", 0),
        cand("inner", "h1", "h2", 10),
        cand("tail", "h4", "h5", 1),
        cand("seq01", "h0", "h1", 1),
        cand("seq23", "h2", "h3", 1),
    ]
    res = audit({"hits": hits, "candidates": candidates})
    assert res["paired_hits"] == 6
    assert res["total_residual"] == 3
    assert [p["id"] for p in res["canonical_pairs"]] == ["seq01", "seq23", "tail"]
    assert set(res["classification"]["never"]) == {"bait", "inner"}
    assert res["classification"]["required"] == ["seq01", "seq23", "tail"]


def test_residual_breaks_tie():
    hits = make_hits(4)
    candidates = [
        cand("x", "h0", "h3", 10),
        cand("y", "h1", "h2", 0),
        cand("p", "h0", "h1", 1),
        cand("q", "h2", "h3", 1),
    ]
    res = audit({"hits": hits, "candidates": candidates})
    # 嵌套方案残差 10，并列方案残差 2。
    assert res["optimal_count"] == "1"
    assert [p["id"] for p in res["canonical_pairs"]] == ["p", "q"]
    assert res["classification"]["required"] == ["p", "q"]
    assert set(res["classification"]["never"]) == {"x", "y"}


def test_required_arc():
    hits = make_hits(5)
    candidates = [
        cand("must", "h0", "h4", 0),
        cand("mid", "h1", "h3", 0),
        cand("only_other", "h2", "h3", 5),
    ]
    res = audit({"hits": hits, "candidates": candidates})
    # 2 对方案必须选 must+mid；其他组合至多 1 对。
    assert res["optimal_count"] == "1"
    assert "must" in res["classification"]["required"]
    assert "mid" in res["classification"]["required"]
    assert "only_other" in res["classification"]["never"]


def test_multiple_matchings_count_and_tiebreak():
    # 4 点上两个不同的完美匹配：顺序 (0,1)+(2,3) 与嵌套 (0,3)+(1,2)。
    hits = make_hits(4)
    candidates = [
        cand("p01", "h0", "h1", 0),
        cand("p23", "h2", "h3", 0),
        cand("out", "h0", "h3", 0),
        cand("in", "h1", "h2", 0),
    ]
    res = audit({"hits": hits, "candidates": candidates})
    assert res["optimal_count"] == "2"
    # 嵌套方案按左端点顺序为 [out, in]；与顺序方案 [p01, p23] 比字典序，
    # 'out' < 'p01'，故规范解为 [out, in]。
    assert [p["id"] for p in res["canonical_pairs"]] == ["out", "in"]
    assert res["classification"]["required"] == []
    assert set(res["classification"]["optional"]) == {"p01", "p23", "out", "in"}


def test_arbitrary_precision_count():
    # 每 3 个击中一组，组内三条候选弧均为最优 1 对，组间独立 => 3^(n/3) 个方案。
    n = 180
    hits = make_hits(n)
    candidates = []
    for a in range(0, n, 3):
        candidates.append(cand(f"span{a}", f"h{a}", f"h{a+2}", 0))
        candidates.append(cand(f"adj1{a}", f"h{a}", f"h{a+1}", 0))
        candidates.append(cand(f"adj2{a}", f"h{a+1}", f"h{a+2}", 0))
    res = audit({"hits": hits, "candidates": candidates})
    assert res["optimal_count"] == str(3 ** (n // 3))
    # 规范解每组选 id 字典序最小的 adj1*（每组右端点未配对）。
    assert [p["id"] for p in res["canonical_pairs"]] == [
        f"adj1{a}" for a in range(0, n, 3)
    ]
    unmatched = [f"h{k}" for k in range(n) if k % 3 == 2]
    assert res["unmatched_hits"] == unmatched
    assert set(res["classification"]["optional"]) == {c["id"] for c in candidates}


def test_unmatched_reported():
    hits = make_hits(6)
    candidates = [cand("m", "h1", "h4", 0)]
    res = audit({"hits": hits, "candidates": candidates})
    assert res["unmatched_hits"] == ["h0", "h2", "h3", "h5"]


# ------------------------------------------- 大批稀疏候选的全局几何约束


def test_sparse_family_global_geometry():
    # 72 击中 / 50 候选：a-cross(0-30,残差0) 与 y-cross(10-49,残差0)
    # 满足 0 < 10 < 30 < 49，绘制后交叉，不能同时入选。正确最优为
    # 配对 4、总残差 10、恰好两种最优方案，规范解选 a-cross + b-short。
    payload = make_family_payload()
    assert len(payload["hits"]) == N_HITS
    assert len(payload["candidates"]) == 50
    res = audit(payload)
    assert_family_result(res)


@pytest.mark.parametrize("seed", range(8))
def test_sparse_family_order_independence(seed):
    # 调整候选录入顺序不得改变目标值、方案数、规范解、未配对集合与归属。
    payload = make_family_payload()
    baseline = audit(payload)
    rng = random.Random(seed)
    shuffled = payload["candidates"][:]
    rng.shuffle(shuffled)
    reordered = audit({"hits": payload["hits"], "candidates": shuffled})
    assert result_signature(reordered) == result_signature(baseline)
    assert_family_result(reordered)


def test_sparse_family_both_optima_noncrossing():
    # 两种最优方案分别为 {a-cross,b-short} 与 {z-outer,y-cross}，
    # 逐对验证二者的弧都不交叉；交叉组合 {a-cross,y-cross} 残差虽为 0
    # 却不是合法径迹解释。
    payload = make_family_payload()
    res = audit(payload)

    def pairs_of(ids):
        by_id = {c["id"]: c for c in payload["candidates"]}
        return [by_id[cid] for cid in ids]

    for ids in (["a-cross", "b-short"], ["y-cross", "z-outer"]):
        assert_canonical_pairs_noncrossing(pairs_of(ids))

    a = next(p for p in res["canonical_pairs"] if p["id"] == "a-cross")
    assert a["left_endpoint"] == "h0" and a["right_endpoint"] == "h30"
    # 交叉的廉价组合绝不能成为规范解。
    assert [p["id"] for p in res["canonical_pairs"]] != ["a-cross", "y-cross"]


def test_dense_candidates_past_threshold_match_bruteforce():
    # 56 条候选跨过旧的 48 条分解阈值：组件 A（11 个端点、55 条全连弧）
    # 的跨度包含组件 B 的独立弧 (3,7)，但 A 的弧 (0,5) 与 B 交叉。
    # 旧实现按跨度包含关系不合并组件，会把交叉弧各自取局部最优后笛卡尔合并。
    n = 13
    isolated = {3, 7}
    group_a = [k for k in range(n) if k not in isolated]
    arcs = []
    for i, left in enumerate(group_a):
        for right in group_a[i + 1 :]:
            residual = 0 if (left, right) == (0, 5) else 50
            arcs.append((f"A-{left}-{right}", left, right, residual))
    arcs.append(("B-cross", 3, 7, 0))
    assert len(arcs) > 48

    hits = make_hits(n)
    payload = {
        "hits": hits,
        "candidates": [
            cand(cid, f"h{left}", f"h{right}", residual)
            for cid, left, right, residual in arcs
        ],
    }
    res = audit(payload)
    ref = brute_solve(n, arcs)

    assert int(res["optimal_count"]) == ref["optimal_count"]
    assert res["paired_hits"] == 2 * ref["max_pairs"]
    assert res["total_residual"] == ref["min_cost"]
    assert [p["id"] for p in res["canonical_pairs"]] == ref["canonical"]
    assert res["classification"] == ref["classification"]
    # 交叉的诱饵弧与 B 弧不得共同出现在规范解中。
    assert not (
        {"A-0-5", "B-cross"} <= {p["id"] for p in res["canonical_pairs"]}
    )
    assert_canonical_pairs_noncrossing(res["canonical_pairs"])


# ---------------------------------------------------------------- 校验错误


def invalid_payload(payload):
    with pytest.raises(ValidationError) as exc:
        audit(payload)
    return exc.value.errors


def test_errors_have_field_paths_and_no_audit_leak():
    errors = invalid_payload({"hits": make_hits(3), "candidates": []})
    assert any(e["field"] == "/hits" for e in errors)

    bad_hits = make_hits(4)
    bad_hits[2]["position"] = bad_hits[1]["position"]
    errors = invalid_payload({"hits": bad_hits, "candidates": []})
    assert any(e["field"] == "/hits/2/position" for e in errors)

    errors = invalid_payload(
        {
            "hits": make_hits(4),
            "candidates": [cand("c", "h0", "ghost", 0)],
        }
    )
    assert any(e["field"] == "/candidates/0/right_endpoint" for e in errors)

    errors = invalid_payload(
        {
            "hits": make_hits(4),
            "candidates": [
                cand("c", "h0", "h1", 0),
                cand("d", "h0", "h1", 1),
            ],
        }
    )
    assert any("重复端点对" in e["message"] and e["field"] == "/candidates/1" for e in errors)

    errors = invalid_payload(
        {"hits": make_hits(4), "candidates": [cand("c", "h2", "h1", 0)]}
    )
    assert any(e["field"] == "/candidates/0/right_endpoint" for e in errors)

    errors = invalid_payload(
        {"hits": make_hits(4), "candidates": [cand("c", "h0", "h1", -1)]}
    )
    assert any(e["field"] == "/candidates/0/residual" for e in errors)

    errors = invalid_payload({"hits": make_hits(4), "candidates": [cand("c", "h0", "h1", True)]})
    assert any(e["field"] == "/candidates/0/residual" for e in errors)

    errors = invalid_payload({"hits": [1, 2, 3, 4], "candidates": []})
    assert any(e["field"] == "/hits/0" for e in errors)

    errors = invalid_payload("not-an-object")
    assert errors[0]["field"] == ""


def test_scale_limits():
    too_few = make_hits(MIN_HITS - 1)
    errors = invalid_payload({"hits": too_few, "candidates": []})
    assert any(e["field"] == "/hits" for e in errors)

    too_many = make_hits(MAX_HITS + 1)
    errors = invalid_payload({"hits": too_many, "candidates": []})
    assert any(e["field"] == "/hits" for e in errors)

    big_hits = make_hits(MAX_HITS)
    many_cands = []
    seen = set()
    rng = random.Random(0)
    while len(many_cands) < MAX_CANDIDATES + 1:
        a = rng.randrange(MAX_HITS - 1)
        b = rng.randrange(a + 1, MAX_HITS)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        many_cands.append(cand(f"c{len(many_cands)}", f"h{a}", f"h{b}", rng.randrange(100)))
    errors = invalid_payload({"hits": big_hits, "candidates": many_cands})
    assert any(e["field"] == "/candidates" for e in errors)


def test_duplicate_hit_id_and_candidate_id():
    hits = make_hits(4)
    hits[2]["id"] = "h0"
    errors = invalid_payload({"hits": hits, "candidates": []})
    assert any(e["field"] == "/hits/2/id" for e in errors)

    errors = invalid_payload(
        {
            "hits": make_hits(4),
            "candidates": [
                cand("same", "h0", "h1", 0),
                cand("same", "h1", "h2", 0),
            ],
        }
    )
    assert any(e["field"] == "/candidates/1/id" for e in errors)


# ---------------------------------------------------------------- 随机暴力对照


@pytest.mark.parametrize("seed", range(60))
def test_matches_bruteforce(seed):
    rng = random.Random(seed)
    n = rng.randint(MIN_HITS, 9)
    ids = [f"h{k}" for k in range(n)]

    # 随机选约 40% 的可能弧；端点对本身不重复（重复端点对属非法输入）。
    possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(possible)
    candidates = []
    counter = itertools.count()
    for a, b in possible:
        if rng.random() < 0.4:
            candidates.append(
                cand(
                    f"cid{next(counter):03d}",
                    ids[a],
                    ids[b],
                    rng.choice([0, 0, 1, 2, 5]),
                )
            )

    res = audit({"hits": make_hits(n), "candidates": candidates})
    ref = brute_solve(n, to_arc_records(candidates, ids))

    assert int(res["optimal_count"]) == ref["optimal_count"]
    assert res["paired_hits"] == 2 * ref["max_pairs"]
    assert res["total_residual"] == ref["min_cost"]
    assert [p["id"] for p in res["canonical_pairs"]] == ref["canonical"]

    hit_ids = [f"h{k}" for k in range(n)]
    assert res["unmatched_hits"] == [hit_ids[k] for k in ref["canonical_unmatched"]]

    cls = res["classification"]
    assert cls == ref["classification"]

    for cid, _a, _b, _r in to_arc_records(candidates, ids):
        used = ref["usage"][cid]
        if used == 0:
            assert cid in cls["never"]
        elif used == ref["optimal_count"]:
            assert cid in cls["required"]
        else:
            assert cid in cls["optional"]


# ---------------------------------------------------------------- 性能


def test_max_scale_performance():
    n = MAX_HITS
    hits = make_hits(n)
    rng = random.Random(42)
    possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(possible)
    candidates = [
        cand(f"c{k:04d}", f"h{a}", f"h{b}", rng.randrange(1000))
        for k, (a, b) in enumerate(possible[:MAX_CANDIDATES])
    ]
    res = audit({"hits": hits, "candidates": candidates})
    assert int(res["optimal_count"]) >= 1
    assert len(res["canonical_pairs"]) * 2 == res["paired_hits"]
