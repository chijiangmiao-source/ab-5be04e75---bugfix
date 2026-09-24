"""共享验收场景与断言。

场景 family：72 个击中、50 条候选的大批稀疏候选族，其中四条关键弧
a-cross/z-outer/b-short/y-cross 构成两个相互交叉的廉价组合，旧版按
连通分量分解的实现会丢失这一全局几何约束。
"""

from __future__ import annotations

import itertools
from typing import Any

N_HITS = 72

# (id, 左端点位置, 右端点位置, 残差)
KEY_CANDIDATES = [
    ("a-cross", 0, 30, 0),
    ("z-outer", 0, 70, 10),
    ("b-short", 10, 20, 10),
    ("y-cross", 10, 49, 0),
]
KEY_IDS = [cid for cid, _a, _b, _r in KEY_CANDIDATES]

# 规范解选中的端点；其余 68 个击中全部未配对。
CANONICAL_MATCHED = {0, 10, 20, 30}
CANONICAL_IDS = ["a-cross", "b-short"]
OPTIMAL_COUNT = "2"
OPTIMAL_PAIRED_HITS = 4
OPTIMAL_TOTAL_RESIDUAL = 10


def hit_id(index: int) -> str:
    return f"h{index}"


def make_family_hits() -> list[dict[str, Any]]:
    return [{"id": hit_id(k), "position": k} for k in range(N_HITS)]


def make_family_candidates() -> list[dict[str, Any]]:
    """生成 50 条候选：4 条关键弧 + 46 条残差 100 的合法诱饵弧。"""
    candidates = [
        {
            "id": cid,
            "left_endpoint": hit_id(left),
            "right_endpoint": hit_id(right),
            "residual": residual,
        }
        for cid, left, right, residual in KEY_CANDIDATES
    ]
    # 20 条：h0 -> h50..h69。
    for right in range(50, 70):
        candidates.append(
            {
                "id": f"decoy0-{right}",
                "left_endpoint": hit_id(0),
                "right_endpoint": hit_id(right),
                "residual": 100,
            }
        )
    # 26 条：h10 -> h21..h29 与 h31..h47。
    for right in list(range(21, 30)) + list(range(31, 48)):
        candidates.append(
            {
                "id": f"decoy10-{right}",
                "left_endpoint": hit_id(10),
                "right_endpoint": hit_id(right),
                "residual": 100,
            }
        )
    return candidates


def make_family_payload() -> dict[str, Any]:
    return {"hits": make_family_hits(), "candidates": make_family_candidates()}


def decoy_ids() -> set[str]:
    return {f"decoy0-{r}" for r in range(50, 70)} | {
        f"decoy10-{r}" for r in list(range(21, 30)) + list(range(31, 48))
    }


def expected_unmatched() -> list[str]:
    return [hit_id(k) for k in range(N_HITS) if k not in CANONICAL_MATCHED]


def endpoint_index(endpoint: str) -> int:
    return int(endpoint[1:])


def assert_canonical_pairs_noncrossing(pairs: list[dict[str, Any]]) -> None:
    """逐对验证响应中的规范弧绘制后互不交叉（端点互异、允许嵌套与并列）。"""
    arcs = [
        (pair["id"], endpoint_index(pair["left_endpoint"]),
         endpoint_index(pair["right_endpoint"]))
        for pair in pairs
    ]
    for (id1, l1, r1), (id2, l2, r2) in itertools.combinations(arcs, 2):
        endpoints = {l1, r1, l2, r2}
        assert len(endpoints) == 4, f"规范弧共享端点: {id1}, {id2}"
        crossing = l1 < l2 < r1 < r2 or l2 < l1 < r2 < r1
        assert not crossing, f"规范弧发生交叉: {id1} 与 {id2}"


def assert_family_result(result: dict[str, Any]) -> None:
    """核对候选族的精确目标、方案数、规范配对、未配对集合与三类归属。"""
    assert result["paired_hits"] == OPTIMAL_PAIRED_HITS
    assert result["total_residual"] == OPTIMAL_TOTAL_RESIDUAL
    assert result["optimal_count"] == OPTIMAL_COUNT
    assert [pair["id"] for pair in result["canonical_pairs"]] == CANONICAL_IDS
    assert result["unmatched_hits"] == expected_unmatched()

    classification = result["classification"]
    assert classification["required"] == []
    assert set(classification["optional"]) == set(KEY_IDS)
    assert set(classification["never"]) == decoy_ids()
    # 三类互不相交且合起来恰好是全部 50 条候选。
    all_classified = (
        set(classification["required"])
        | set(classification["optional"])
        | set(classification["never"])
    )
    assert all_classified == set(KEY_IDS) | decoy_ids()
    assert len(all_classified) == 50
    assert_canonical_pairs_noncrossing(result["canonical_pairs"])


def result_signature(result: dict[str, Any]) -> tuple[Any, ...]:
    """与候选录入顺序无关的结果指纹，用于打乱顺序后逐字段比对。"""
    classification = result["classification"]
    return (
        result["optimal_count"],
        result["paired_hits"],
        result["total_residual"],
        [(pair["id"], pair["left_endpoint"], pair["right_endpoint"],
          pair["residual"]) for pair in result["canonical_pairs"]],
        list(result["unmatched_hits"]),
        {label: sorted(ids) for label, ids in classification.items()},
    )
