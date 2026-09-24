"""稀疏候选族构造与几何检查工具：回归"大批稀疏候选丢失全局非交叉约束"。

候选族（72 个击中、50 条候选）：
* 四条关键弧：a-cross h0–h30 残差 0、z-outer h0–h70 残差 10、
  b-short h10–h20 残差 10、y-cross h10–h49 残差 0；
* 46 条残差 100 的填充弧：h0–h50..h69（20 条）、
  h10–h21..h29 与 h10–h31..h47（26 条）。

唯一正确结果：配对 4 个击中、最小总残差 10、恰 2 个最优方案
（a-cross+b-short 或 z-outer+y-cross），规范解为前者；
a-cross 与 y-cross 交叉（0 < 10 < 30 < 49），不得同时入选。
"""

from __future__ import annotations

import itertools

KEY_IDS = ("a-cross", "z-outer", "b-short", "y-cross")


def sparse_bait_family() -> dict:
    hits = [{"id": f"h{k}", "position": k} for k in range(72)]
    candidates = [
        {"id": "a-cross", "left_endpoint": "h0", "right_endpoint": "h30", "residual": 0},
        {"id": "z-outer", "left_endpoint": "h0", "right_endpoint": "h70", "residual": 10},
        {"id": "b-short", "left_endpoint": "h10", "right_endpoint": "h20", "residual": 10},
        {"id": "y-cross", "left_endpoint": "h10", "right_endpoint": "h49", "residual": 0},
    ]
    candidates += [
        {
            "id": f"f0-{b}",
            "left_endpoint": "h0",
            "right_endpoint": f"h{b}",
            "residual": 100,
        }
        for b in range(50, 70)
    ]
    candidates += [
        {
            "id": f"f10-{b}",
            "left_endpoint": "h10",
            "right_endpoint": f"h{b}",
            "residual": 100,
        }
        for b in list(range(21, 30)) + list(range(31, 48))
    ]
    return {"hits": hits, "candidates": candidates}


def crossing_pairs(canonical_pairs: list[dict], hits: list[dict]) -> list[tuple[str, str]]:
    """逐对检查规范弧，返回所有交叉弧对（合法径迹解释下应为空）。"""
    pos = {h["id"]: h["position"] for h in hits}
    spans = [
        (p["id"], pos[p["left_endpoint"]], pos[p["right_endpoint"]])
        for p in canonical_pairs
    ]
    bad = []
    for (id1, a, b), (id2, c, d) in itertools.combinations(spans, 2):
        if a < c < b < d or c < a < d < b:
            bad.append((id1, id2))
    return bad
