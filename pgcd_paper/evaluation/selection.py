from __future__ import annotations
import numpy as np


def select_condition_subset(condition_ids: np.ndarray, material_codes: np.ndarray, count: int | None,
                            policy: str = "full", seed: int = 0) -> np.ndarray:
    ids, mats = np.asarray(condition_ids, int), np.asarray(material_codes, int)
    if ids.ndim != 1 or mats.shape != ids.shape: raise ValueError("ids and material codes must be aligned vectors")
    if count is None or count == 0 or count >= len(ids): return ids.copy()
    if count < 1: raise ValueError("count must be positive, zero, or None")
    if policy == "legacy-prefix": return ids[:count].copy()
    if policy != "stratified": raise ValueError(f"unknown subset policy: {policy}")
    rng = np.random.RandomState(seed); groups = {m: ids[mats == m].copy() for m in sorted(set(mats.tolist()))}
    for values in groups.values(): rng.shuffle(values)
    chosen = []
    while len(chosen) < count:
        progressed = False
        for material in groups:
            values = groups[material]
            index = sum(1 for cid in chosen if cid in set(values.tolist()))
            if index < len(values) and len(chosen) < count:
                chosen.append(int(values[index])); progressed = True
        if not progressed: break
    return np.array(chosen, int)
