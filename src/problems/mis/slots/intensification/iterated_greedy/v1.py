import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Repeatedly destroy a small portion and greedily reconstruct the set."""
    n = env_data["num_vertices"]
    adj = env_data["adj"]
    original = np.asarray(state.sequence, dtype=np.int32)
    selected = set(np.flatnonzero(original).tolist())
    best_seq = original.copy()
    best_score = state.makespan

    if not selected:
        return state
    for _ in range(8):
        count = min(len(selected), max(1, int(round(len(selected) * 0.15))))
        removed = set(random.sample(list(selected), count))
        chosen = selected - removed
        order = [v for v in range(n) if v not in chosen]
        random.shuffle(order)
        order.sort(key=lambda v: len(adj[v]) + random.random() * 2.0)
        for vertex in order:
            if adj[vertex].isdisjoint(chosen):
                chosen.add(vertex)
        if len(chosen) <= len(selected):
            continue
        candidate = np.zeros(n, dtype=np.int32)
        candidate[list(chosen)] = 1
        score = calc_makespan_fn(candidate, env_data)
        if score < best_score:
            best_seq, best_score = candidate, score
            break

    state.sequence = best_seq
    state.makespan = best_score
    return state
