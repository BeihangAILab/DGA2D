import random
import numpy as np


def _rebuild(selected, removed, n, adj):
    chosen = set(selected) - set(removed)
    order = [v for v in range(n) if v not in chosen]
    random.shuffle(order)
    order.sort(key=lambda v: len(adj[v]))
    for vertex in order:
        if adj[vertex].isdisjoint(chosen):
            chosen.add(vertex)
    return chosen


def run(env_data, state, calc_makespan_fn):
    """Try bounded destroy sizes 1, 2, and 3; retain only the best improvement."""
    n = env_data["num_vertices"]
    adj = env_data["adj"]
    original = np.asarray(state.sequence, dtype=np.int32)
    selected = set(np.flatnonzero(original).tolist())
    best_seq = original.copy()
    best_score = state.makespan

    for size in (1, 2, 3):
        if len(selected) < size:
            continue
        for _ in range(4):
            removed = random.sample(list(selected), size)
            chosen = _rebuild(selected, removed, n, adj)
            if len(chosen) <= len(selected):
                continue
            candidate = np.zeros(n, dtype=np.int32)
            candidate[list(chosen)] = 1
            score = calc_makespan_fn(candidate, env_data)
            if score < best_score:
                best_seq, best_score = candidate, score
                break
        if best_score < state.makespan:
            break

    state.sequence = best_seq
    state.makespan = best_score
    return state
