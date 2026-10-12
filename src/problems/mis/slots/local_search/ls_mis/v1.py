import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    n = env_data["num_vertices"]
    adj = env_data["adj"]
    seq = np.array(state.sequence, dtype=np.int32)
    best_seq = seq.copy()
    best_mk = state.makespan
    degrees = [len(neighbors) for neighbors in adj]

    for _ in range(30):
        r = random.random()
        if r < 0.6:
            ones = np.flatnonzero(seq == 1).tolist()
            if not ones:
                continue
            v_remove = random.choice(ones)
            seq[v_remove] = 0
            blocked = np.zeros(n, dtype=np.int32)
            for i in np.flatnonzero(seq == 1):
                blocked[list(adj[i])] = 1
            candidates = np.flatnonzero((seq == 0) & (blocked == 0)).tolist()
            candidates.sort(key=lambda i: degrees[i])
            for c in candidates:
                if blocked[c] == 0:
                    seq[c] = 1
                    blocked[list(adj[c])] = 1
        else:
            blocked = np.zeros(n, dtype=np.int32)
            for i in np.flatnonzero(seq == 1):
                blocked[list(adj[i])] = 1
            candidates = np.flatnonzero((seq == 0) & (blocked == 0)).tolist()
            if not candidates:
                continue
            candidates.sort(key=lambda i: degrees[i])
            for c in candidates[:5]:
                seq[c] = 1
                mk = calc_makespan_fn(seq, env_data)
                if mk < best_mk:
                    best_mk = mk
                    best_seq = seq.copy()
                    break
                seq[c] = 0

        mk = calc_makespan_fn(seq, env_data)
        if mk < best_mk:
            best_mk, best_seq = mk, seq.copy()
        else:
            seq = best_seq.copy()

    state.sequence, state.makespan = best_seq, best_mk
    return state
