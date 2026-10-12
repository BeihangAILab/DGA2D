"""
MIS domain evaluator — Maximum Independent Set.

Exposes the DGA2D domain protocol:
  load_instance_group, get_reward_spec, describe_instance, calc_makespan,
  get_initial_pipeline, initialize_state, get_prompt

WARNING — MIS is a MAXIMISATION problem:
  The engine minimises, so calc_makespan returns ** -|IS| **.
  GAP is computed via: (-score - (-UB)) / (-UB) × 100
  which simplifies to (UB - |IS|) / UB × 100 (gap to optimal).

Encoding:
  state.sequence: numpy int32 array of length N.
    1  = vertex IS in the independent set
    0  = vertex NOT in the set
  Feasibility: no two vertices in IS may be adjacent.

Instances: DIMACS .clq maximum-clique files (edge list format).  Each input
  graph is converted to its complement so that a maximum independent set in
  the evaluator corresponds exactly to a maximum clique and its published BKS.
  Groups: BHOSLIB, DIMACS_all, DIMACS_subset.
  BKS (best known solution) in mis_bks.json.
"""

import os
import json
import random
import numpy as np

from src.core.configuration import problem_data_dir

# ── Module-level config (set by Hydra before loading) ──
INVALID_SCORE = 999999999.0
SEQUENCE_ONLY_OBJECTIVE = True
MIS_LE2000_GROUP = "BHOSLIB_DIMACS_le2000"
MIS_LE2000_SOURCE_GROUPS = ("BHOSLIB", "DIMACS_all", "DIMACS_subset")
MIS_LE2000_EXCLUDED = {"c4000.5", "mann_a81", "keller6"}


def _get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _get_instances_dir():
    private_fallback = os.path.join(_get_project_root(), "src", "problems", "mis", "instances")
    return str(problem_data_dir("mis", private_fallback))


# ── DIMACS .clq parser ───────────────────────────────────────────


def _parse_clq(filepath):
    """
    Parse a DIMACS-format maximum-clique file into its complement graph.

    Format:
      c ...                  comment
      p edge <N> <M>         header: N vertices, M edges
      e <u> <v>              edge between vertices u and v (1-indexed)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    n_vertices = 0
    edges = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("c"):
            continue
        parts = line.split()
        if parts[0] == "p":
            n_vertices = int(parts[2])
        elif parts[0] == "e":
            u, v = int(parts[1]), int(parts[2])
            edges.append((u - 1, v - 1))  # 0-indexed

    if n_vertices == 0:
        raise ValueError(f"Invalid .clq file: no 'p edge' line in {filepath}")

    # The bundled BKS values are maximum-clique values for the DIMACS input
    # graph.  MIS(G_complement) is exactly Clique(G), so the conflict graph
    # exposed to operators must be the complement of the file's edge set.
    input_edges = set()
    for u, v in edges:
        if 0 <= u < n_vertices and 0 <= v < n_vertices and u != v:
            input_edges.add((min(u, v), max(u, v)))

    adj = [set() for _ in range(n_vertices)]
    for u in range(n_vertices):
        for v in range(u + 1, n_vertices):
            if (u, v) not in input_edges:
                adj[u].add(v)
                adj[v].add(u)

    return n_vertices, adj


# ── BKS loading ───────────────────────────────────────────────────


def _load_bks():
    path = os.path.join(_get_instances_dir(), "mis_bks.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Instance listing ──────────────────────────────────────────────


def _list_instances_in_group(group):
    if group == MIS_LE2000_GROUP:
        by_name = {}
        for source_group in MIS_LE2000_SOURCE_GROUPS:
            for path in _list_instances_in_group(source_group):
                stem = os.path.splitext(os.path.basename(path))[0].casefold()
                if stem not in MIS_LE2000_EXCLUDED:
                    by_name.setdefault(stem, path)
        return [by_name[name] for name in sorted(by_name)]
    group_dir = os.path.join(_get_instances_dir(), group)
    if not os.path.isdir(group_dir):
        return []
    files = sorted([f for f in os.listdir(group_dir) if f.endswith(".clq")])
    return [os.path.join(group_dir, f) for f in files]


def _inst_to_env(path, bks):
    name = os.path.splitext(os.path.basename(path))[0]
    n, adj = _parse_clq(path)
    bks_val = float(bks.get(name, n))  # clique BKS = |IS| in the complement
    # score = -|IS| (maximisation → minimisation)
    # upper_bound = -bks_val  so GAP = (-|IS| - (-bks)) / (-bks) × 100
    #                           = (bks - |IS|) / bks × 100
    # 0 % = optimal, negative = suboptimal
    neg_bks = -bks_val
    return {
        "num_vertices": n,
        "adj": adj,
        "name": name,
        "lower_bound": neg_bks,
        "upper_bound": neg_bks,
    }


def load_instance_group(group):
    """Load every instance in the requested dataset group."""
    paths = _list_instances_in_group(group)
    bks = _load_bks()
    return [_inst_to_env(p, bks) for p in paths]


# ── IS evaluation ─────────────────────────────────────────────────


def _count_and_check(seq, adj):
    """
    Count the independent set size and verify feasibility.

    Returns:
      (size, feasible) — size is meaningful only if feasible.
      If infeasible (adjacent 1s), returns (0, False).
    """
    ones = np.flatnonzero(seq).tolist()
    remaining = set(ones)

    # Feasibility: for every pair of vertices in IS, no edge can exist
    for v in ones:
        remaining.remove(v)
        if not adj[v].isdisjoint(remaining):
            return 0, False

    return len(ones), True


# ── Public evaluation interface ───────────────────────────────────


def calc_makespan(sequence_or_state, env_data):
    """
    Compute MIS score.

    Since the engine MINIMISES, this returns ** -|IS| **.
    A larger independent set produces a more negative (i.e. better) score.

    Feasibility check: any two adjacent vertices both set to 1 → 0.0 (worst).
    """
    if hasattr(sequence_or_state, "sequence"):
        seq = sequence_or_state.sequence
    else:
        seq = sequence_or_state

    if seq is None:
        return INVALID_SCORE

    n = env_data["num_vertices"]
    adj = env_data["adj"]

    try:
        raw = np.asarray(seq).ravel()
        if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
            return INVALID_SCORE
        seq_arr = raw.astype(np.int32)
    except (TypeError, ValueError, OverflowError):
        return INVALID_SCORE
    if len(seq_arr) != n or np.any((seq_arr != 0) & (seq_arr != 1)):
        return INVALID_SCORE

    size, feasible = _count_and_check(seq_arr, adj)
    if not feasible:
        return INVALID_SCORE

    return -float(size)


# ── State initialisation (greedy construction) ────────────────────


def initialize_state(env_data, state):
    """
    Greedy independent set construction.

    Sorts vertices by degree (ascending), then greedily adds vertices
    that do not conflict.  With random tie-breaking for diversity.
    """
    if (
        state.sequence is not None
        and len(state.sequence) > 0
        and state.makespan < 0.0  # valid non-trivial solution
    ):
        return

    n = env_data["num_vertices"]
    adj = env_data["adj"]

    # Compute degrees
    degrees = np.array([len(adj[i]) for i in range(n)], dtype=np.int32)
    # Sort ascending by degree, resolve ties randomly
    order = list(range(n))
    random.shuffle(order)
    order.sort(key=lambda i: degrees[i])

    seq = np.zeros(n, dtype=np.int32)
    occupied_neighbors = np.zeros(n, dtype=np.int32)  # >0 means blocked

    for v in order:
        if occupied_neighbors[v] == 0:
            seq[v] = 1
            for u in adj[v]:
                occupied_neighbors[u] = 1

    size, _ = _count_and_check(seq, adj)
    state.sequence = seq
    state.makespan = -float(size)
    state.metadata = {}


# ── Pipeline configuration ────────────────────────────────────────


def get_initial_pipeline():
    return [
        "initialization|init_greedy",
        "perturbation|mut_flip",
        "local_search|ls_mis",
        "acceptance|simulated_annealing",
    ]


# ── Domain prompt ─────────────────────────────────────────────────


def get_allowed_nodes(node_ablation_level):
    """Return the exact nested MIS operator pool for a V-Level."""
    level_4 = {
        "acceptance|simulated_annealing",
        "initialization|init_greedy",
        "local_search|ls_mis",
        "perturbation|mut_flip",
    }
    level_8 = level_4 | {
        "initialization|init_random",
        "intensification|tabu_search_lite",
        "local_search|ls_one_two_swap",
        "perturbation|mut_remove_rebuild",
    }
    level_15 = level_8 | {
        "diversification|div_random_walk",
        "initialization|init_dynamic_degree",
        "intensification|vns_descent",
        "local_search|ls_add_drop",
        "local_search|ls_plateau_swap",
        "perturbation|mut_conflict_repair",
        "perturbation|mut_neighborhood_destroy",
    }
    level_20 = level_15 | {
        "diversification|div_partial_restart",
        "history_mining|history_elite_crossover",
        "initialization|init_multistart_greedy",
        "intensification|iterated_greedy",
        "ruin_and_recreate|ruin_random",
    }
    pools = {4: level_4, 8: level_8, 15: level_15, 20: level_20}
    try:
        level = int(node_ablation_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("MIS node_ablation_level must be one of 4, 8, 15, or 20") from exc
    if level not in pools:
        raise ValueError(
            f"Unsupported MIS node_ablation_level={node_ablation_level!r}; "
            "expected one of 4, 8, 15, or 20"
        )
    return set(pools[level])


def get_prompt(root_dir=None):
    if root_dir is None:
        root_dir = _get_project_root()
    prompt_path = os.path.join(root_dir, "prompts", "mis", "domain_knowledge.txt")
    if not os.path.exists(prompt_path):
        return ""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_reward_spec(env_data):
    from src.core.contracts import default_reward_spec

    return default_reward_spec(env_data)


def describe_instance(env_data):
    from src.core.contracts import default_describe_instance

    return default_describe_instance(env_data)
