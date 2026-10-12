from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .checkpoint import CheckpointManager
from .configuration import cfg_get, configure_huggingface_cache
from .contracts import (
    CandidateResult,
    EVALUATION_SCHEMA_VERSION,
    PipelineSample,
    PolicyUpdateStats,
    describe_from_domain,
)
from .credit import TransitionCreditStore, instantiate_pipeline
from .dataset import select_evolution_batch, split_domain_instances
from .execution import RewardBatchEvaluator
from .evolution import GraphEvolver, RuleBasedOperatorEvolver
from .operators import RunOperatorPool, discover_operator_versions
from .policy import PipelinePolicy
from .reporting import TrainingReporter
from .structure import (
    DirectedOperatorGraph,
    END,
    START,
    InitializationResult,
    PaperInitializer,
    configured_structure_mode,
    resolve_fixed_evolution_node,
)


def learning_diagnostics(graph, max_length, results, stats, skipped=False):
    rewards = [result.raw_reward for result in results]
    unique = len({result.sample.signature for result in results})
    routes = graph.bounded_route_count(max_length)
    finite_rewards = bool(rewards) and all(math.isfinite(value) for value in rewards)
    variance = statistics.pvariance(rewards) if finite_rewards else None
    finite = finite_rewards and math.isfinite(stats.loss) and math.isfinite(stats.grad_norm)
    failures = sum(result.failed for result in results)
    effective = not skipped and finite and variance > 0 and stats.grad_norm > 0
    if skipped:
        reason = "policy_update_disabled"
    elif not finite:
        reason = "nonfinite_values"
    elif failures:
        reason = "candidate_execution_failure"
    elif routes == 1:
        reason = "single_bounded_route"
    elif unique == 1:
        reason = "repeated_samples"
    elif variance == 0:
        reason = "distinct_pipelines_equal_rewards"
    elif stats.grad_norm == 0:
        reason = "zero_gradient"
    else:
        reason = "effective_learning"
    return {
        "bounded_route_count": routes,
        "unique_pipeline_count": unique,
        "reward_variance": variance,
        "finite": finite,
        "failed_candidates": failures,
        "effective_learning": effective,
        "reason": reason,
    }


def _source_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("v*.py"))
    }


def _json_safe_config(config) -> Any:
    try:
        from omegaconf import OmegaConf

        value = OmegaConf.to_container(config, resolve=True)
    except Exception:
        if isinstance(config, dict):
            value = json.loads(json.dumps(config))
        else:
            return repr(config)
    sensitive_fragments = (
        "api_key",
        "authorization",
        "password",
        "secret",
        "token",
    )

    def redact(item):
        if isinstance(item, dict):
            return {
                key: (
                    "<redacted>"
                    if any(fragment in str(key).lower() for fragment in sensitive_fragments)
                    else redact(child)
                )
                for key, child in item.items()
            }
        if isinstance(item, list):
            return [redact(child) for child in item]
        return item

    return redact(value)


def _task_context(prompt: str, domain_evaluator, instances: list[dict]) -> str:
    summaries = [describe_from_domain(domain_evaluator, instance) for instance in instances[:3]]
    return f"{prompt}\nRepresentative benchmark summaries:\n" + "\n".join(summaries)


def _resume_config_signature(config_value: Any) -> Any:
    value = json.loads(json.dumps(config_value))
    if isinstance(value, dict):
        engine = value.get("engine")
        if isinstance(engine, dict):
            engine.pop("generations", None)
        checkpoint = value.get("checkpoint")
        if isinstance(checkpoint, dict):
            for key in ("resume", "save_last", "save_best"):
                checkpoint.pop(key, None)
    return value


def _record_faults(candidate: CandidateResult, credit: TransitionCreditStore) -> None:
    for implementation, details in candidate.operator_faults.items():
        messages = details.get("messages", []) if isinstance(details, dict) else []
        credit.record_fault(implementation, "\n".join(map(str, messages)))
    if candidate.crashed_component:
        for category, name, version in candidate.sample.implementations:
            if f"{category}|{name}" == candidate.crashed_component:
                credit.record_fault(
                    credit.implementation_id(f"{category}|{name}", version),
                    candidate.failure_reason,
                )


class PaperRLTrainer:
    def __init__(
        self,
        domain_evaluator,
        prompt_context: str,
        config,
        source_slots: str,
        run_dir: str,
        project_root: str,
        allowed_nodes: set[str] | None = None,
        policy_factory: Callable[..., Any] = PipelinePolicy,
        evaluator_factory: Callable[..., Any] = RewardBatchEvaluator,
        llm_client=None,
        reporter: TrainingReporter | None = None,
    ):
        self.domain = domain_evaluator
        self.prompt = prompt_context
        self.config = config
        self.source_slots = Path(source_slots).resolve()
        problem_dir = str(self.source_slots.parent)
        if problem_dir not in sys.path:
            sys.path.insert(0, problem_dir)
        self.run_dir = Path(run_dir).resolve()
        self.project_root = Path(project_root).resolve()
        self.allowed_nodes = allowed_nodes
        self.policy_factory = policy_factory
        self.evaluator_factory = evaluator_factory
        if llm_client is None:
            raise ValueError("PaperRLTrainer requires an explicit run-scoped LLM client")
        self.llm = llm_client
        self.reporter = reporter or TrainingReporter()
        self.seed = int(cfg_get(config, "engine.seed", 0))
        self.rng = random.Random(self.seed)
        self.structure_mode = configured_structure_mode(config)

    def _make_evaluator(self, pool: RunOperatorPool, steps: int | None = None):
        evaluator = self.evaluator_factory(
            self.domain,
            str(pool.root),
            int(steps or cfg_get(self.config, "engine.pipeline_steps", 500)),
            float(cfg_get(self.config, "engine.pipeline_timeout", 90.0)),
            float(cfg_get(self.config, "engine.failure_reward", -5.0)),
            float(cfg_get(self.config, "engine.reward_epsilon", 1.0e-8)),
        )
        evaluator.max_workers = max(1, int(cfg_get(self.config, "engine.evaluator_workers", 4)))
        return evaluator

    def _preflight(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        configure_huggingface_cache(
            self.project_root,
            cfg_get(self.config, "policy.cache_dir", ".model_cache/huggingface"),
        )
        import torch

        random.seed(self.seed)
        try:
            import numpy as np

            np.random.seed(self.seed)
        except Exception:
            pass
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        tier = str(cfg_get(self.config, "engine.run_tier", "quick"))
        if not torch.cuda.is_available():
            if tier != "quick":
                raise RuntimeError(
                    "Paper-RL non-quick training requires CUDA; current PyTorch is CPU-only"
                )
            self.reporter.warning(
                "QUICK TIER IS RUNNING ON CPU. Qwen2.5 policy training will be "
                f"slow; policy.cpu_threads={int(cfg_get(self.config, 'policy.cpu_threads', 4))} "
                f"and policy.cpu_interop_threads={int(cfg_get(self.config, 'policy.cpu_interop_threads', 1))}."
            )

    def _instantiate(self, sample, pool, credit, temperature, greedy=False) -> PipelineSample:
        return instantiate_pipeline(sample, pool, credit, temperature, self.rng, greedy=greedy)

    def _runtime_validation_route(
        self,
        route: list[str],
        pool: RunOperatorPool,
        credit: TransitionCreditStore,
        instance: dict,
        temperature: float,
    ) -> bool:
        try:
            sample = self._instantiate(
                PipelineSample(nodes=route), pool, credit, temperature, greedy=True
            )
            result = self._make_evaluator(
                pool, steps=min(5, int(cfg_get(self.config, "engine.pipeline_steps", 500)))
            ).evaluate(
                [sample],
                [instance],
                standardize=False,
                evaluation_seed=self.seed,
            )[0]
            if result.failed:
                self.reporter.warning(
                    f"Graph route runtime_validation rejected {route}: {result.failure_reason}"
                )
            return not result.failed
        except Exception as exc:
            self.reporter.warning(f"Graph route runtime_validation rejected: {type(exc).__name__}: {exc}")
            return False

    def run(self) -> dict[str, Any]:
        run_started = time.perf_counter()
        initialization_started = time.perf_counter()
        self._preflight()
        group = str(cfg_get(self.config, "problem.target_group", ""))
        splits = split_domain_instances(
            self.domain,
            group,
            float(cfg_get(self.config, "dataset.train_ratio", 0.60)),
            float(cfg_get(self.config, "dataset.validation_ratio", 0.20)),
            float(cfg_get(self.config, "dataset.test_ratio", 0.20)),
            int(cfg_get(self.config, "dataset.split_seed", 0)),
        )
        generations = int(cfg_get(self.config, "engine.generations", 50))
        candidates_per_generation = int(
            cfg_get(self.config, "engine.candidates_per_generation", 4)
        )
        shared_instances = int(
            cfg_get(self.config, "engine.shared_instances_per_generation", 3)
        )
        pool_dir = self.run_dir / "operator_pool"
        checkpoints = CheckpointManager(self.run_dir)
        resume = bool(cfg_get(self.config, "checkpoint.resume", False))
        resume = resume and checkpoints.exists("last")
        start_generation = 1
        best_validation = float("-inf")
        best_metadata: dict[str, Any] = {}

        if resume:
            checkpoint_dir = checkpoints.path("last")
            manifest = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
            if int(manifest.get("schema_version", -1)) != 3:
                raise RuntimeError(
                    "Checkpoint predates fixed run-scoped evolution batches and cannot be resumed"
                )
            if manifest.get("evaluation_schema_version") != EVALUATION_SCHEMA_VERSION:
                raise RuntimeError(
                    "Checkpoint predates indexed benchmark rewards and cannot be resumed"
                )
            if manifest.get("dataset") != splits.manifest:
                raise RuntimeError(
                    "Current dataset split/content does not match the last checkpoint"
                )
            if manifest.get("structure", {}).get("name") != self.structure_mode:
                raise RuntimeError(
                    "Checkpoint structure mode does not match the current configuration"
                )
            if _resume_config_signature(manifest.get("config")) != _resume_config_signature(
                _json_safe_config(self.config)
            ):
                raise RuntimeError(
                    "Training configuration changed since the last checkpoint; "
                    "only generations and checkpoint save/resume flags may change"
                )
            current_hashes = _source_hashes(self.source_slots)
            if manifest.get("baseline_source_hashes") != current_hashes:
                raise RuntimeError("Baseline slots changed since the last checkpoint")
            graph = DirectedOperatorGraph.from_dict(
                json.loads((checkpoint_dir / "graph.json").read_text(encoding="utf-8"))
            )
            init_manifest = manifest["initialization"]
            initialized = InitializationResult(
                graph=graph,
                max_operator_count=int(init_manifest["max_operator_count"]),
                max_pipeline_length=int(init_manifest["max_pipeline_length"]),
                sampling_temperature=float(init_manifest["sampling_temperature"]),
                degraded_initialization=bool(init_manifest.get("degraded_initialization", False)),
                llm_response=init_manifest.get("llm_response"),
            )
            pool = RunOperatorPool(pool_dir)
            credit = TransitionCreditStore(
                float(cfg_get(self.config, "credit.alpha", 0.1)),
                str(cfg_get(self.config, "credit.order", "first")),
                float(cfg_get(self.config, "credit.aggregation_epsilon", 1.0e-8)),
            )
            benchmark_batch, benchmark_manifest = select_evolution_batch(
                splits, shared_instances, self.rng, manifest.get("evolution_batch")
            )
        else:
            checkpoints.clear()
            events_path = self.run_dir / "events.jsonl"
            events_path.unlink(missing_ok=True)
            benchmark_batch, benchmark_manifest = select_evolution_batch(
                splits, shared_instances, self.rng
            )
            initializer = PaperInitializer(
                str(self.source_slots),
                self.prompt,
                self.domain.get_initial_pipeline(),
                self.config,
                self.llm,
                self.allowed_nodes,
            )
            initialized = initializer.initialize()
            graph = initialized.graph
            graph.validate_mode(self.structure_mode)
            if pool_dir.exists():
                shutil.rmtree(pool_dir)
            pool = RunOperatorPool.create_from_source(self.source_slots, pool_dir, graph.nodes)
            credit = TransitionCreditStore(
                float(cfg_get(self.config, "credit.alpha", 0.1)),
                str(cfg_get(self.config, "credit.order", "first")),
                float(cfg_get(self.config, "credit.aggregation_epsilon", 1.0e-8)),
            )
            baseline_route = [
                node for node in graph.shortest_path(START, END) or [] if node not in (START, END)
            ]
            initial_routes = {
                tuple(route)
                for edge in graph.edges
                for route in [graph.route_through_edge(edge)]
                if route is not None and len(route) <= initialized.max_pipeline_length
            }
            initial_graph_valid = bool(initial_routes) and all(
                self._runtime_validation_route(
                    list(route),
                    pool,
                    credit,
                    benchmark_batch[0],
                    initialized.sampling_temperature,
                )
                for route in initial_routes
            )
            if not initial_graph_valid:
                self.reporter.warning("Initial graph failed runtime_validation execution; using baseline fallback")
                initialized = initializer._fallback(initialized.llm_response)
                graph = initialized.graph
                if pool_dir.exists():
                    shutil.rmtree(pool_dir)
                pool = RunOperatorPool.create_from_source(self.source_slots, pool_dir, graph.nodes)
                baseline_route = [
                    node
                    for node in graph.shortest_path(START, END) or []
                    if node not in (START, END)
                ]
                if not self._runtime_validation_route(
                    baseline_route,
                    pool,
                    credit,
                    benchmark_batch[0],
                    initialized.sampling_temperature,
                ):
                    raise RuntimeError("Even the safe baseline graph failed runtime_validation execution")

            fixed_evolve_node = (
                resolve_fixed_evolution_node(graph, self.config)
                if self.structure_mode == "fixed"
                else None
            )
            public_config = getattr(self.llm, "public_config", None)
            proposer_manifest = (
                public_config()
                if callable(public_config)
                else {"model": getattr(self.llm, "model_name", None)}
            )
            active_nodes = sorted(
                self.allowed_nodes
                if self.allowed_nodes is not None
                else discover_operator_versions(self.source_slots)
            )
            ablation_level = cfg_get(self.config, "problem.node_ablation_level", None)
            manifest = {
                "schema": "dga2d-run",
                "schema_version": 3,
                "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                "config": _json_safe_config(self.config),
                "dataset": splits.manifest,
                "evolution_batch": benchmark_manifest,
                "structure": {
                    "name": self.structure_mode,
                    "fixed_evolve_node": fixed_evolve_node,
                },
                "initialization": {
                    "degraded_initialization": initialized.degraded_initialization,
                    "max_operator_count": initialized.max_operator_count,
                    "max_pipeline_length": initialized.max_pipeline_length,
                    "sampling_temperature": initialized.sampling_temperature,
                    "llm_response": initialized.llm_response,
                },
                "node_ablation": (
                    {
                        "level": int(ablation_level),
                        "active_node_count": len(active_nodes),
                        "active_nodes": active_nodes,
                    }
                    if ablation_level is not None
                    else None
                ),
                "baseline_source_hashes": _source_hashes(self.source_slots),
                "proposer": proposer_manifest,
            }

        graph.validate_mode(self.structure_mode)
        fixed_evolve_node = manifest.get("structure", {}).get("fixed_evolve_node")
        if self.structure_mode == "fixed":
            fixed_evolve_node = fixed_evolve_node or resolve_fixed_evolution_node(
                graph, self.config
            )
            eligible_components = {str(fixed_evolve_node)}
        else:
            eligible_components = set(graph.nodes)

        self.reporter.initialization(
            graph, initialized, time.perf_counter() - initialization_started
        )
        model_started = time.perf_counter()
        policy = self.policy_factory(self.config, graph, str(self.project_root))
        if resume:
            restored = checkpoints.load("last", policy, pool, self.rng, load_optimizer=True)
            graph, credit = restored["graph"], restored["credit"]
            checkpoint_meta = restored["checkpoint"]
            self.llm.load_state_dict(checkpoint_meta.get("llm", {}))
            start_generation = int(checkpoint_meta["generation"]) + 1
            best_metadata = checkpoint_meta.get("metadata", {})
            best_validation = float(best_metadata.get("best_validation", float("-inf")))
        else:
            manifest["policy"] = {
                "model_name": getattr(policy, "model_name", None),
                "revision": getattr(policy, "revision", None),
                "resolved_revision": getattr(policy, "resolved_revision", None),
                "action_mapping": policy.alias_by_action,
            }
        self.reporter.model_ready(policy, time.perf_counter() - model_started)
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        baseline_route = [
            node for node in graph.shortest_path(START, END) or [] if node not in (START, END)
        ]
        evaluator = self._make_evaluator(pool)

        validation_interval = int(cfg_get(self.config, "validation.interval", 5))
        validation_context = _task_context(self.prompt, self.domain, splits.validation)
        generation_context = _task_context(self.prompt, self.domain, benchmark_batch)
        events = []
        completed_generation = start_generation - 1

        for generation in range(start_generation, generations + 1):
            self.reporter.generation_start(
                generation,
                generations,
                candidates_per_generation,
                shared_instances,
            )
            sampling_started = time.perf_counter()
            samples = [
                self._instantiate(
                    policy.sample(
                        generation_context,
                        graph,
                        initialized.max_pipeline_length,
                        initialized.sampling_temperature,
                    ),
                    pool,
                    credit,
                    initialized.sampling_temperature,
                )
                for _ in range(candidates_per_generation)
            ]
            sampling_elapsed = time.perf_counter() - sampling_started
            self.reporter.sampling_complete(samples, sampling_elapsed)
            execution_started = time.perf_counter()
            results = evaluator.evaluate(
                samples,
                benchmark_batch,
                standardize=True,
                evaluation_seed=self.seed + generation * 100_000,
            )
            execution_elapsed = time.perf_counter() - execution_started
            self.reporter.candidates(results, execution_elapsed)
            policy_started = time.perf_counter()
            policy_update_skipped = self.structure_mode in {"fixed", "linear"}
            policy_stats = (
                PolicyUpdateStats(loss=0.0, grad_norm=0.0)
                if policy_update_skipped
                else policy.update(results)
            )
            policy_elapsed = time.perf_counter() - policy_started
            self.reporter.policy_update(policy_stats, policy_elapsed)
            diagnostics = learning_diagnostics(
                graph, initialized.max_pipeline_length, results,
                policy_stats, policy_update_skipped,
            )
            self.reporter.info(f"Learning diagnostics: {json.dumps(diagnostics)}")
            for result in results:
                credit.update(
                    result.sample.transition_keys,
                    result.sample.operator_edges,
                    result.normalized_reward,
                )
                if result.failed:
                    _record_faults(result, credit)
            generation_event = {
                "generation": generation,
                "policy_loss": policy_stats.loss,
                "gradient_norm": policy_stats.grad_norm,
                "policy_update_skipped": policy_update_skipped,
                "learning_diagnostics": diagnostics,
                "candidates": [result.to_log_dict() for result in results],
                "operator_evolution": None,
                "graph_evolution": None,
                "timing": {
                    "sampling_seconds": sampling_elapsed,
                    "execution_seconds": execution_elapsed,
                    "policy_update_seconds": policy_elapsed,
                },
            }

            validate_now = generation % validation_interval == 0 or generation == generations
            if validate_now:
                validation_started = time.perf_counter()
                model = getattr(policy, "model", None)
                if model is not None:
                    model.eval()
                beams = policy.beam_search(
                    validation_context,
                    graph,
                    initialized.max_pipeline_length,
                    initialized.sampling_temperature,
                    int(cfg_get(self.config, "validation.beam_size", 4)),
                )
                if not beams:
                    beams = [
                        policy.sample(
                            validation_context,
                            graph,
                            initialized.max_pipeline_length,
                            initialized.sampling_temperature,
                            greedy=True,
                        )
                    ]
                validation_samples = [
                    self._instantiate(
                        sample, pool, credit, initialized.sampling_temperature, greedy=True
                    )
                    for sample in beams
                ]
                validation_results = evaluator.evaluate(
                    validation_samples,
                    splits.validation,
                    standardize=False,
                    evaluation_seed=self.seed + 10_000_000 + generation * 100_000,
                )
                if model is not None:
                    model.train()
                champion = max(
                    (item for item in validation_results
                     if not item.failed and math.isfinite(item.raw_reward)),
                    key=lambda item: item.raw_reward,
                    default=None,
                )
                generation_event["validation"] = [
                    result.to_log_dict() for result in validation_results
                ]
                is_best = champion is not None and champion.raw_reward > best_validation
                if champion is None:
                    self.reporter.warning(
                        "All validation candidates failed; no best checkpoint selected"
                    )
                if is_best:
                    best_validation = champion.raw_reward
                    best_metadata = {
                        "best_validation": best_validation,
                        "champion_nodes": champion.sample.nodes,
                        "champion_implementations": champion.sample.implementations,
                    }
                    if bool(cfg_get(self.config, "checkpoint.save_best", True)):
                        checkpoint_started = time.perf_counter()
                        checkpoints.save(
                            "best",
                            generation,
                            graph,
                            credit,
                            policy,
                            pool,
                            manifest,
                            best_metadata,
                            self.llm.state_dict(),
                            self.rng,
                        )
                        self.reporter.checkpoint(
                            "best",
                            checkpoints.path("best"),
                            time.perf_counter() - checkpoint_started,
                        )
                validation_elapsed = time.perf_counter() - validation_started
                generation_event["timing"]["validation_seconds"] = validation_elapsed
                self.reporter.validation(
                    champion.raw_reward if champion is not None else float("-inf"),
                    is_best, validation_elapsed,
                )

            if generation <= generations:
                evolution_started = time.perf_counter()
                performed = None
                delta = None
                operator_interval = int(cfg_get(self.config, "evolution.operator_interval", 1))
                changes = int(cfg_get(self.config, "evolution.operator_changes_per_event", 1))
                if operator_interval > 0 and generation % operator_interval == 0:
                    evolver = RuleBasedOperatorEvolver(
                        self.config,
                        self.llm,
                        self.prompt,
                        self.domain,
                        benchmark_batch[0],
                        str(self.source_slots.parent),
                    )
                    performed = []

                    def validate_combination(component, version):
                        route = graph.route_through_node(component)
                        if route is None or len(route) > initialized.max_pipeline_length:
                            return False
                        try:
                            sample = self._instantiate(
                                PipelineSample(nodes=route),
                                pool,
                                credit,
                                initialized.sampling_temperature,
                                greedy=True,
                            )
                            for index, (category, name, _old_version) in enumerate(
                                sample.implementations
                            ):
                                if f"{category}|{name}" == component:
                                    sample.implementations[index] = (category, name, version)
                                    break
                            outcome = self._make_evaluator(
                                pool,
                                steps=min(
                                    5, int(cfg_get(self.config, "engine.pipeline_steps", 500))
                                ),
                            ).evaluate(
                                [sample],
                                [benchmark_batch[0]],
                                standardize=False,
                                evaluation_seed=self.seed + generation * 100_000 + 50_000,
                            )[0]
                            return not outcome.failed
                        except Exception:
                            return False

                    for _ in range(changes):
                        change = evolver.evolve(
                            pool,
                            credit,
                            combination_validator=validate_combination,
                            eligible_components=eligible_components,
                        )
                        if change is not None:
                            performed.append(change.__dict__)
                    generation_event["operator_evolution"] = performed

                graph_interval = int(cfg_get(self.config, "evolution.graph_interval", 1))
                if (
                    self.structure_mode in {"dag", "dg"}
                    and graph_interval > 0
                    and generation % graph_interval == 0
                ):

                    def runtime_validation(edge, candidate_graph):
                        route = candidate_graph.route_through_edge(edge)
                        return (
                            route is not None
                            and len(route) <= initialized.max_pipeline_length
                            and self._runtime_validation_route(
                                route,
                                pool,
                                credit,
                                benchmark_batch[0],
                                initialized.sampling_temperature,
                            )
                        )

                    graph, delta = GraphEvolver(self.config, self.llm, self.prompt).evolve(
                        graph, credit, runtime_validation
                    )
                    policy.set_graph(graph)
                    generation_event["graph_evolution"] = (
                        delta.to_dict() if delta is not None else None
                    )
                evolution_elapsed = time.perf_counter() - evolution_started
                generation_event["timing"]["evolution_seconds"] = evolution_elapsed
                self.reporter.evolution(performed, delta, evolution_elapsed)

            if bool(cfg_get(self.config, "checkpoint.save_last", True)):
                checkpoint_started = time.perf_counter()
                checkpoints.save(
                    "last",
                    generation,
                    graph,
                    credit,
                    policy,
                    pool,
                    manifest,
                    {**best_metadata, "best_validation": best_validation},
                    self.llm.state_dict(),
                    self.rng,
                )
                checkpoint_elapsed = time.perf_counter() - checkpoint_started
                generation_event["timing"]["checkpoint_seconds"] = checkpoint_elapsed
                self.reporter.checkpoint("last", checkpoints.path("last"), checkpoint_elapsed)
            events.append(generation_event)
            completed_generation = generation
            with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(generation_event, ensure_ascii=False) + "\n")

        if checkpoints.exists("best"):
            restored = checkpoints.load("best", policy, pool, self.rng, load_optimizer=False)
            graph, credit = restored["graph"], restored["credit"]
            best_metadata = restored["checkpoint"].get("metadata", {})
        champion = PipelineSample(nodes=list(best_metadata.get("champion_nodes", baseline_route)))
        saved_implementations = [
            tuple(item) for item in best_metadata.get("champion_implementations", [])
        ]
        if saved_implementations and all(
            pool.version_path(f"{category}|{name}", version).is_file()
            for category, name, version in saved_implementations
        ):
            champion.implementations = saved_implementations
        else:
            champion = self._instantiate(
                champion, pool, credit, initialized.sampling_temperature, greedy=True
            )
        test_result = evaluator.evaluate(
            [champion],
            splits.test,
            standardize=False,
            evaluation_seed=self.seed + 20_000_000,
        )[0]
        result = {
            "status": (
                "validation_failed" if not best_metadata.get("champion_nodes")
                else "test_failed" if test_result.failed else "completed"
            ),
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "structure": self.structure_mode,
            "seed": self.seed,
            "best_validation_reward": best_validation,
            "test": test_result.to_log_dict(),
            "test_gap_percent": (
                None if test_result.failed or not best_metadata.get("champion_nodes")
                else -100.0 * float(test_result.raw_reward)
            ),
            "generations_planned": generations,
            "generations_completed": completed_generation,
            "llm_calls_used": self.llm.calls_used,
            "llm_audit": list(getattr(self.llm, "audit_records", [])),
            "markov_record_count": len(credit.records),
            "markov_total_trials": sum(record.trials for record in credit.records.values()),
            "events": events,
        }
        if getattr(policy, "torch", None) is not None and policy.torch.cuda.is_available():
            result["peak_cuda_memory_mb"] = policy.torch.cuda.max_memory_allocated() / (1024 * 1024)
        result_path = self.run_dir / "result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if _source_hashes(self.source_slots) != manifest["baseline_source_hashes"]:
            raise RuntimeError("Baseline slots changed during Paper-RL training")
        if result["status"] == "completed":
            self.reporter.complete(
                test_result.raw_reward,
                result_path,
                time.perf_counter() - run_started,
            )
        else:
            self.reporter.error(
                f"{result['status']}: "
                f"{test_result.failure_reason or 'no valid validation champion'}; "
                f"result={result_path}"
            )
        return result


def run_paper_rl_engine(**kwargs):
    return PaperRLTrainer(**kwargs).run()
