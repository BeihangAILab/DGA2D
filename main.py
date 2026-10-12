from __future__ import annotations

import importlib
import sys
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.core.configuration import (  # noqa: E402
    cfg_get,
    configure_dataset_root,
)
from src.core.contracts import validate_domain_evaluator  # noqa: E402
from src.core.reporting import (  # noqa: E402
    ConsoleReporter,
    configure_third_party_logging,
)


def dispatch_run(cfg: DictConfig, run_dir: Path, reporter: ConsoleReporter):
    """Run the configured training experiment."""
    configure_dataset_root(
        PROJECT_ROOT,
        cfg_get(cfg, "dataset.root", "data"),
    )
    problem_name = str(cfg.problem.name).lower()
    domain_evaluator = importlib.import_module(f"src.problems.{problem_name}.domain_evaluator")
    validate_domain_evaluator(domain_evaluator)

    problem_dir = PROJECT_ROOT / "src" / "problems" / problem_name
    slots_dir = problem_dir / "slots"
    if str(problem_dir) not in sys.path:
        sys.path.insert(0, str(problem_dir))

    action = str(cfg_get(cfg, "engine.action", "train")).strip().lower()
    if action != "train":
        raise ValueError(f"Unsupported engine.action={action!r}; expected 'train'")

    from src.core.trainer import run_paper_rl_engine
    from src.utils import llm_client

    prompt_context = domain_evaluator.get_prompt(root_dir=str(PROJECT_ROOT))

    allowed_nodes = None
    if hasattr(cfg.problem, "node_ablation_level") and hasattr(
        domain_evaluator, "get_allowed_nodes"
    ):
        allowed_nodes = domain_evaluator.get_allowed_nodes(cfg.problem.node_ablation_level)
        expected_count = int(cfg.problem.node_ablation_level)
        if len(allowed_nodes) != expected_count:
            raise ValueError(
                f"{problem_name.upper()} V{expected_count} declares "
                f"{len(allowed_nodes)} nodes; expected exactly {expected_count}"
            )
        from src.core.operators import discover_operator_versions

        discovered = discover_operator_versions(slots_dir, allowed_nodes)
        if len(discovered) != expected_count:
            raise ValueError(
                f"{problem_name.upper()} V{expected_count} exposes "
                f"{len(discovered)} executable nodes; expected exactly {expected_count}"
            )

    client = llm_client.create_client(
        api_key=cfg_get(cfg, "llm.api_key", None),
        base_url=cfg_get(cfg, "llm.base_url", None),
        model_name=cfg_get(cfg, "llm.model_name", None),
        temperature=cfg_get(cfg, "llm.temperature", None),
        max_calls=cfg_get(cfg, "llm.max_calls", 500),
        request_timeout_seconds=cfg_get(cfg, "llm.request_timeout_seconds", 60.0),
        retries=cfg_get(cfg, "llm.retries", 2),
        warning_callback=reporter.warning,
    )

    import torch

    requested_device = str(cfg_get(cfg, "policy.device", "auto"))
    actual_device = requested_device
    if requested_device == "auto":
        actual_device = "cuda" if torch.cuda.is_available() else "cpu"
    if actual_device == "cpu":
        actual_device = f"CPU ({int(cfg_get(cfg, 'policy.cpu_threads', 4))} threads)"
    else:
        actual_device = actual_device.upper()
    reporter.banner(
        problem=problem_name.upper(),
        dataset=str(cfg.problem.target_group),
        policy=str(cfg.policy.model_name),
        device=actual_device,
        run=str(run_dir),
    )
    reporter.info(
        f"Structure     {str(cfg.structure.name).upper()} | "
        f"Proposer {str(cfg_get(cfg, 'llm.provider', 'openai_compatible')).upper()}"
    )

    return run_paper_rl_engine(
        domain_evaluator=domain_evaluator,
        prompt_context=prompt_context,
        config=cfg,
        source_slots=str(slots_dir),
        run_dir=str(run_dir),
        project_root=str(PROJECT_ROOT),
        allowed_nodes=allowed_nodes,
        llm_client=client,
        reporter=reporter,
    )


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    reporter = ConsoleReporter(
        run_dir,
        level=str(cfg_get(cfg, "logging.level", "INFO")),
        color=str(cfg_get(cfg, "logging.color", "auto")),
        show_candidates=bool(cfg_get(cfg, "logging.show_candidates", True)),
    )
    configure_third_party_logging(
        str(cfg_get(cfg, "logging.third_party_level", "WARNING")), reporter
    )
    try:
        result = dispatch_run(cfg, run_dir, reporter)
        if result["status"] != "completed":
            reporter.error(
                f"Run failed: status={result['status']}; "
                f"reason={result.get('test', {}).get('failure_reason') or 'no valid validation champion'}; "
                f"result={run_dir / 'result.json'}"
            )
            raise SystemExit(1)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        reporter.exception("Training failed", exc)
        raise SystemExit(1) from None
    finally:
        configure_third_party_logging(str(cfg_get(cfg, "logging.third_party_level", "WARNING")))
        reporter.close()


if __name__ == "__main__":
    main()
