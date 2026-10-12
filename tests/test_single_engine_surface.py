import unittest
from pathlib import Path
from unittest import mock

from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException


ROOT = Path(__file__).resolve().parents[1]


class SingleEngineSurfaceTests(unittest.TestCase):
    def test_cli_exit_status_matches_final_evaluation(self):
        import main
        from omegaconf import OmegaConf

        for status in ("completed", "validation_failed", "test_failed"):
            with (
                self.subTest(status=status),
                mock.patch.object(main.HydraConfig, "get", return_value=OmegaConf.create(
                    {"runtime": {"output_dir": str(ROOT / "artifacts")}}
                )),
                mock.patch.object(main, "ConsoleReporter") as reporter_type,
                mock.patch.object(main, "configure_third_party_logging"),
                mock.patch.object(main, "dispatch_run", return_value={
                    "status": status, "test": {"failure_reason": "pipeline timeout"}
                }),
            ):
                if status == "completed":
                    main.main.__wrapped__(OmegaConf.create({}))
                    reporter_type.return_value.error.assert_not_called()
                else:
                    with self.assertRaises(SystemExit) as raised:
                        main.main.__wrapped__(OmegaConf.create({}))
                    self.assertEqual(raised.exception.code, 1)
                    self.assertIn(status, reporter_type.return_value.error.call_args.args[0])
                reporter_type.return_value.close.assert_called_once()

    def test_default_config_uses_general_generation_driven_training(self):
        with initialize_config_dir(config_dir=str((ROOT / "cfg").resolve()), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=["problem=fssp"],
            )
        self.assertEqual(cfg.engine.action, "train")
        self.assertEqual(cfg.engine.generations, 50)
        self.assertEqual(cfg.engine.candidates_per_generation, 4)
        self.assertEqual(cfg.engine.shared_instances_per_generation, 3)
        self.assertEqual(cfg.engine.pipeline_steps, 500)
        self.assertEqual(cfg.engine.pipeline_timeout, 90.0)
        self.assertEqual(cfg.initialization.max_operator_count.max, 20)
        self.assertEqual(cfg.initialization.max_pipeline_length.max, 10)
        self.assertFalse(cfg.initialization.max_pipeline_length.llm_select)
        self.assertFalse(cfg.initialization.sampling_temperature.llm_select)
        self.assertEqual(cfg.evolution.graph_interval, 1)
        self.assertEqual(cfg.validation.interval, 5)
        self.assertEqual(cfg.validation.beam_size, 4)
        self.assertEqual(cfg.structure.name, "dg")
        self.assertEqual(cfg.llm.provider, "openai_compatible")
        self.assertEqual(
            set(cfg.engine),
            {
                "action",
                "seed",
                "run_tier",
                "generations",
                "candidates_per_generation",
                "shared_instances_per_generation",
                "pipeline_steps",
                "pipeline_timeout",
                "evaluator_workers",
                "failure_reward",
                "reward_epsilon",
            },
        )
        self.assertNotIn("instances_per_candidate", cfg.engine)


    def test_unsupported_action_is_rejected_before_training(self):
        import main
        with initialize_config_dir(config_dir=str((ROOT / "cfg").resolve()), version_base=None):
            cfg = compose(config_name="config", overrides=["problem=fssp", "engine.action=invalid"])
        with self.assertRaisesRegex(ValueError, "expected 'train'"):
            main.dispatch_run(cfg, ROOT / "artifacts", None)

    def test_removed_instance_count_name_is_rejected(self):
        with (
            initialize_config_dir(config_dir=str((ROOT / "cfg").resolve()), version_base=None),
            self.assertRaises(ConfigCompositionException),
        ):
            compose(
                config_name="config",
                overrides=["problem=fssp", "engine.instances_per_candidate=3"],
            )

    def test_all_domains_and_structure_modes_compose(self):
        problems = {path.stem for path in (ROOT / "cfg" / "problem").glob("*.yaml")}
        structures = {path.stem for path in (ROOT / "cfg" / "structure").glob("*.yaml")}
        self.assertEqual(len(problems), 12)
        self.assertEqual(structures, {"fixed", "linear", "dag", "dg"})
        with initialize_config_dir(config_dir=str((ROOT / "cfg").resolve()), version_base=None):
            for problem in sorted(problems):
                for structure in sorted(structures):
                    cfg = compose(
                        config_name="config",
                        overrides=[f"problem={problem}", f"structure={structure}"],
                    )
                    self.assertEqual(cfg.structure.name, structure)

    def test_public_readme_documents_locked_training_environment(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        for text in ("constrained graph walks", "reinforce", "markov credit",
                     "uv sync --locked --extra cu126", "python 3.13",
                     "fixed evolution batch", "credit ablation"):
            self.assertIn(text, readme)
        self.assertTrue((ROOT / "assets" / "framework.png").is_file())

    def test_paper_experiment_presets_define_the_complete_matrices(self):
        expected = {
            "main_table.yaml": (
                "problem: tsp,fjsp,mis,3dclp",
                "llm: deepseek_v4_flash,gpt_5_6_sol",
            ),
            "structure_ablation.yaml": (
                "problem: cvrp,fjsp,mis,3dclp",
                "structure: fixed,linear,dag,dg",
            ),
            "credit_order_ablation.yaml": (
                "problem: tsp,fjsp,mis,3dclp",
                "credit: zero,first,second,full",
            ),
        }
        for filename, matrix_lines in expected.items():
            text = (ROOT / "cfg" / "experiment" / filename).read_text(encoding="utf-8")
            self.assertIn("generations: 50", text)
            self.assertIn("engine.seed: 0,1,2,3,4,5,6,7,8,9", text)
            for matrix_line in matrix_lines:
                self.assertIn(matrix_line, text)

    def test_core_package_is_flattened(self):
        core = ROOT / "src" / "core"
        files = {path.name for path in core.glob("*.py")}
        self.assertEqual(
            files,
            {
                "__init__.py",
                "checkpoint.py",
                "configuration.py",
                "contracts.py",
                "credit.py",
                "dataset.py",
                "evolution.py",
                "execution.py",
                "operators.py",
                "policy.py",
                "reporting.py",
                "structure.py",
                "trainer.py",
            },
        )
        self.assertFalse((core / "paper_rl").exists())


if __name__ == "__main__":
    unittest.main()
