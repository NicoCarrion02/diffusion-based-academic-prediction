"""
Main entrypoint for the Diffusion-Based Academic Prediction Project.
Usage:
    # Full end-to-end execution (Automatic Nested CV model selection)
    python run.py --step all

    # Individual steps
    python run.py --step download
    python run.py --step convert
    python run.py --step metadata
    python run.py --step prepare
    python run.py --step select
    python run.py --step diffusion
    python run.py --step benchmark

    # Manual single-model tuning & evaluation
    python run.py --step benchmark --mode manual --model xgboost --regime both --metric f1_high_performing
    python run.py --step benchmark --mode manual --model random_forest --regime balanced --n-trials 20
    python run.py --step benchmark --mode manual --model tabpfn --regime both
"""

import argparse
import logging
from src.pipeline import MasterPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MAIN")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prediction of Student Academic Performance Using Diffusion Models (PISA 2022)"
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["download", "convert", "metadata", "prepare", "select", "diffusion", "benchmark", "all"],
        help="Pipeline step to execute (default: 'all')",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to central config file",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="configs/features.yaml",
        help="Path to features configuration file",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="configs/models.yaml",
        help="Path to models configuration file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of steps, ignoring cached files",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["auto", "manual"],
        help="Model selection mode: 'auto' (rank all candidates) or 'manual' (explicit --model)",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=None,
        choices=["random_forest", "xgboost", "tabpfn"],
        help="Explicit model(s) to tune and evaluate (required when --mode manual)",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default=None,
        choices=["both", "unbalanced", "balanced"],
        help="Training class-balance regime(s) to evaluate (default from configs/models.yaml: 'both')",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Primary validation metric for tuning and champion selection (e.g. f1_high_performing, balanced_accuracy, pr_auc, roc_auc, recall_high_performing)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override Optuna trial budget per model during inner Nested CV tuning",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Disable inner hyperparameter search and evaluate baseline configurations directly",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "manual" and not args.model:
        parser.error("--mode manual requires specifying at least one --model {random_forest,xgboost,tabpfn}.")

    pipeline = MasterPipeline(
        config_path=args.config,
        features_path=args.features,
        models_path=args.models,
    )

    logger.info(
        f"Executing pipeline with step='{args.step}', force={args.force}, "
        f"mode={args.mode or 'config'}, model={args.model or 'config'}"
    )

    if args.step == "download":
        pipeline.run_download()
    elif args.step == "convert":
        pipeline.run_convert(force=args.force)
    elif args.step == "metadata":
        pipeline.run_metadata_dictionary(force=args.force)
    elif args.step == "prepare":
        pipeline.run_prepare(force=args.force)
    elif args.step == "select":
        pipeline.run_feature_selection(force=args.force)
    elif args.step == "diffusion":
        pipeline.run_diffusion_balancing(force=args.force)
    elif args.step == "benchmark":
        pipeline.run_benchmark(
            force=args.force,
            mode=args.mode,
            selected_models=args.model,
            regime=args.regime,
            primary_metric=args.metric,
            n_trials_override=args.n_trials,
            no_tune=args.no_tune,
        )
    elif args.step == "all":
        pipeline.run_download()
        pipeline.run_convert(force=args.force)
        pipeline.run_metadata_dictionary(force=args.force)
        pipeline.run_prepare(force=args.force)
        pipeline.run_feature_selection(force=args.force)
        pipeline.run_diffusion_balancing(force=args.force)
        pipeline.run_benchmark(
            force=args.force,
            mode=args.mode,
            selected_models=args.model,
            regime=args.regime,
            primary_metric=args.metric,
            n_trials_override=args.n_trials,
            no_tune=args.no_tune,
        )

    logger.info("Command completed successfully.")


if __name__ == "__main__":
    main()
