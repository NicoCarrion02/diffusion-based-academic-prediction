"""
Main entrypoint for the Diffusion-Based Academic Prediction Project.
Usage:
    python run.py --step all
    python run.py --step download
    python run.py --step convert
    python run.py --step metadata
    python run.py --step prepare
    python run.py --step select
    python run.py --step diffusion
    python run.py --step benchmark
"""

import argparse
import sys
import logging
from src.pipeline import MasterPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MAIN")


def main():
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

    args = parser.parse_args()

    pipeline = MasterPipeline(
        config_path=args.config,
        features_path=args.features,
        models_path=args.models,
    )

    logger.info(f"Executing pipeline with step='{args.step}', force={args.force}")

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
        pipeline.run_benchmark(force=args.force)
    elif args.step == "all":
        pipeline.run_download()
        pipeline.run_convert(force=args.force)
        pipeline.run_metadata_dictionary(force=args.force)
        pipeline.run_prepare(force=args.force)
        pipeline.run_feature_selection(force=args.force)
        pipeline.run_diffusion_balancing(force=args.force)
        pipeline.run_benchmark(force=args.force)

    logger.info("Command completed successfully.")


if __name__ == "__main__":
    main()
