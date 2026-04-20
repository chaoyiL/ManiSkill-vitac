#!/usr/bin/env python3
import argparse

from openpi.training import config as _config
from openpi.policies import policy_config
import openpi.policies.vb_policy_vitac as vb_policy_vitac


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-step inference.")
    parser.add_argument(
        "--config",
        default="pi05_chaoyi_vitac",
        help="Config name for policy.",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="checkpoints/pi05_chaoyi_vitac/my_experiment/50",
        help="Checkpoint directory for the trained policy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _config.get_config(args.config)
    ckpt_dir = args.ckpt_dir

    # Create a trained policy.
    policy = policy_config.create_trained_policy(config, ckpt_dir)

    # Run inference on a dummy example.
    example = vb_policy_vitac.make_vitac_example()
    action_chunk = policy.infer(example)["actions"]
    print("action_chunk: ", action_chunk)
    print("action_chunk shape: ", action_chunk.shape)


if __name__ == "__main__":
    main()