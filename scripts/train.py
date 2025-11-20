"""
Training script for Chiplet-Gym with Stable-Baselines3
Supports PPO, SAC, and other SB3 algorithms
"""

import argparse

import gymnasium as gym
import torch
from chiplet_gym.utils.config import DEFAULT_CONFIG
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


def main():
    parser = argparse.ArgumentParser(description="Train Chiplet-Gym with RL")
    parser.add_argument(
        "--algo", type=str, default="PPO", choices=["PPO", "SAC"], help="RL algorithm"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100000, help="Total training timesteps"
    )
    parser.add_argument(
        "--save_freq", type=int, default=10000, help="Save checkpoint every N timesteps"
    )
    parser.add_argument("--log_dir", type=str, default="./logs", help="Log directory")
    parser.add_argument(
        "--model_dir", type=str, default="./models", help="Model save directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "mps", "cpu", "auto"],
        help="Device",
    )

    args = parser.parse_args()

    # # Check CUDA availability
    # if args.device == "cuda" and not torch.cuda.is_available():
    #     print("CUDA not available, using CPU")
    #     args.device = "cpu"

    # print(f"Using device: {args.device}")

    # Auto-detect best device
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
            print("Auto-detected: CUDA GPU")
        elif torch.backends.mps.is_available():
            args.device = "mps"
            print("Auto-detected: Apple Silicon GPU (MPS)")
        else:
            args.device = "cpu"
            print("Auto-detected: CPU only")
    else:
        # Validate requested device
        if args.device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available, falling back to CPU")
            args.device = "cpu"
        elif args.device == "mps" and not torch.backends.mps.is_available():
            print("WARNING: MPS requested but not available, falling back to CPU")
            args.device = "cpu"

    print(f"Using device: {args.device}")

    # Create environment
    print("Creating Chiplet-Gym environment...")
    env = gym.make("ChipletEnv-v0", config=DEFAULT_CONFIG)
    env = Monitor(env, args.log_dir)

    # Validate environment
    print("Validating environment with check_env()...")
    check_env(env, warn=True)
    print("Environment validation passed!")

    # Create eval environment
    eval_env = gym.make("ChipletEnv-v0", config=DEFAULT_CONFIG)
    eval_env = Monitor(eval_env, args.log_dir + "/eval")

    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{args.model_dir}/best",
        log_path=args.log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=f"{args.model_dir}/checkpoints",
        name_prefix="chiplet_gym",
    )

    # Create RL model
    print(f"\nInitializing {args.algo} agent...")
    if args.algo == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=args.log_dir,
            device=args.device,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            normalize_advantage=True,
        )
    else:
        raise NotImplementedError(f"Algorithm {args.algo} not yet implemented")

    # Train
    print(f"\nStarting training for {args.timesteps} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    final_model_path = f"{args.model_dir}/final_{args.algo.lower()}_chiplet_gym"
    model.save(final_model_path)
    print(f"\nTraining complete! Model saved to {final_model_path}")

    # Cleanup
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
