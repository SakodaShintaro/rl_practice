# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import pandas as pd
import torch

import wandb
from agents.avg import AvgAgent
from agents.sac import SacAgent
from utils import concat_images
from wrappers import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--agent_type", type=str, default="sac", choices=["sac", "avg"])
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--render", type=int, default=1, choices=[0, 1])
    parser.add_argument("--off_wandb", action="store_true")
    parser.add_argument("--image_encoder", type=str, default="ae", choices=["ae", "smolvlm"])
    parser.add_argument("--actor_hidden_dim", type=int, default=512)
    parser.add_argument("--actor_block_num", type=int, default=1)
    parser.add_argument("--critic_hidden_dim", type=int, default=1024)
    parser.add_argument("--critic_block_num", type=int, default=1)
    parser.add_argument("--num_bins", type=int, default=51)
    parser.add_argument("--sparsity", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--step_limit", type=int, default=200_000)
    parser.add_argument("--debug", action="store_true")

    # for SAC
    parser.add_argument("--buffer_size", type=int, default=int(2e4))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_starts", type=int, default=4000)
    parser.add_argument("--action_noise", type=float, default=0.0)
    parser.add_argument("--predictor_hidden_dim", type=int, default=1024)
    parser.add_argument("--predictor_block_num", type=int, default=2)
    parser.add_argument("--apply_masks_during_training", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_weight_projection", action="store_true")
    parser.add_argument("--enable_sequence_modeling", action="store_true")

    # for AVG
    parser.add_argument("--use_eligibility_trace", action="store_true")
    parser.add_argument("--et_lambda", default=0.8, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.off_wandb = True
        args.learning_starts = 10
        args.render = 0
        args.step_limit = 100

    if args.off_wandb:
        os.environ["WANDB_MODE"] = "offline"

    exp_name = f"{args.agent_type.upper()}_{args.exp_name}"
    wandb.init(project="rl_practice", config=vars(args), name=exp_name, save_code=True)

    # seeding
    seed = args.seed if args.seed != -1 else np.random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(__file__).resolve().parent / "results" / f"{datetime_str}_{exp_name}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # save seed to file
    with open(result_dir / "seed.txt", "w") as f:
        f.write(str(seed))

    image_dir = result_dir / "image"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_save_interval = 50
    log_step = []
    log_episode = []

    # env setup
    env = make_env()
    env.action_space.seed(seed)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    start_time = time.time()

    # start the game
    global_step = 0
    score_list = []
    obs, _ = env.reset(seed=seed)
    curr_image_dir = None
    step_limit = args.step_limit

    # Initialize dummy prediction images
    curr_obs_float = np.zeros((96, 96, 3), dtype=np.float32)
    pred_obs_float = np.zeros((96, 96, 3), dtype=np.float32)

    if args.agent_type == "sac":
        agent = SacAgent(args, env.observation_space, env.action_space)
    elif args.agent_type == "avg":
        agent = AvgAgent(args, env.observation_space, env.action_space)
    else:
        raise ValueError(f"Unknown agent type: {args.agent_type}")

    for episode_id in range(10000):
        if episode_id == 0 or (episode_id + 1) % image_save_interval == 0:
            curr_image_dir = image_dir / f"ep_{episode_id + 1:08d}"
            curr_image_dir.mkdir(parents=True, exist_ok=True)
        else:
            curr_image_dir = None

        obs, _ = env.reset()
        agent.initialize_for_episode()
        reward_list = []

        while True:
            global_step += 1

            # select action
            action, action_info = agent.select_action(global_step, obs)

            # step
            next_obs, reward, termination, truncation, env_info = env.step(action)

            # render
            if args.render:
                bgr_array = concat_images(env.render(), curr_obs_float, pred_obs_float)
                cv2.imshow("CarRacing", bgr_array)
                cv2.waitKey(1)

            if curr_image_dir is not None:
                bgr_array = concat_images(env.render(), curr_obs_float, pred_obs_float)
                cv2.imwrite(str(curr_image_dir / f"{global_step:08d}.png"), bgr_array)

            if termination or truncation:
                break

            if global_step >= step_limit:
                break

            # process environment feedback
            feedback_info = agent.process_env_feedback(
                global_step, next_obs, action, reward, termination, truncation
            )

            obs = next_obs

            if global_step % 10 == 0:
                elapsed_time = time.time() - start_time
                data_dict = {
                    "global_step": global_step,
                    "charts/elapse_time_sec": elapsed_time,
                    "charts/SPS": global_step / elapsed_time,
                    "reward": reward,
                    **action_info,
                    **feedback_info,
                }

                wandb.log(data_dict)
                log_step.append(data_dict)

        if global_step >= step_limit:
            break

        score = env_info["episode"]["r"]
        score_list.append(score)
        score_list = score_list[-20:]
        recent_average_score = np.mean(score_list)

        data_dict = {
            "global_step": global_step,
            "episodic_return": env_info["episode"]["r"],
            "episodic_length": env_info["episode"]["l"],
            "recent_average_score": recent_average_score,
        }
        wandb.log(data_dict)

        log_episode.append(data_dict)
        log_episode_df = pd.DataFrame(log_episode)
        log_episode_df.to_csv(result_dir / "log_episode.tsv", sep="\t", index=False)

        is_solved = recent_average_score > env.spec.reward_threshold

        if episode_id % 5 == 0 or is_solved:
            print(
                f"Ep: {episode_id}\tStep: {global_step}\tLast score: {score:.2f}\tAverage score: {recent_average_score:.2f}\tLength: {env_info['episode']['l']:.2f}"
            )

        episode_id += 1

        if is_solved:
            print(
                f"Solved! Running reward is now {recent_average_score} and the last episode runs to {score}!"
            )
            break

    env.close()
