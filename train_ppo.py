# Reference: https://github.com/xtma/pytorch_car_caring
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
from agents.ppo import PpoAgent
from utils import concat_images
from wrappers import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--agent_type", type=str, default="ppo", choices=["ppo"])
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--render", type=int, default=1, choices=[0, 1])
    parser.add_argument("--target_score", type=float, default=None)
    parser.add_argument("--off_wandb", action="store_true")
    parser.add_argument("--step_limit", type=int, default=200_000)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--buffer_capacity", type=int, default=2000)
    parser.add_argument("--seq_len", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--model_name", type=str, default="default", choices=["default", "paligemma"]
    )
    return parser.parse_args()




if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.off_wandb = True
        args.render = 0
        args.step_limit = 100
        args.buffer_capacity = args.batch_size = 32

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

    video_dir = result_dir / "video"
    image_dir = result_dir / "image"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_save_interval = 100
    log_step = []
    log_episode = []

    device = torch.device("cuda")

    # env setup
    env = make_env()
    env.action_space.seed(seed)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = PpoAgent(args, env.observation_space, env.action_space)

    target_score = args.target_score if args.target_score is not None else env.spec.reward_threshold

    start = time.time()

    global_step = 0
    score_list = []
    step_limit = args.step_limit
    reward = 0.0

    for i_ep in range(args.step_limit):
        state, _ = env.reset()

        if i_ep % image_save_interval == 0:
            curr_image_dir = image_dir / f"ep_{i_ep:08d}"
            curr_image_dir.mkdir(parents=True, exist_ok=True)

        reward_list = []
        first_value = None

        while True:
            global_step += 1
            action, a_logp, value, net_out_dict = agent.select_action_original(reward, state)
            state_, reward, done, die, info = env.step(action)
            done = bool(done or die)
            normed_reward = reward / 10.0
            if len(reward_list) == 0:
                first_value = value
            reward_list.append(reward)

            # render
            if args.model_name == "default":
                observation_img = np.zeros((96, 96, 3), dtype=np.float32)
                reconstructed_img = np.zeros((96, 96, 3), dtype=np.float32)
                bgr_array = concat_images(env.render(), observation_img, reconstructed_img)
            elif args.model_name == "paligemma":
                rgb_array = env.render()
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            if args.render:
                cv2.imshow("CarRacing", bgr_array)
                cv2.waitKey(1)
            if i_ep % image_save_interval == 0:
                cv2.imwrite(str(curr_image_dir / f"{global_step:08d}.png"), bgr_array)

            data_dict = {
                "global_step": global_step,
                "a_logp": a_logp,
                "value": value,
                "reward": reward,
                "normed_reward": normed_reward,
            }

            for key in ["x", "value_x", "policy_x"]:
                value_tensor = net_out_dict[key]
                data_dict[f"activation/{key}_norm"] = value_tensor.norm(dim=1).mean().item()
                data_dict[f"activation/{key}_mean"] = value_tensor.mean(dim=1).mean().item()
                data_dict[f"activation/{key}_std"] = value_tensor.std(dim=1).mean().item()

            if agent.store((state, action, a_logp, normed_reward, value, done)):
                print("updating", end="\r")
                train_result = agent.update()
                data_dict.update(train_result)
                fixed_data = {k.replace("ppo/", ""): v for k, v in data_dict.items()}
                log_step.append(fixed_data)
                log_step_df = pd.DataFrame(log_step)
                log_step_df.to_csv(result_dir / "log_step.tsv", sep="\t", index=False)

                for name, p in agent.net.named_parameters():
                    data_dict[f"params/{name}"] = p.norm().item()

                wandb.log(data_dict)
            elif global_step % 100 == 0:
                wandb.log(data_dict)

            state = state_
            if done or die:
                break

            if global_step >= step_limit:
                break

        if global_step >= step_limit:
            break

        score = info["episode"]["r"]
        score_list.append(score)
        score_list = score_list[-20:]
        recent_average_score = np.mean(score_list)
        is_solved = recent_average_score > target_score

        weighted_reward = 0.0
        coeff = 1.0
        for r in reward_list:
            weighted_reward += coeff * r
            coeff *= agent.gamma

        if i_ep % args.log_interval == 0 or is_solved:
            elapsed_time = time.time() - start
            msec_per_step = 1000.0 * elapsed_time / global_step
            print(
                f"Time: {elapsed_time:.2f}s\t"
                f"Msec/step: {msec_per_step:.1f}\t"
                f"Ep: {i_ep}\t"
                f"Step: {global_step}\t"
                f"Last score: {score:.2f}\t"
                f"Average score: {recent_average_score:.2f}\t"
                f"Length: {info['episode']['l']:.2f}"
            )
        data_dict = {
            "global_step": global_step,
            "episode": i_ep,
            "score": score,
            "recent_average_score": recent_average_score,
            "episodic_return": info["episode"]["r"],
            "episodic_length": info["episode"]["l"],
            "weighted_reward": weighted_reward,
            "first_value": first_value,
        }
        wandb.log(data_dict)

        log_episode.append(data_dict)
        log_episode_df = pd.DataFrame(log_episode)
        log_episode_df.to_csv(result_dir / "log_episode.tsv", sep="\t", index=False)
        if is_solved:
            print(
                f"Solved! Running reward is now {recent_average_score} and the last episode runs to {score}!"
            )
            break
