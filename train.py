# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import csv
import os
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import hydra
import imageio
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from agents.off_policy import OffPolicyAgent
from agents.on_policy import OnPolicyAgent
from utils import concat_labeled_images, create_reward_image
from wrappers import make_env


def main(cfg: DictConfig, exp_name: str, seed: int) -> None:
    wandb.init(
        project=f"rl_practice_{cfg.env_id}",
        config=OmegaConf.to_container(cfg, resolve=True),
        name=exp_name,
        save_code=True,
        settings=wandb.Settings(quiet=True),
    )

    # seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)

    # Create result directories and save files only if not in debug mode
    if not cfg.debug:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path(__file__).resolve().parent / "results" / f"{datetime_str}_{exp_name}"
        result_dir.mkdir(parents=True, exist_ok=True)

        # save seed to file
        with open(result_dir / "seed.txt", "w") as f:
            f.write(str(seed))

        # git show -sの結果とgit diffの結果を保存
        with open(result_dir / "git_info.txt", "w") as f:
            f.write(f"git show -s:\n{os.popen('git show -s').read()}\n")
            f.write(f"git diff:\n{os.popen('git diff').read()}\n")

        video_dir = result_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)

        image_dir = result_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
    else:
        result_dir = None
        video_dir = None
        image_dir = None
    image_save_interval = 500
    log_episode_path = result_dir / "log_episode.tsv" if result_dir is not None else None
    log_episode_file = None
    log_episode_writer = None

    # env setup
    env = make_env(cfg.env_id)
    env.action_space.seed(seed)

    target_score = cfg.target_score if cfg.target_score is not None else env.spec.reward_threshold

    start_time = time.time()

    # start the game
    global_step = 0
    score_list = []
    best_score = -float("inf")
    obs, _ = env.reset(seed=seed)
    step_limit = cfg.step_limit

    if cfg.agent_type == "off_policy":
        agent = OffPolicyAgent(cfg, env.observation_space, env.action_space)
    elif cfg.agent_type == "on_policy":
        agent = OnPolicyAgent(cfg, env.observation_space, env.action_space)
    else:
        raise ValueError(f"Unknown agent type: {cfg.agent_type}")

    parameter_count = sum(p.numel() for p in agent.network.parameters())
    print(f"Parameter count: {parameter_count:,}")

    episode_id = 0
    while True:
        # initialize episode
        obs, _ = env.reset()
        agent.initialize_for_episode()

        # initial action
        action, agent_info = agent.select_action(global_step, obs, 0.0, False, False)

        # initial render
        obs_for_render = obs.copy().transpose(1, 2, 0)
        reward_image = create_reward_image(0.0, 0.0)
        initial_rgb_image = concat_labeled_images(
            env.render(),
            obs_for_render,
            np.zeros_like(obs_for_render),
            reward_image,
        )
        bgr_image_list = [cv2.cvtColor(initial_rgb_image, cv2.COLOR_RGB2BGR)]

        # initial prediction for next step
        pred_image = agent_info.get("next_image", np.zeros_like(obs_for_render))
        pred_reward = agent_info.get("next_reward", 0.0)

        while True:
            global_step += 1

            # step
            obs, reward, terminated, truncated, env_info = env.step(action)
            action, agent_info = agent.step(global_step, obs, reward, terminated, truncated)

            # log
            elapsed_time_sec = time.time() - start_time
            elapsed_time_min = elapsed_time_sec / 60
            # wandbにログする前にndarrayを除外
            log_agent_info = {k: v for k, v in agent_info.items() if not isinstance(v, np.ndarray)}
            data_dict = {
                "global_step": global_step,
                "elapsed_time_min": elapsed_time_min,
                "SPS": global_step / elapsed_time_sec,
                "reward": reward,
                **log_agent_info,
            }
            data_dict["losses/pred_image_loss"] = np.mean(np.abs(pred_image - obs_for_render))
            data_dict["losses/pred_reward_loss"] = np.abs(pred_reward - reward)
            wandb.log(data_dict)

            # render
            obs_for_render = obs.copy().transpose(1, 2, 0)

            reward_image = create_reward_image(pred_reward, reward)
            rgb_image = concat_labeled_images(
                env.render(), obs_for_render, pred_image, reward_image
            )
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            bgr_image_list.append(bgr_image)
            if cfg.render:
                cv2.imshow("CarRacing", bgr_image)
                cv2.waitKey(1)

            if terminated or truncated:
                break

            if global_step >= step_limit:
                break

            # update prediction for next step
            pred_image = agent_info.get("next_image", np.zeros_like(obs_for_render))
            pred_reward = agent_info.get("next_reward", 0.0)

        if global_step >= step_limit:
            break

        score = env_info["episode"]["r"]
        score_list.append(score)
        score_list = score_list[-cfg.eval_range :]
        recent_average_score = np.mean(score_list)

        data_dict = {
            "global_step": global_step,
            "episodic_return": env_info["episode"]["r"],
            "episodic_length": env_info["episode"]["l"],
            "recent_average_score": recent_average_score,
        }
        wandb.log(data_dict)

        if result_dir is not None:
            if log_episode_writer is None:
                log_episode_file = open(log_episode_path, "w", newline="")
                fieldnames = list(data_dict.keys())
                log_episode_writer = csv.DictWriter(
                    log_episode_file, fieldnames=fieldnames, delimiter="\t"
                )
                log_episode_writer.writeheader()
            log_episode_writer.writerow(data_dict)
            log_episode_file.flush()

        is_solved = recent_average_score > target_score and episode_id >= cfg.eval_range

        if episode_id % 5 == 0 or is_solved:
            print(
                f"Ep: {episode_id}\tStep: {global_step}\tLast score: {score:.2f}\tAverage score: {recent_average_score:.2f}\tLength: {env_info['episode']['l']:.2f}"
            )

        is_best = score > best_score

        if is_best and result_dir is not None:
            with open(result_dir / "best_score.txt", "w") as f:
                f.write(f"{episode_id + 1}\t{score:.2f}")
            best_score = score
            video_path = video_dir / f"best_episode.mp4"
            if bgr_image_list:
                rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in bgr_image_list]
                imageio.mimsave(str(video_path), rgb_images, fps=10, macro_block_size=1)

                # 画像としても保存
                curr_image_dir = image_dir / "best_episode_frames"
                curr_image_dir.mkdir(parents=True, exist_ok=True)
                for idx, img in enumerate(bgr_image_list):
                    image_path = curr_image_dir / f"{idx:08d}.png"
                    cv2.imwrite(str(image_path), img)

        if (
            episode_id == 0 or (episode_id + 1) % image_save_interval == 0
        ) and result_dir is not None:
            video_path = video_dir / f"ep_{episode_id + 1:08d}.mp4"
            if bgr_image_list:
                rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in bgr_image_list]
                imageio.mimsave(str(video_path), rgb_images, fps=10, macro_block_size=1)

                # 画像としても保存
                curr_image_dir = image_dir / f"ep_{episode_id + 1:08d}_frames"
                curr_image_dir.mkdir(parents=True, exist_ok=True)
                for idx, img in enumerate(bgr_image_list):
                    image_path = curr_image_dir / f"{idx:08d}.png"
                    cv2.imwrite(str(image_path), img)

        episode_id += 1

        if is_solved:
            print(
                f"Solved! Running reward is now {recent_average_score} and the last episode runs to {score}!"
            )
            break

    env.close()
    if log_episode_file is not None:
        log_episode_file.close()
    wandb.finish()


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg: DictConfig) -> None:
    if cfg.debug:
        cfg.off_wandb = True
        cfg.render = 0
        cfg.step_limit = 100
        cfg.seq_len = 8

        # off_policy specific settings
        if cfg.agent_type == "off_policy":
            cfg.learning_starts = 10
            cfg.buffer_size = int(2e4)

        # on_policy specific settings
        if cfg.agent_type == "on_policy":
            cfg.buffer_capacity = 50

    if cfg.off_wandb:
        os.environ["WANDB_MODE"] = "offline"

    if not os.environ.get("DISPLAY"):
        print("Because a headless environment is detected, rendering is automatically disabled.")
        cfg.render = 0

    exp_name = f"{cfg.agent_type.upper()}_{cfg.exp_name}"
    seed = cfg.seed if cfg.seed != -1 else np.random.randint(0, 10000)

    for i in range(cfg.trial_num):
        suffix = f"_{i:02d}" if cfg.trial_num > 1 else ""
        main(cfg, exp_name + suffix, seed + i)


if __name__ == "__main__":
    run()
