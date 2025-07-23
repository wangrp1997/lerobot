# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
)
from lerobot.scripts.rl.gym_manipulator import make_robot_env
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    so101_leader,  # noqa: F401
)

logging.basicConfig(level=logging.INFO)


def eval_policy(env, policy, n_episodes):
    # 优先从 env 读取 fps，其次 config，再否则默认 10
    fps = getattr(env, 'fps', None)
    if fps is None and hasattr(env, 'unwrapped'):
        fps = getattr(env.unwrapped, 'fps', None)
    if fps is None:
        fps = 10
    sum_reward_episode = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        while True:
            start_time = time.perf_counter()
            action = policy.select_action(obs)
            action[..., -1] = (action[..., -1] + 1)  # 把 [-1, 1] 映射到 [0, 2]
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
            # 控制评估速度与训练一致
            dt_time = time.perf_counter() - start_time
            sleep_time = max(0, 1 / fps - dt_time)
            time.sleep(sleep_time)
        sum_reward_episode.append(episode_reward)

    logging.info(f"Success after 20 steps {sum_reward_episode}")
    logging.info(f"success rate {sum(sum_reward_episode) / len(sum_reward_episode)}")


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    env_cfg = cfg.env
    env = make_robot_env(env_cfg)
    dataset_cfg = cfg.dataset
    dataset = LeRobotDataset(repo_id=dataset_cfg.repo_id)
    dataset_meta = dataset.meta

    policy = make_policy(
        cfg=cfg.policy,
        # env_cfg=cfg.env,
        ds_meta=dataset_meta,
    )

    policy.from_pretrained(env_cfg.pretrained_policy_name_or_path)
    policy.eval()

    eval_policy(env, policy=policy, n_episodes=10)


if __name__ == "__main__":
    main()
