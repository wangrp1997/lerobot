import os
from pathlib import Path
from lerobot.datasets.utils import write_stats, load_episodes_stats
from lerobot.datasets.compute_stats import aggregate_stats

# 1. 设置你的 meta 目录路径
meta_dir = Path(os.path.expanduser("~/.cache/huggingface/lerobot/rpwang/il_gym4_state"))

# 2. 读取所有 episode 的统计
episodes_stats = load_episodes_stats(meta_dir)
if not episodes_stats:
    raise RuntimeError(f"没有找到 episodes_stats.jsonl，请先采集数据并生成 episode 统计！")

# 3. 聚合为全局统计
stats = aggregate_stats(list(episodes_stats.values()))

# 4. 写入 stats.json
write_stats(stats, meta_dir)
print(f"已生成 {meta_dir / 'stats.json'}，可直接用于 config 的 dataset_stats 字段！")