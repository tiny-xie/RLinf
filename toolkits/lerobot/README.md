# LeRobot dataset tools

## Delete episodes by index

`delete_lerobot_episodes.py` deletes one or more current `episode_index`
values from a single LeRobot v2.1 dataset directory. It then compacts the
remaining episode indices and global frame indices and synchronizes
`meta/info.json`, `meta/episodes.jsonl`, and
`meta/episodes_stats.jsonl`.

Preview an operation first:

```bash
python toolkits/lerobot/delete_lerobot_episodes.py \
  --dataset-dir /data/run/rank_0/id_2 \
  --episode-index 0 5 9 \
  --dry-run
```

Perform the deletion:

```bash
python toolkits/lerobot/delete_lerobot_episodes.py \
  --dataset-dir /data/run/rank_0/id_2 \
  --episode-index 0 5 9 \
  --yes
```

The indices are interpreted against the dataset at command start. For
example, `--episode-index 0 5` deletes the original episodes 0 and 5, not
episode 0 followed by the newly renumbered episode 5.

To retain the complete original dataset, add
`--backup-dir /path/to/backup/id_2`. Without this option, the original is
deleted only after a rebuilt temporary dataset passes consistency checks.

The tool currently supports embedded-image Parquet datasets and rejects
video-backed datasets.
