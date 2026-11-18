from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.config import load_config
from dataset.incremental_manager import IncrementalDataManager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dm = IncrementalDataManager(cfg)
    print('Canonical class discovery by task:')
    for tid in range(dm.num_tasks()):
        info = dm.prepare_task(tid)
        print(f"\nTask {tid} {info['task_name']}")
        print(f"classes={len(info['current_ids'])}, bridges={len(info['recurring_ids'])}, novel={len(info['novel_ids'])}")
        if info['recurring_ids']:
            print('bridges:', [dm.registry.id_to_class[i] for i in info['recurring_ids']])
        if info['novel_ids']:
            print('novel:', [dm.registry.id_to_class[i] for i in info['novel_ids']])


if __name__ == '__main__':
    main()
