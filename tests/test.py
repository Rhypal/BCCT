from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from dataset.incremental_manager import IncrementalDataManager
from engine.trainer import IncrementalTrainer
from model.dinov3_backbone import FrozenDINOv3
from model.bridge_cdil import BridgeCDILModel
from model.style_router import DomainStyleBank
from tools.stats import ClassStatsBank
from tools.config import load_config
from tools.seed import seed_everything
from tools.checkpoint import load_model_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument(
        '--refit-density-keys',
        action='store_true',
        help=(
            'Rebuild compact density keys from cached training features using '
            'the router configuration. Does not train or modify the checkpoint.'
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get('seed', 42)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dm = IncrementalDataManager(cfg)
    backbone = FrozenDINOv3(cfg['backbone']).to(device)
    model = BridgeCDILModel(backbone.feature_dim, backbone.style_dim, cfg)
    stats_bank = ClassStatsBank()
    style_bank = DomainStyleBank()
    state = load_model_state(args.checkpoint, model, stats_bank, style_bank, dm.registry, map_location='cpu')
    model.to(device)

    trainer = IncrementalTrainer(cfg, dm, backbone, model, stats_bank, style_bank, device)
    if args.refit_density_keys:
        trainer.refit_density_keys(int(state['task_id']))
    metrics = trainer.evaluate_seen_tasks(int(state['task_id']))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
