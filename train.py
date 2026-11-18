from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--resume', default=None, help='Optional checkpoint; training continues from next task.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get('seed', 42)))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    dm = IncrementalDataManager(cfg)
    backbone = FrozenDINOv3(cfg['backbone']).to(device)
    model = BridgeCDILModel(backbone.feature_dim, backbone.style_dim, cfg).to(device)
    stats_bank = ClassStatsBank(
        rank=int(cfg['statistics'].get('class_rank', 32)),
        eps=float(cfg['statistics'].get('min_eigenvalue', 1e-5)),
    )
    style_bank = DomainStyleBank()

    start_task = 0
    stage_history = []
    if args.resume:
        state = load_model_state(args.resume, model, stats_bank, style_bank, dm.registry, map_location='cpu')
        model.to(device)
        start_task = int(state['task_id']) + 1
        stage_history = state.get('metrics', {}).get('stage_history', [])
        print(f'Resumed from task {state["task_id"]}; next task = {start_task}')

    trainer = IncrementalTrainer(cfg, dm, backbone, model, stats_bank, style_bank, device, stage_history=stage_history)
    for task_id in range(start_task, dm.num_tasks()):
        trainer.run_task(task_id)


if __name__ == '__main__':
    main()
