from __future__ import annotations

import json
import hashlib
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from dataset.feature_cache import load_or_extract
from tools.losses import (
    angular_margin_cross_entropy,
    prepare_robust_boundary_stats,
    robust_boundary_expansion_loss,
    supervised_contrastive_loss,
)
from tools.metrics import (
    classification_metrics,
    continual_forgetting,
    per_class_accuracy,
    per_task_accuracy,
    recurring_novel_harmonic,
)
from tools.stats import ClassStatsBank, class_means
from tools.transport import estimate_bridge_operator
from tools.checkpoint import save_checkpoint
from model.operator import DomainOperator
from model.classifier import ExpandableCosineClassifier


class IncrementalTrainer:
    def __init__(
        self,
        cfg,
        data_manager,
        backbone,
        model,
        stats_bank,
        style_bank,
        device,
        stage_history=None,
    ):
        self.cfg = cfg
        self.dm = data_manager
        self.backbone = backbone
        self.model = model.to(device)
        self.stats_bank = stats_bank
        self.style_bank = style_bank
        self.device = device
        self.output_dir = Path(cfg['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cfg.get('cache_dir', 'cache/features'))
        self.stage_history = self._normalize_stage_history(stage_history or [])
        # Compatibility with checkpoints produced before stage_history was saved.
        if not self.stage_history and len(self.model.operators) > 0:
            self.stage_history = self._load_legacy_jsonl_history()

    @staticmethod
    def _normalize_stage_history(history):
        normalized_by_task = {}
        for record in history:
            task_id = int(record['task_id'])
            per_task = {
                int(key): float(value)
                for key, value in record.get('per_task_accuracy', {}).items()
            }
            current_task_accuracy = record.get('current_task_accuracy')
            if current_task_accuracy is None:
                current_task_accuracy = per_task.get(task_id, 0.0)
            normalized_by_task[task_id] = {
                'task_id': task_id,
                'task_name': str(record.get('task_name', 'Task {}'.format(task_id))),
                'accuracy': float(record.get('accuracy', 0.0)),
                'current_task_accuracy': float(current_task_accuracy),
                'mean_class_accuracy': float(record.get('mean_class_accuracy', 0.0)),
                'router_accuracy': float(record.get('router_accuracy', 0.0)),
                'oracle_domain_accuracy': float(
                    record.get('oracle_domain_accuracy', 0.0)
                ),
                'router_per_task_accuracy': {
                    int(key): float(value)
                    for key, value in record.get(
                        'router_per_task_accuracy', {}
                    ).items()
                },
                'oracle_per_task_accuracy': {
                    int(key): float(value)
                    for key, value in record.get(
                        'oracle_per_task_accuracy', {}
                    ).items()
                },
                'per_task_accuracy': per_task,
            }
        return [normalized_by_task[key] for key in sorted(normalized_by_task)]

    def _load_legacy_jsonl_history(self):
        """Read an old metrics.jsonl only for resume compatibility."""
        legacy_path = self.output_dir / 'metrics.jsonl'
        if not legacy_path.exists():
            return []
        history = []
        with legacy_path.open('r', encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if int(record.get('task_id', -1)) < len(self.model.operators):
                    history.append(record)
        return self._normalize_stage_history(history)

    def _task_name(self, task_id):
        task_id = int(task_id)
        if 0 <= task_id < len(self.dm.tasks):
            return str(self.dm.tasks[task_id].name)
        return 'Task {}'.format(task_id)

    @staticmethod
    def _percent(value):
        return '{:.2f}%'.format(float(value) * 100.0)

    @classmethod
    def _percent_or_na(cls, value):
        return 'N/A' if value is None else cls._percent(value)

    def _write_metrics_txt(self, metrics):
        """Write one readable report; do not append dense JSONL records."""
        lines = [
            'Bridge-CDIL Training Metrics',
            '=' * 78,
            '',
            'Results After Each Incremental Task',
            '-' * 78,
        ]

        for record in self.stage_history:
            lines.extend([
                'Task {} - {}'.format(record['task_id'], record['task_name']),
                '  Current Task Accuracy: {}'.format(
                    self._percent(record['current_task_accuracy'])
                ),
                '  Cumulative Accuracy on All Seen Samples: {}'.format(
                    self._percent(record['accuracy'])
                ),
                '  Mean Class Accuracy: {}'.format(
                    self._percent(record['mean_class_accuracy'])
                ),
                '  Domain Router Accuracy: {}'.format(
                    self._percent(record['router_accuracy'])
                ),
                '  Known-Domain Oracle Accuracy: {}'.format(
                    self._percent(record['oracle_domain_accuracy'])
                ),
                '  Accuracy on Each Seen Task:',
            ])
            for evaluated_task_id, accuracy in record['per_task_accuracy'].items():
                lines.append(
                    '    Task {} - {}: {}'.format(
                        evaluated_task_id,
                        self._task_name(evaluated_task_id),
                        self._percent(accuracy),
                    )
                )
            if record['router_per_task_accuracy']:
                lines.append('  Router / Known-Domain Oracle by Task:')
                for evaluated_task_id, route_accuracy in (
                    record['router_per_task_accuracy'].items()
                ):
                    oracle_accuracy = record['oracle_per_task_accuracy'].get(
                        evaluated_task_id, 0.0
                    )
                    lines.append(
                        '    Task {} - {}: router={}, oracle={}'.format(
                            evaluated_task_id,
                            self._task_name(evaluated_task_id),
                            self._percent(route_accuracy),
                            self._percent(oracle_accuracy),
                        )
                    )
            lines.append('')

        lines.extend([
            'Final-Stage Accuracy on Each Task',
            '-' * 78,
        ])
        for evaluated_task_id, accuracy in metrics['per_task_accuracy'].items():
            lines.append(
                '  Task {} - {}: {}'.format(
                    evaluated_task_id,
                    self._task_name(evaluated_task_id),
                    self._percent(accuracy),
                )
            )

        lines.extend(['', 'Forgetting on Each Previous Task', '-' * 78])
        if metrics['per_task_forgetting']:
            for evaluated_task_id, forgetting in metrics['per_task_forgetting'].items():
                lines.append(
                    '  Task {} - {}: {}'.format(
                        evaluated_task_id,
                        self._task_name(evaluated_task_id),
                        self._percent(forgetting),
                    )
                )
        else:
            lines.append('  N/A (no previous task has been learned)')

        lines.extend([
            '',
            'Final Continual-Learning Summary',
            '=' * 78,
            'Last Accuracy (Final-Stage Accuracy Across All Seen Classes): {}'.format(
                self._percent(metrics['last_accuracy'])
            ),
            'Average Accuracy (Mean Cumulative Accuracy Across All Stages): {}'.format(
                self._percent(metrics['average_accuracy'])
            ),
            'Average Forgetting (Mean Accuracy Drop on Previous Tasks): {}'.format(
                self._percent(metrics['average_forgetting'])
            ),
            'Hrec (Recurring-Class Accuracy): {}'.format(
                self._percent_or_na(metrics['h_rec'])
            ),
            'Hnov (Novel-Class Accuracy): {}'.format(
                self._percent_or_na(metrics['h_nov'])
            ),
            'H-RN (Recurring/Novel Harmonic Mean): {}'.format(
                self._percent_or_na(metrics['h_rn'])
            ),
            'Recurring/Novel Test Samples: {}/{}'.format(
                metrics['recurring_count'], metrics['novel_count']
            ),
            'Cumulative Accuracy on All Seen Samples: {}'.format(
                self._percent(metrics['accuracy'])
            ),
            'Domain Router Accuracy: {}'.format(
                self._percent(metrics['router_accuracy'])
            ),
            'Known-Domain Oracle Accuracy: {}'.format(
                self._percent(metrics['oracle_domain_accuracy'])
            ),
            '',
        ])
        (self.output_dir / 'metrics.txt').write_text(
            '\n'.join(lines), encoding='utf-8'
        )

    def _print_final_summary(self, metrics):
        print('\n' + '=' * 90)
        print('BRIDGE-CDIL CONTINUAL-LEARNING SUMMARY')
        print('-' * 90)
        print('Final-stage accuracy on each task:')
        for evaluated_task_id, accuracy in metrics['per_task_accuracy'].items():
            print(
                '  Task {} - {}: {}'.format(
                    evaluated_task_id,
                    self._task_name(evaluated_task_id),
                    self._percent(accuracy),
                )
            )
        print('Forgetting on each previous task:')
        if metrics['per_task_forgetting']:
            for evaluated_task_id, forgetting in metrics['per_task_forgetting'].items():
                print(
                    '  Task {} - {}: {}'.format(
                        evaluated_task_id,
                        self._task_name(evaluated_task_id),
                        self._percent(forgetting),
                    )
                )
        else:
            print('  N/A (no previous task has been learned)')
        print('-' * 90)
        print(
            'Last Accuracy (Final-Stage Accuracy Across All Seen Classes): {}'.format(
                self._percent(metrics['last_accuracy'])
            )
        )
        print(
            'Average Accuracy (Mean Cumulative Accuracy Across All Stages): {}'.format(
                self._percent(metrics['average_accuracy'])
            )
        )
        print(
            'Average Forgetting (Mean Accuracy Drop on Previous Tasks): {}'.format(
                self._percent(metrics['average_forgetting'])
            )
        )
        print(
            'Hrec (Recurring-Class Accuracy): {}'.format(
                self._percent_or_na(metrics['h_rec'])
            )
        )
        print(
            'Hnov (Novel-Class Accuracy): {}'.format(
                self._percent_or_na(metrics['h_nov'])
            )
        )
        print(
            'H-RN (Recurring/Novel Harmonic Mean): {}'.format(
                self._percent_or_na(metrics['h_rn'])
            )
        )
        print(
            'Recurring/Novel Test Samples: {}/{}'.format(
                metrics['recurring_count'], metrics['novel_count']
            )
        )
        print(
            'Cumulative Accuracy on All Seen Samples: {}'.format(
                self._percent(metrics['accuracy'])
            )
        )
        print(
            'Domain Router Accuracy: {}'.format(
                self._percent(metrics['router_accuracy'])
            )
        )
        print(
            'Known-Domain Oracle Accuracy: {}'.format(
                self._percent(metrics['oracle_domain_accuracy'])
            )
        )
        print('Metrics text file: {}'.format(self.output_dir / 'metrics.txt'))
        print('=' * 90)

    def _cache_path(self, task_id, split, augmentation_view=None):
        task_name = self.dm.tasks[task_id].name
        backbone_cfg = self.cfg.get('backbone', {})
        backbone_type = str(
            backbone_cfg.get('type', 'vit_b16_in21k')
        ).strip().lower().replace('-', '_')
        section = (
            backbone_cfg.get('dinov3', {})
            if backbone_type.startswith('dinov3')
            else backbone_cfg.get('vit_b16_in21k', {})
        ) or {}
        model_name = str(
            section.get('model_name', backbone_cfg.get('model_name', 'unknown'))
        )
        weight_value = (
            section.get('checkpoint_path')
            or section.get('weights_path')
            or 'pretrained'
        )
        weight_name = Path(str(weight_value)).stem
        image_size = self.cfg.get('data', {}).get('image_size') or 'auto'
        raw_tag = '{}_{}_{}_{}'.format(
            backbone_type,
            model_name,
            weight_name,
            image_size,
        )
        cache_tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', raw_tag)
        split_tag = str(split)
        if augmentation_view is not None:
            split_tag = '{}_aug{:02d}'.format(split_tag, int(augmentation_view))
        return (
            self.cache_dir
            / cache_tag
            / f'{task_id:02d}_{task_name}_{split_tag}.pt'
        )

    def get_features(self, task_id, split='train', *, augmented=False, view_id=None):
        """Return cached frozen features, optionally for one training view.

        Evaluation and transport statistics always use the deterministic view.
        Augmented views are extracted once from current-task images and cached
        as ordinary labelled samples for supervised adapter/classifier fitting.
        The cache does not construct original/crop pairs and is not consumed by
        an image-instance contrastive objective, routing, or bridge statistics.
        """
        if augmented and split != 'train':
            raise ValueError('Augmented frozen-feature views are only defined for train.')
        if augmented and view_id is None:
            raise ValueError('An augmented training view requires view_id.')
        ds = self.dm.build_dataset(task_id, split, train_aug=bool(augmented))
        fc = self.cfg['feature_cache']
        backbone_cfg = self.cfg.get('backbone', {})
        backbone_type = str(
            backbone_cfg.get('type', 'vit_b16_in21k')
        ).strip().lower().replace('-', '_')
        selected_backbone_cfg = (
            backbone_cfg.get('dinov3', {})
            if backbone_type.startswith('dinov3')
            else backbone_cfg.get('vit_b16_in21k', {})
        ) or {}
        amp = bool(
            selected_backbone_cfg.get(
                'amp',
                backbone_cfg.get('amp', True),
            )
        )
        # Labels live in a mutable global registry. A cache made with an older
        # alias map/task order must never be silently reused with new labels.
        fingerprint = hashlib.sha256()
        for path, label, class_name in ds.samples:
            fingerprint.update(str(Path(path).resolve()).encode('utf-8'))
            fingerprint.update(b'\0')
            fingerprint.update(str(int(label)).encode('ascii'))
            fingerprint.update(b'\0')
            fingerprint.update(str(class_name).encode('utf-8'))
            fingerprint.update(b'\n')
        cache_metadata = {
            'version': 4,
            'task_id': int(task_id),
            'task_name': str(self.dm.tasks[task_id].name),
            'split': str(split),
            'augmented': bool(augmented),
            'view_id': None if view_id is None else int(view_id),
            'sample_count': len(ds),
            'sample_fingerprint': fingerprint.hexdigest(),
            'class_registry': {
                class_name: int(self.dm.registry.class_to_id[class_name])
                for class_name in sorted({item[2] for item in ds.samples})
            },
            'transform': repr(ds.transform),
            'backbone_type': backbone_type,
            'model_name': str(selected_backbone_cfg.get('model_name', 'unknown')),
            'weights_path': str(
                selected_backbone_cfg.get('checkpoint_path')
                or selected_backbone_cfg.get('weights_path')
                or 'pretrained'
            ),
            'amp': amp,
            'feature_pooling': str(
                getattr(self.backbone, 'feature_pooling', 'cls')
            ),
        }
        return load_or_extract(
            self.backbone,
            ds,
            self._cache_path(task_id, split, view_id if augmented else None),
            batch_size=int(fc.get('batch_size', 32)),
            num_workers=int(self.cfg['data'].get('num_workers', 8)),
            device=self.device,
            amp=amp,
            enabled=bool(fc.get('enabled', True)),
            overwrite=bool(fc.get('overwrite', False)),
            cache_metadata=cache_metadata,
            seed=(
                int(self.cfg.get('seed', 42))
                + 104729 * int(task_id)
                + (7919 * (int(view_id) + 1) if augmented else 0)
            ),
        )

    def train_style_router(
        self,
        task_id: int,
        current_styles: torch.Tensor,
        current_features: torch.Tensor,
    ):
        router_cfg = self.cfg.get('router', {}) or {}
        inference_mode = str(router_cfg.get('inference_mode', 'learned')).lower()
        density_mode = inference_mode in {
            'density', 'density_mixture', 'domain_keys', 'semantic_mixture'
        }
        key_space = str(router_cfg.get('key_space', 'raw')).strip().lower()
        self.style_bank.update(
            task_id,
            current_styles,
            current_features if density_mode else None,
            key_count=int(router_cfg.get('key_count', 128)),
            key_iterations=int(router_cfg.get('key_iterations', 15)),
            key_max_samples=int(router_cfg.get('key_max_samples', 16384)),
            seed=int(self.cfg.get('seed', 42)),
            device=self.device,
            feature_key_space=key_space,
        )
        self.model.router.expand(task_id + 1)
        if inference_mode != 'learned':
            if density_mode:
                key_count = len(self.style_bank.feature_keys[task_id])
                print(
                    f'  domain routing   : {key_count} {key_space}-space density keys; '
                    f'MLP skipped (inference_mode={inference_mode})'
                )
            else:
                print(f'  style router     : skipped (inference_mode={inference_mode})')
            return
        if task_id == 0:
            return

        # Use the same number of examples for every domain. Previously all
        # current styles (e.g. 25,200 for NWPU) competed against only 512
        # synthetic styles per old domain, making "latest domain" the optimum.
        max_per_domain = int(
            self.cfg['router'].get(
                'samples_per_domain',
                self.cfg['router'].get('synthetic_per_old_domain', 2048),
            )
        )
        n_per_domain = min(len(current_styles), max_per_domain)
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(self.cfg.get('seed', 42)) + 1009 * task_id)
        if len(current_styles) > n_per_domain:
            indices = torch.randperm(len(current_styles), generator=generator)[:n_per_domain]
            current_train_styles = current_styles[indices]
        else:
            current_train_styles = current_styles
        xs = [current_train_styles.float()]
        ys = [torch.full((len(current_train_styles),), task_id, dtype=torch.long)]
        for old_tid in range(task_id):
            xs.append(self.style_bank.sample(old_tid, n_per_domain, device='cpu').float())
            ys.append(torch.full((n_per_domain,), old_tid, dtype=torch.long))
        x = torch.cat(xs, 0)
        y = torch.cat(ys, 0)
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=int(self.cfg['router'].get('batch_size', 256)), shuffle=True)
        self.model.router.train()
        opt = torch.optim.AdamW(
            self.model.router.parameters(),
            lr=float(self.cfg['router'].get('lr', 1e-3)),
            weight_decay=float(self.cfg['router'].get('weight_decay', 1e-4)),
        )
        epochs = int(self.cfg['router'].get('epochs', 80))
        print(
            f'  router training: epochs={epochs}, samples={len(x)} '
            f'({n_per_domain}/domain), '
            f'batches_per_epoch={len(loader)}, total_updates={epochs * len(loader)}'
        )
        for epoch in range(epochs):
            total_loss = 0.0
            total_samples = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(self.model.router(xb), yb)
                loss.backward()
                opt.step()
                batch_size = int(yb.shape[0])
                total_loss += float(loss.detach().item()) * batch_size
                total_samples += batch_size

            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
                print(
                    f'    router epoch {epoch + 1:03d}/{epochs:03d}: '
                    f'loss={total_loss / max(total_samples, 1):.6f}'
                )
        self.model.router.eval()

    def refit_density_keys(self, max_task_id: int):
        """Refresh density-routing centres from cached training features only.

        It is useful when evaluating an older checkpoint whose stored centre
        budget differs from the selected configuration.  This does not train
        any module or consume test labels, and never writes the checkpoint.
        """
        router_cfg = self.cfg.get('router', {}) or {}
        inference_mode = str(router_cfg.get('inference_mode', 'learned')).lower()
        if inference_mode not in {'density', 'density_mixture', 'domain_keys', 'semantic_mixture'}:
            raise ValueError(
                'Density-key refitting requires a density routing inference mode.'
            )
        key_space = str(router_cfg.get('key_space', 'raw')).strip().lower()
        if key_space not in {'raw', 'native'}:
            raise ValueError('router.key_space must be raw or native.')
        self.style_bank.feature_keys = {}
        self.style_bank.feature_key_space = key_space
        for task_id in range(int(max_task_id) + 1):
            pack = self.get_features(task_id, 'train')
            features = pack['features']
            if inference_mode == 'semantic_mixture':
                self.style_bank.fit_class_keys(
                    task_id, features, pack['labels'],
                    keys_per_class=int(router_cfg.get('class_keys_per_class', 8)),
                    iterations=int(router_cfg.get('class_key_iterations', 10)),
                    max_samples=int(router_cfg.get('class_key_max_samples', 2048)),
                    seed=int(self.cfg.get('seed', 42)), device=self.device,
                )
            if key_space == 'native':
                with torch.no_grad():
                    features = self.model.adapt_known_domain(
                        features.to(self.device), task_id
                    ).cpu()
            self.style_bank.refit_feature_keys(
                task_id,
                features,
                key_count=int(router_cfg.get('key_count', 128)),
                key_iterations=int(router_cfg.get('key_iterations', 15)),
                key_max_samples=int(router_cfg.get('key_max_samples', 16384)),
                seed=int(self.cfg.get('seed', 42)),
                device=self.device,
            )

    def train_feature_adapter(self, task_info, raw_features, labels):
        if not self.model.feature_adapter_enabled:
            return
        task_id = int(task_info['task_id'])
        current_ids = list(task_info['current_ids'])
        adapter = self.model.feature_adapters[task_id]
        adapter.train()
        for parameter in adapter.parameters():
            parameter.requires_grad = True

        adapter_cfg = self.cfg.get('feature_adapter', {}) or {}
        x = raw_features.float().cpu()
        y = labels.long().cpu()
        counts = torch.bincount(
            y, minlength=len(self.dm.registry.id_to_class)
        ).float()
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(self.cfg.get('seed', 42)) + 4099 * task_id)
        sampler = WeightedRandomSampler(
            counts[y].reciprocal(),
            num_samples=len(y),
            replacement=True,
            generator=generator,
        )
        loader = DataLoader(
            TensorDataset(x, y),
            batch_size=int(adapter_cfg.get('batch_size', 256)),
            sampler=sampler,
        )

        local_lookup = torch.full(
            (max(current_ids) + 1,), -1, dtype=torch.long, device=self.device
        )
        local_lookup[torch.tensor(current_ids, device=self.device)] = torch.arange(
            len(current_ids), device=self.device
        )
        local_head = ExpandableCosineClassifier(
            self.model.classifier.dim,
            scale=float(
                adapter_cfg.get('classifier_scale', self.model.classifier.scale)
            ),
        ).to(self.device)
        with torch.no_grad():
            local_head.expand(
                len(current_ids),
                init_vectors=class_means(x, y, current_ids).to(self.device),
            )
        parameters = list(adapter.parameters()) + list(local_head.parameters())
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(adapter_cfg.get('lr', 1e-3)),
            weight_decay=float(adapter_cfg.get('weight_decay', 1e-4)),
        )
        epochs = int(adapter_cfg.get('epochs', 20))
        dropout = float(adapter_cfg.get('feature_dropout', 0.1))
        noise_std = float(adapter_cfg.get('noise_std', 0.01))
        contrastive_lambda = float(adapter_cfg.get('contrastive_lambda', 0.1))
        temperature = float(adapter_cfg.get('temperature', 0.1))
        angular_margin = float(adapter_cfg.get('angular_margin', 0.0))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=0.0
        )
        print(
            f'  feature adapter  : epochs={epochs}, samples={len(x)}, '
            f'batches_per_epoch={len(loader)}'
        )

        for epoch in range(epochs):
            total, total_ce, total_contrastive, samples = 0.0, 0.0, 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                targets = local_lookup[yb]
                # Match the clean feature distribution used at inference.
                # Applying dropout+noise to both branches had removed this
                # training signal and hurt first-task discriminability.
                view_a = xb
                view_b = F.dropout(xb, p=dropout, training=True)
                if noise_std > 0:
                    view_b = view_b + noise_std * torch.randn_like(view_b)
                features_a = adapter(view_a)
                features_b = adapter(view_b)
                loss_ce = (
                    0.75 * angular_margin_cross_entropy(
                        local_head(features_a),
                        targets,
                        scale=local_head.scale,
                        margin=angular_margin,
                    )
                    + 0.25 * angular_margin_cross_entropy(
                        local_head(features_b),
                        targets,
                        scale=local_head.scale,
                        margin=angular_margin,
                    )
                )
                loss_contrastive = supervised_contrastive_loss(
                    torch.stack([features_a, features_b], dim=1),
                    yb,
                    temperature=temperature,
                )
                loss = loss_ce + contrastive_lambda * loss_contrastive
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()

                batch_size = int(yb.shape[0])
                total += float(loss.detach()) * batch_size
                total_ce += float(loss_ce.detach()) * batch_size
                total_contrastive += float(loss_contrastive.detach()) * batch_size
                samples += batch_size
            scheduler.step()
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
                denominator = max(samples, 1)
                print(
                    f'    adapter epoch {epoch + 1:03d}/{epochs:03d}: '
                    f'loss={total / denominator:.6f}, '
                    f'ce={total_ce / denominator:.6f}, '
                    f'supcon={total_contrastive / denominator:.6f}'
                )
        adapter.eval()
        for parameter in adapter.parameters():
            parameter.requires_grad = False

    def estimate_operator(self, task_info, raw_features, labels):
        task_id = task_info['task_id']
        if task_id == 0 or not self.cfg['transport'].get('enabled', True):
            return DomainOperator(task_id=task_id)
        return estimate_bridge_operator(
            task_id=task_id,
            features=raw_features,
            labels=labels,
            bridge_ids=task_info['recurring_ids'],
            stats_bank=self.stats_bank,
            cfg=self.cfg['transport'],
            class_rank=int(self.cfg['statistics'].get('class_rank', 32)),
            seed=int(self.cfg.get('seed', 42)),
            device=self.device,
        )

    def _expand_classifier(self, canonical_features, labels, novel_ids):
        target_n = len(self.dm.registry.id_to_class)
        old_n = self.model.classifier.num_classes
        if target_n <= old_n:
            return
        expected_new = list(range(old_n, target_n))
        if expected_new != list(novel_ids):
            raise RuntimeError(
                f'Class registry is expected to append classes contiguously. '
                f'Expected new IDs {expected_new}, got {novel_ids}.'
            )
        init = class_means(canonical_features, labels, expected_new).to(self.device)
        self.model.classifier.expand(target_n, init_vectors=init)

    def train_classifier(self, task_info, canonical_features, labels):
        current_ids = task_info['current_ids']
        novel_ids = task_info['novel_ids']
        old_ids = task_info['old_ids_before_task']
        task_id = task_info['task_id']

        x = canonical_features.float().cpu()
        y = labels.long().cpu()
        old_only_ids = sorted(set(old_ids) - set(current_ids))
        recurring_ids = sorted(set(old_ids) & set(current_ids))
        replay_per_old_only_class = int(
            self.cfg['classifier'].get('replay_per_old_class', 0)
        )
        replay_per_bridge_class = int(
            self.cfg['classifier'].get(
                'bridge_replay_per_class', replay_per_old_only_class
            )
        )
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(self.cfg.get('seed', 42)) + 2027 * task_id)
        # Old-only classes need retention replay, while recurring bridge
        # classes need a larger historical contribution. Otherwise their many
        # current-domain samples pull the shared classifier away from the
        # canonical bridge consensus.
        old_only_x, old_only_y = self.stats_bank.sample(
            old_only_ids,
            replay_per_old_only_class,
            generator=generator,
        )
        bridge_x, bridge_y = self.stats_bank.sample(
            recurring_ids,
            replay_per_bridge_class,
            generator=generator,
        )
        replay_x = torch.cat([old_only_x, bridge_x], dim=0)
        replay_y = torch.cat([old_only_y, bridge_y], dim=0)
        if replay_y.numel():
            x = torch.cat([x, replay_x], dim=0)
            y = torch.cat([y, replay_y], dim=0)

        dataset = TensorDataset(x, y)
        class_balanced = bool(
            self.cfg['classifier'].get('class_balanced_sampling', True)
        )
        if class_balanced:
            counts = torch.bincount(
                y, minlength=self.model.classifier.num_classes
            ).float()
            sample_weights = counts[y].reciprocal()
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(y),
                replacement=True,
                generator=generator,
            )
            loader = DataLoader(
                dataset,
                batch_size=int(self.cfg['classifier'].get('batch_size', 256)),
                sampler=sampler,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=int(self.cfg['classifier'].get('batch_size', 256)),
                shuffle=True,
                generator=generator,
            )
        self.model.classifier.train()
        old_weight_reference = self.model.classifier.weight.detach().clone()
        opt = torch.optim.AdamW(
            [self.model.classifier.weight],
            lr=float(self.cfg['classifier'].get('lr', 3e-3)),
            weight_decay=float(self.cfg['classifier'].get('weight_decay', 1e-4)),
        )
        epochs = int(
            self.cfg['classifier'].get('epochs_first', 60)
            if task_id == 0
            else self.cfg['classifier'].get('epochs_incremental', 40)
        )
        robust_lambda = float(self.cfg['classifier'].get('robust_lambda', 0.5))
        anchor_lambda = float(
            self.cfg['classifier'].get('old_weight_anchor_lambda', 0.0)
        )
        freeze_absent = bool(self.cfg['classifier'].get('freeze_absent_old_classes', True))
        trainable_ids = sorted(y.unique().tolist())
        robust_stats = prepare_robust_boundary_stats(
            self.stats_bank,
            old_ids,
            device=self.device,
            dtype=self.model.classifier.weight.dtype,
        )

        print(
            f'  classifier training: epochs={epochs}, samples={len(x)} '
            f'(current={len(labels)}, old-only replay={len(old_only_y)}, '
            f'bridge replay={len(bridge_y)}), '
            f'batches_per_epoch={len(loader)}, total_updates={epochs * len(loader)}'
        )

        for epoch in range(epochs):
            total_loss = 0.0
            total_ce = 0.0
            total_rb = 0.0
            total_anchor = 0.0
            total_samples = 0

            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(set_to_none=True)
                logits = self.model.classifier(xb)
                loss_cls = F.cross_entropy(logits, yb)
                loss_rb = robust_boundary_expansion_loss(
                    self.model.classifier.weight[:self.model.classifier.num_classes],
                    self.stats_bank,
                    old_class_ids=old_ids,
                    new_class_ids=novel_ids,
                    margin=float(self.cfg['classifier'].get('robust_margin', 0.2)),
                    rho=float(self.cfg['classifier'].get('robust_rho', 1.0)),
                    prepared_stats=robust_stats,
                )
                if old_only_ids and anchor_lambda > 0:
                    current_old = F.normalize(
                        self.model.classifier.weight[old_only_ids], dim=-1
                    )
                    reference_old = F.normalize(
                        old_weight_reference[old_only_ids].to(current_old), dim=-1
                    )
                    loss_anchor = (1.0 - (current_old * reference_old).sum(-1)).mean()
                else:
                    loss_anchor = loss_cls.new_zeros(())
                loss = (
                    loss_cls
                    + robust_lambda * loss_rb
                    + anchor_lambda * loss_anchor
                )
                loss.backward()
                if freeze_absent:
                    self.model.classifier.mask_gradient(trainable_ids)
                opt.step()

                batch_size = int(yb.shape[0])
                total_loss += float(loss.detach().item()) * batch_size
                total_ce += float(loss_cls.detach().item()) * batch_size
                total_rb += float(loss_rb.detach().item()) * batch_size
                total_anchor += float(loss_anchor.detach().item()) * batch_size
                total_samples += batch_size

            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
                denom = max(total_samples, 1)
                print(
                    f'    classifier epoch {epoch + 1:03d}/{epochs:03d}: '
                    f'loss={total_loss / denom:.6f}, '
                    f'ce={total_ce / denom:.6f}, '
                    f'robust={total_rb / denom:.6f}, '
                    f'anchor={total_anchor / denom:.6f}'
                )
        self.model.classifier.eval()

    def _evidence_split_indices(self, task_info, labels, samples_per_view):
        """Image-level holdout shared by adapter, classifier and evidence fit."""
        y = labels.long().cpu()
        samples_per_view = int(samples_per_view)
        if samples_per_view <= 0 or len(y) % samples_per_view:
            raise ValueError('Evidence views must contain aligned sample blocks.')
        num_views = len(y) // samples_per_view
        base_y = y[:samples_per_view]
        for view in range(1, num_views):
            if not torch.equal(y[view * samples_per_view:(view + 1) * samples_per_view], base_y):
                raise RuntimeError('Evidence-view labels are not sample aligned.')
        fraction = float(self.cfg.get('domain_evidence', {}).get('calibration_fraction', 0.10))
        if not 0.0 <= fraction <= 0.5:
            raise ValueError('Evidence calibration_fraction must be in [0, 0.5].')
        generator = torch.Generator().manual_seed(
            int(self.cfg.get('seed', 42)) + 5051 * int(task_info['task_id'])
        )
        fit, calibration = [], []
        for class_id in task_info['current_ids']:
            indices = torch.where(base_y == int(class_id))[0]
            permutation = indices[torch.randperm(len(indices), generator=generator)]
            count = (
                min(max(round(len(indices) * fraction), 1), max(len(indices) - 1, 0))
                if fraction > 0 else 0
            )
            calibration.append(permutation[:count])
            for view in range(num_views):
                fit.append(permutation[count:] + view * samples_per_view)
        return torch.cat(fit), torch.cat(calibration)

    def train_domain_evidence_head(
        self,
        task_info,
        native_features,
        canonical_features,
        labels,
        samples_per_view: int | None = None,
    ):
        """Fit one frozen native-space residual evidence expert.

        The shared classifier remains the canonical decision maker.  This head
        preserves current-domain discriminability before bridge transport and
        only distinguishes labels that co-occurred in this domain. It needs no
        historical image or replay feature.
        """
        if not self.model.domain_evidence_enabled:
            return
        task_id = int(task_info['task_id'])
        current_ids = list(task_info['current_ids'])
        head = self.model.domain_evidence_heads[task_id]
        x_native = native_features.float().cpu()
        x_canonical = canonical_features.float().cpu()
        if x_native.shape != x_canonical.shape:
            raise ValueError('Native and canonical evidence features must align.')
        y_global = labels.long().cpu()
        lookup = torch.full(
            (max(current_ids) + 1,), -1, dtype=torch.long
        )
        lookup[torch.as_tensor(current_ids, dtype=torch.long)] = torch.arange(
            len(current_ids), dtype=torch.long
        )
        y = lookup[y_global]
        if bool((y < 0).any()):
            raise RuntimeError('Current-domain evidence head received an unseen label.')
        cfg = self.cfg.get('domain_evidence', {}) or {}
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(self.cfg.get('seed', 42)) + 5051 * task_id)
        if samples_per_view is None:
            samples_per_view = len(y)
        train_indices, calibration_indices = self._evidence_split_indices(
            task_info, y_global, samples_per_view
        )
        x_train, y_train = x_native[train_indices], y[train_indices]
        y_global_train = y_global[train_indices]

        def fit_diagonal_evidence(x_fit, y_local_fit, y_global_fit):
            with torch.no_grad():
                head.weight.copy_(
                    class_means(x_fit, y_global_fit, current_ids).to(head.weight)
                )
                # Estimate one pooled within-class diagonal covariance for this
                # domain.  Shrinkage prevents a nearly constant ViT dimension
                # from receiving an arbitrarily large discriminative weight.
                normalized_train = F.normalize(x_fit, dim=-1)
                normalized_means = F.normalize(head.weight.detach().cpu(), dim=-1)
                residual = normalized_train - normalized_means[y_local_fit]
                degrees_of_freedom = max(
                    len(x_fit) - len(current_ids), 1
                )
                variance = residual.square().sum(dim=0) / degrees_of_freedom
                shrinkage = float(cfg.get('variance_shrinkage', 0.20))
                shrinkage = min(max(shrinkage, 0.0), 1.0)
                average_variance = variance.mean().clamp_min(1e-8)
                variance = (
                    (1.0 - shrinkage) * variance
                    + shrinkage * average_variance
                ).clamp_min(1e-8)
                precision = variance.reciprocal()
                precision = precision / precision.mean().clamp_min(1e-8)
                precision_clip = max(float(cfg.get('precision_clip', 8.0)), 1.0)
                precision = precision.clamp(max=precision_clip)
                precision = precision / precision.mean().clamp_min(1e-8)
                head.precision.copy_(precision.to(head.precision))

        selected_shrinkage = float(cfg.get('variance_shrinkage', 0.20))
        if head.mode == 'shrinkage_lda':
            means, covariance = head.lda_statistics(x_train, y_train)
            shrinkages = cfg.get('covariance_shrinkage_candidates', [0.03, 0.10, 0.30])
            temperatures = cfg.get('temperature_candidates', [1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
            validation_x = x_native[calibration_indices].to(self.device)
            validation_y = y[calibration_indices].to(self.device)
            best = (float('inf'), selected_shrinkage, 1.0)
            for shrinkage in shrinkages:
                head.fit_lda_statistics(means, covariance, shrinkage)
                head.logit_temperature.fill_(1.0)
                with torch.no_grad():
                    logits = head(validation_x)
                    for temperature in temperatures:
                        if float(temperature) <= 0:
                            raise ValueError('Evidence temperatures must be positive.')
                        loss = float(F.cross_entropy(
                            logits / float(temperature), validation_y
                        ).item()) if len(validation_y) else 0.0
                        candidate = (loss, float(shrinkage), float(temperature))
                        if candidate < best:
                            best = candidate
            _, selected_shrinkage, selected_temperature = best
            head.fit_lda_statistics(means, covariance, selected_shrinkage)
            head.logit_temperature.fill_(selected_temperature)
            print(
                f'  native evidence : mode=shrinkage_lda, shrinkage={selected_shrinkage}, '
                f'temperature={selected_temperature}, calibration={len(validation_y)}'
            )
        elif head.mode == 'diagonal_lda':
            fit_diagonal_evidence(x_train, y_train, y_global_train)
        else:
            with torch.no_grad():
                head.weight.copy_(
                    class_means(x_train, y_global_train, current_ids).to(head.weight)
                )

        if head.mode in {'diagonal_lda', 'shrinkage_lda'}:
            head.eval()
            reported_shrinkage = (
                selected_shrinkage if head.mode == 'shrinkage_lda'
                else float(cfg.get('variance_shrinkage', 0.20))
            )
            print(
                f'  native evidence : mode={head.mode}, train={len(x_train)}, '
                f'calibration={len(calibration_indices)}, classes={len(current_ids)}, '
                f'shrinkage={reported_shrinkage:.2f}'
            )
        else:
            counts = torch.bincount(y_train, minlength=len(current_ids)).float()
            sampler = WeightedRandomSampler(
                counts[y_train].reciprocal(),
                num_samples=len(y_train),
                replacement=True,
                generator=generator,
            )
            loader = DataLoader(
                TensorDataset(x_train, y_train),
                batch_size=int(cfg.get('batch_size', 256)),
                sampler=sampler,
            )
            for parameter in head.parameters():
                parameter.requires_grad = True
            head.train()
            optimizer = torch.optim.AdamW(
                head.parameters(),
                lr=float(cfg.get('lr', 3e-3)),
                weight_decay=float(cfg.get('weight_decay', 1e-4)),
            )
            epochs = int(cfg.get('epochs', 20))
            label_smoothing = float(cfg.get('label_smoothing', 0.0))
            print(
                f'  native evidence : mode=cosine, epochs={epochs}, '
                f'train={len(x_train)}, calibration={len(calibration_indices)}, '
                f'classes={len(current_ids)}, batches_per_epoch={len(loader)}'
            )
            for epoch in range(epochs):
                total_loss, total_samples = 0.0, 0
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(
                        head(xb), yb, label_smoothing=label_smoothing
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(yb)
                    total_samples += len(yb)
                if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
                    print(
                        f'    evidence epoch {epoch + 1:03d}/{epochs:03d}: '
                        f'loss={total_loss / max(total_samples, 1):.6f}'
                    )
        head.eval()
        # Select this domain's residual-evidence strength using only held-out
        # current-task training features.  A zero weight is an allowed outcome,
        # so the auxiliary head cannot be forced to harm a strong shared head.
        if len(calibration_indices) == 0:
            head.blend_weight.zero_()
        else:
            with torch.no_grad():
                calibration_native = x_native[calibration_indices].to(self.device)
                calibration_canonical = x_canonical[calibration_indices].to(
                    self.device
                )
                calibration_y = y_global[calibration_indices].to(self.device)
                global_log_prob = F.log_softmax(
                    self.model.classifier(calibration_canonical), dim=-1
                )
                local_full = torch.full_like(global_log_prob, float('-inf'))
                local_full[:, head.class_ids] = F.log_softmax(
                    head(calibration_native), dim=-1
                )
                candidates = [
                    float(value)
                    for value in cfg.get(
                        'blend_candidates', [0.0, 0.10, 0.20, 0.35, 0.50, 0.65]
                    )
                    if 0.0 <= float(value) < 1.0
                ]
                if not candidates:
                    candidates = [0.0]
                calibration_metric = str(
                    cfg.get('calibration_metric', 'balanced_accuracy')
                ).strip().lower()
                best_weight, best_score, best_loss = 0.0, float('-inf'), float('inf')
                for candidate in sorted(set(candidates)):
                    if candidate == 0:
                        mixed_log_prob = global_log_prob
                    else:
                        mixed_log_prob = torch.logaddexp(
                            global_log_prob + math.log1p(-candidate),
                            local_full + math.log(candidate),
                        )
                    loss = float(F.nll_loss(mixed_log_prob, calibration_y).item())
                    if calibration_metric == 'nll':
                        score = -loss
                    elif calibration_metric == 'balanced_accuracy':
                        predictions = mixed_log_prob.argmax(dim=-1)
                        per_class = []
                        for class_id in head.class_ids:
                            mask = calibration_y == class_id
                            if bool(mask.any()):
                                per_class.append(
                                    (predictions[mask] == calibration_y[mask])
                                    .float()
                                    .mean()
                                )
                        score = float(torch.stack(per_class).mean().item())
                    else:
                        raise ValueError(
                            'domain_evidence.calibration_metric must be '
                            'balanced_accuracy or nll.'
                        )
                    if (
                        score > best_score + 1e-12
                        or (abs(score - best_score) <= 1e-12 and loss < best_loss)
                    ):
                        best_weight, best_score, best_loss = candidate, score, loss
                head.blend_weight.fill_(best_weight)
        if head.mode == 'diagonal_lda':
            # Calibration selected the mixture strength out of sample. Refit
            # the analytic sufficient statistics on all current-domain train
            # samples afterwards so the deployed head does not discard 10% of
            # the available data. No test sample or historical exemplar enters.
            fit_diagonal_evidence(x_native, y, y_global)
        elif head.mode == 'shrinkage_lda':
            means, covariance = head.lda_statistics(x_native, y)
            head.fit_lda_statistics(means, covariance, selected_shrinkage)
        print(f'  evidence blend  : weight={float(head.blend_weight):.2f}')
        for parameter in head.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def evaluate_seen_tasks(self, max_task_id: int):
        self.model.eval()
        all_pred, all_oracle_pred, all_route, all_y, all_tid = [], [], [], [], []
        router_correct = 0
        router_total = 0
        oracle_correct = 0
        for tid in range(max_task_id + 1):
            pack = self.get_features(tid, 'test')
            f = pack['features']
            s = pack['styles']
            y = pack['labels']
            task_ids = pack['task_ids']
            bs = int(self.cfg['eval'].get('batch_size', 512))
            for st in range(0, len(y), bs):
                ed = min(st + bs, len(y))
                fb = f[st:ed].to(self.device)
                sb = s[st:ed].to(self.device)
                inference_mode = str(
                    self.cfg.get('router', {}).get('inference_mode', 'learned')
                ).lower()
                if inference_mode in {'density', 'density_mixture', 'domain_keys', 'semantic_mixture'}:
                    router_cfg = self.cfg.get('router', {}) or {}
                    native_candidates = None
                    routing_features = fb
                    if self.style_bank.feature_key_space == 'native':
                        native_candidates = self.model.native_candidate_features(fb)
                        routing_features = native_candidates
                    alpha = self.style_bank.domain_probabilities(
                        routing_features,
                        num_domains=len(self.model.operators),
                        temperature=float(
                            router_cfg.get('key_temperature', 0.02)
                        ),
                        neighbors=int(router_cfg.get('key_neighbors', 1)),
                        hard=bool(router_cfg.get('key_hard_inference', False)),
                    )
                    logits, alpha = self.model.forward_domain_mixture(
                        fb, alpha, native_candidates=native_candidates,
                        class_domain_scores=(
                            self.style_bank.class_domain_scores(
                                fb, len(self.model.operators), self.model.classifier.num_classes,
                                neighbors=int(router_cfg.get('class_key_neighbors', 2)),
                            ) if inference_mode == 'semantic_mixture' else None
                        ),
                    )
                else:
                    logits, alpha = self.model.forward_auto(fb, sb)
                pred = logits.argmax(-1).cpu()
                oracle_pred = self.model.forward_known_domain(fb, tid).argmax(-1).cpu()
                route = alpha.argmax(-1).cpu()
                all_pred.append(pred)
                all_oracle_pred.append(oracle_pred)
                all_route.append(route)
                all_y.append(y[st:ed])
                all_tid.append(task_ids[st:ed])
                router_correct += int((route == task_ids[st:ed]).sum().item())
                oracle_correct += int((oracle_pred == y[st:ed]).sum().item())
                router_total += int(ed - st)

        pred = torch.cat(all_pred)
        oracle_pred = torch.cat(all_oracle_pred)
        routes = torch.cat(all_route)
        y = torch.cat(all_y)
        tids = torch.cat(all_tid)
        overall = classification_metrics(pred, y)
        task_acc = per_task_accuracy(pred, y, tids)
        router_task_acc = per_task_accuracy(routes, tids, tids)
        oracle_task_acc = per_task_accuracy(oracle_pred, y, tids)
        class_acc = per_class_accuracy(pred, y)
        task_values = list(task_acc.values())
        prev_values = [v for k, v in task_acc.items() if k < max_task_id]
        rn_metrics = recurring_novel_harmonic(
            pred,
            y,
            tids,
            self.dm.recurring_novel_class_groups(max_task_id),
        )
        return {
            'accuracy': overall['accuracy'],
            'correct': overall['correct'],
            'total': overall['total'],
            'average_task_accuracy': sum(task_values) / max(len(task_values), 1),
            'current_task_accuracy': task_acc.get(max_task_id, 0.0),
            'previous_task_accuracy': sum(prev_values) / max(len(prev_values), 1) if prev_values else None,
            'per_task_accuracy': task_acc,
            'mean_class_accuracy': sum(class_acc.values()) / max(len(class_acc), 1),
            'per_class_accuracy': class_acc,
            'router_accuracy': router_correct / max(router_total, 1),
            'oracle_domain_accuracy': oracle_correct / max(router_total, 1),
            'router_per_task_accuracy': router_task_acc,
            'oracle_per_task_accuracy': oracle_task_acc,
            **rn_metrics,
        }

    def run_task(self, task_id: int):
        info = self.dm.prepare_task(task_id)
        print('\n' + '=' * 90)
        print(f"Task {task_id}: {info['task_name']}")
        print(f"  current classes : {len(info['current_ids'])}")
        print(f"  recurring/bridge: {len(info['recurring_ids'])}")
        print(f"  novel classes   : {len(info['novel_ids'])}")
        print(f"  total seen      : {len(self.dm.registry.id_to_class)}")

        train_pack = self.get_features(task_id, 'train')
        raw = train_pack['features'].float()
        styles = train_pack['styles'].float()
        labels = train_pack['labels'].long()

        training_feature_views = [raw]
        training_label_views = [labels]
        augmentation_views = int(
            self.cfg.get('feature_cache', {}).get('augmentation_views', 0)
        )
        for view_id in range(max(augmentation_views, 0)):
            augmented_pack = self.get_features(
                task_id, 'train', augmented=True, view_id=view_id
            )
            if not torch.equal(augmented_pack['labels'].long(), labels):
                raise RuntimeError(
                    'Augmented feature cache labels do not match the base training view.'
                )
            if augmented_pack['paths'] != train_pack['paths']:
                raise RuntimeError(
                    'Augmented feature cache order does not match the base training view.'
                )
            training_feature_views.append(augmented_pack['features'].float())
            training_label_views.append(augmented_pack['labels'].long())
        training_raw = torch.cat(training_feature_views, dim=0)
        training_labels = torch.cat(training_label_views, dim=0)
        if self.model.domain_evidence_enabled:
            fit_indices, calibration_indices = self._evidence_split_indices(
                info, training_labels, len(labels)
            )
            base_fit_indices = fit_indices[fit_indices < len(labels)]
            print(f'  image holdout    : {len(calibration_indices)} current-train images')
        else:
            fit_indices = torch.arange(len(training_labels))
            base_fit_indices = torch.arange(len(labels))
        if augmentation_views:
            print(
                f'  frozen augmentations: {augmentation_views} cached view(s), '
                f'training samples={len(training_raw)}'
            )

        self.model.add_feature_adapter(task_id)
        self.train_feature_adapter(
            info, training_raw[fit_indices], training_labels[fit_indices]
        )
        with torch.no_grad():
            native_training_views = [
                self.model.adapt_known_domain(
                    view.to(self.device), task_id
                ).cpu()
                for view in training_feature_views
            ]
            adapted = native_training_views[0]
            native_training = torch.cat(native_training_views, dim=0)

        op = self.estimate_operator(
            info, adapted[base_fit_indices], labels[base_fit_indices]
        )
        self.model.add_operator(op)
        print(f'  domain operator : rank={0 if op.U is None else op.U.shape[1]}, bridges={op.bridge_count}')
        if op.bw_before is not None:
            print(f'  BW covariance   : {op.bw_before:.5f} -> {op.bw_after:.5f}')
        if op.mean_before is not None:
            print(
                f'  bridge geometry : {op.mean_before:.5f} -> '
                f'{op.mean_after:.5f}'
            )

        router_key_space = str(
            (self.cfg.get('router', {}) or {}).get('key_space', 'raw')
        ).strip().lower()
        self.train_style_router(
            task_id,
            styles,
            adapted if router_key_space == 'native' else raw,
        )
        if str(self.cfg.get('router', {}).get('inference_mode', '')).lower() == 'semantic_mixture':
            router_cfg = self.cfg['router']
            self.style_bank.fit_class_keys(
                task_id, raw, labels,
                keys_per_class=int(router_cfg.get('class_keys_per_class', 8)),
                iterations=int(router_cfg.get('class_key_iterations', 10)),
                max_samples=int(router_cfg.get('class_key_max_samples', 2048)),
                seed=int(self.cfg.get('seed', 42)), device=self.device,
            )

        with torch.no_grad():
            canonical_training_views = [
                F.normalize(
                    op.apply(native_view.to(self.device)), dim=-1
                ).cpu()
                for native_view in native_training_views
            ]
            canonical = canonical_training_views[0]
            canonical_training = torch.cat(canonical_training_views, dim=0)

        self._expand_classifier(
            canonical[base_fit_indices], labels[base_fit_indices], info['novel_ids']
        )
        self.train_classifier(
            info, canonical_training[fit_indices], training_labels[fit_indices]
        )
        self.model.add_domain_evidence_head(task_id, info['current_ids'])
        self.train_domain_evidence_head(
            info,
            native_training,
            canonical_training,
            training_labels,
            samples_per_view=len(labels),
        )

        # Historical knowledge is updated only with compact canonical sufficient statistics.
        self.stats_bank.update_from_features(canonical, labels, info['current_ids'])

        metrics = self.evaluate_seen_tasks(task_id)

        previous_records = [
            record for record in self.stage_history
            if record['task_id'] < task_id
        ]
        stage_accuracies = [
            record['accuracy'] for record in previous_records
        ] + [metrics['accuracy']]
        task_accuracy_history = [
            record['per_task_accuracy'] for record in previous_records
        ] + [metrics['per_task_accuracy']]
        average_forgetting, per_task_forgetting = continual_forgetting(
            task_accuracy_history,
            current_task_id=task_id,
        )

        # DSPCL paper definitions:
        # Last = cumulative accuracy over all seen classes after the final stage.
        # Average = mean cumulative accuracy over all incremental stages.
        metrics['last_accuracy'] = metrics['accuracy']
        metrics['average_accuracy'] = (
            sum(stage_accuracies) / len(stage_accuracies)
        )
        metrics['incremental_average_accuracy'] = metrics['average_accuracy']
        metrics['average_forgetting'] = average_forgetting
        metrics['forgetting'] = average_forgetting
        metrics['per_task_forgetting'] = per_task_forgetting

        compact_record = {
            'task_id': task_id,
            'task_name': info['task_name'],
            'accuracy': metrics['accuracy'],
            'current_task_accuracy': metrics['current_task_accuracy'],
            'mean_class_accuracy': metrics['mean_class_accuracy'],
            'router_accuracy': metrics['router_accuracy'],
            'oracle_domain_accuracy': metrics['oracle_domain_accuracy'],
            'router_per_task_accuracy': dict(
                metrics['router_per_task_accuracy']
            ),
            'oracle_per_task_accuracy': dict(
                metrics['oracle_per_task_accuracy']
            ),
            'per_task_accuracy': dict(metrics['per_task_accuracy']),
        }
        self.stage_history = previous_records + [compact_record]
        metrics['stage_history'] = self.stage_history

        print(
            '  evaluation complete: {}/{} correct'.format(
                metrics['correct'], metrics['total']
            )
        )

        self.dm.registry.save_json(self.output_dir / 'class_registry.json')
        ckpt = self.output_dir / f"task_{task_id:02d}_{info['task_name']}.pt"
        save_checkpoint(
            ckpt,
            task_id=task_id,
            model=self.model,
            stats_bank=self.stats_bank,
            style_bank=self.style_bank,
            registry=self.dm.registry,
            cfg=self.cfg,
            metrics=metrics,
        )
        print(f'  checkpoint         : {ckpt}')
        self._write_metrics_txt(metrics)
        self._print_final_summary(metrics)
        return metrics
