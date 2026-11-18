from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleRouter(nn.Module):
    """Shared style-conditioned router over compact domain operators."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_domains: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.body = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, num_domains)

    @property
    def num_domains(self):
        return self.head.out_features

    def expand(self, new_num_domains: int):
        if new_num_domains <= self.num_domains:
            return
        old = self.head
        new = nn.Linear(self.hidden_dim, new_num_domains).to(old.weight.device)
        nn.init.normal_(new.weight, std=0.02)
        nn.init.zeros_(new.bias)
        with torch.no_grad():
            new.weight[:old.out_features].copy_(old.weight)
            new.bias[:old.out_features].copy_(old.bias)
        self.head = new

    def forward(self, style):
        return self.head(self.body(style))


class DomainStyleBank:
    """Compact domain statistics and frozen-feature density keys.

    Style moments are retained for the legacy learned-router ablation.  The
    feature keys are spherical k-means centres in the frozen backbone space.
    Labelled class keys add semantic evidence for joint class/domain inference.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.stats: Dict[int, Dict[str, torch.Tensor | int]] = {}
        self.feature_keys: Dict[int, torch.Tensor] = {}
        self.feature_key_space = 'raw'
        self.class_keys: Dict[int, Dict[int, torch.Tensor]] = {}

    @torch.no_grad()
    def fit_class_keys(
        self, task_id, features, labels, *, keys_per_class=8,
        iterations=10, max_samples=2048, seed=42, device='cpu',
    ):
        """Keep labelled centres, never training images, in the common raw space."""
        x = features.detach().float().cpu()
        y = labels.detach().long().cpu()
        if len(x) != len(y) or x.ndim != 2:
            raise ValueError('Semantic keys require aligned raw features and labels.')
        self.class_keys[int(task_id)] = {
            int(class_id): self._fit_feature_keys(
                x[y == class_id], keys_per_class, iterations, max_samples,
                int(seed) + 1009 * int(task_id) + 9176 * int(class_id), device,
            )
            for class_id in y.unique(sorted=True).tolist()
        }

    def class_domain_scores(self, features, num_domains, num_classes, neighbors=2):
        """Score every candidate (domain, class); no query label/ID is accepted."""
        query = F.normalize(features.float(), dim=-1)
        if query.ndim != 2:
            raise ValueError('Semantic routing requires raw [batch, features].')
        scores = query.new_full((len(query), num_domains, num_classes), float('-inf'))
        for domain_id in range(num_domains):
            if domain_id not in self.class_keys:
                raise RuntimeError('Semantic routing keys missing; train from task 0.')
            for class_id, keys in self.class_keys[domain_id].items():
                keys = keys.to(query)
                k = min(max(int(neighbors), 1), len(keys))
                scores[:, domain_id, class_id] = (
                    (query @ keys.T).topk(k, dim=-1).values.mean(-1)
                )
        if not bool(torch.isfinite(scores).any(dim=1).all()):
            raise RuntimeError('A registered class has no semantic routing keys.')
        return scores

    @staticmethod
    @torch.no_grad()
    def _fit_feature_keys(
        features: torch.Tensor,
        key_count: int,
        iterations: int,
        max_samples: int,
        seed: int,
        device,
    ):
        if features.ndim != 2 or len(features) == 0:
            raise ValueError('Domain-key fitting requires a non-empty [N, D] tensor.')
        x_cpu = F.normalize(features.detach().float().cpu(), dim=-1)
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        max_samples = max(int(max_samples), 1)
        if len(x_cpu) > max_samples:
            indices = torch.randperm(len(x_cpu), generator=generator)[:max_samples]
            x_cpu = x_cpu[indices]
        x = x_cpu.to(device)
        key_count = min(max(int(key_count), 1), len(x))

        # Farthest-point initialization covers minority modes better than a
        # purely random subset and remains deterministic for reproducibility.
        first = int(torch.randint(len(x), (1,), generator=generator).item())
        selected = [first]
        min_distance = 1.0 - x @ x[first]
        for _ in range(1, key_count):
            index = int(min_distance.argmax().item())
            selected.append(index)
            min_distance = torch.minimum(min_distance, 1.0 - x @ x[index])
        keys = x[selected].clone()

        for _ in range(max(int(iterations), 1)):
            assignment = (x @ keys.T).argmax(dim=-1)
            sums = torch.zeros_like(keys)
            sums.index_add_(0, assignment, x)
            counts = torch.bincount(assignment, minlength=key_count).to(x.dtype)
            nonempty = counts > 0
            updated = keys.clone()
            updated[nonempty] = sums[nonempty] / counts[nonempty, None]
            updated = F.normalize(updated, dim=-1)
            if torch.allclose(updated, keys, atol=1e-5, rtol=0.0):
                keys = updated
                break
            keys = updated
        return keys.float().cpu()

    def update(
        self,
        task_id: int,
        styles: torch.Tensor,
        features: torch.Tensor | None = None,
        *,
        key_count: int = 128,
        key_iterations: int = 15,
        key_max_samples: int = 16384,
        seed: int = 42,
        device='cpu',
        feature_key_space: str = 'raw',
    ):
        styles = styles.float().cpu()
        task_id = int(task_id)
        self.stats[task_id] = {
            'mean': styles.mean(0),
            'var': styles.var(0, unbiased=False).clamp_min(self.eps),
            'count': int(styles.shape[0]),
        }
        if features is not None:
            key_space = str(feature_key_space).strip().lower()
            if key_space not in {'raw', 'native'}:
                raise ValueError('feature_key_space must be raw or native.')
            if self.feature_keys and key_space != self.feature_key_space:
                raise RuntimeError(
                    'Cannot mix raw and native density keys in one style bank.'
                )
            self.feature_key_space = key_space
            self.refit_feature_keys(
                task_id,
                features,
                key_count=key_count,
                key_iterations=key_iterations,
                key_max_samples=key_max_samples,
                seed=seed,
                device=device,
            )

    @torch.no_grad()
    def refit_feature_keys(
        self,
        task_id: int,
        features: torch.Tensor,
        *,
        key_count: int = 128,
        key_iterations: int = 15,
        key_max_samples: int = 16384,
        seed: int = 42,
        device='cpu',
    ):
        """Rebuild only the compact density ledger from current-task features.

        This is intentionally separate from :meth:`update`: checkpoint
        evaluation can upgrade the number of stored centres from the cached
        *training* features without altering style statistics, classifier
        weights, transports, or looking at test labels.
        """
        task_id = int(task_id)
        self.feature_keys[task_id] = self._fit_feature_keys(
            features,
            key_count=key_count,
            iterations=key_iterations,
            max_samples=key_max_samples,
            seed=int(seed) + 1009 * task_id,
            device=device,
        )

    def domain_scores(
        self,
        features: torch.Tensor,
        num_domains: int | None = None,
        neighbors: int = 1,
    ):
        if num_domains is None:
            num_domains = len(self.feature_keys)
        missing = [
            domain_id for domain_id in range(int(num_domains))
            if domain_id not in self.feature_keys
        ]
        if missing:
            raise RuntimeError(
                'Frozen-feature domain keys are missing for domains {}. '
                'This checkpoint predates density routing; rerun from task 0 or '
                'select the legacy learned/classwise_max inference mode.'.format(missing)
            )
        if self.feature_key_space == 'native':
            if features.ndim != 3 or features.shape[1] != int(num_domains):
                raise ValueError(
                    'Native density routing requires [batch, domains, features].'
                )
            query = F.normalize(features.float(), dim=-1)
        else:
            if features.ndim != 2:
                raise ValueError('Raw density routing requires [batch, features].')
            query = F.normalize(features.float(), dim=-1)
        neighbors = max(int(neighbors), 1)
        scores = []
        for domain_id in range(int(num_domains)):
            keys = self.feature_keys[domain_id].to(query.device, query.dtype)
            local_neighbors = min(neighbors, len(keys))
            domain_query = (
                query[:, domain_id]
                if self.feature_key_space == 'native'
                else query
            )
            scores.append(
                (domain_query @ keys.T)
                .topk(local_neighbors, dim=-1)
                .values.mean(dim=-1)
            )
        return torch.stack(scores, dim=-1)

    def domain_probabilities(
        self,
        features: torch.Tensor,
        num_domains: int | None = None,
        temperature: float = 0.02,
        neighbors: int = 1,
        hard: bool = False,
    ):
        temperature = float(temperature)
        if temperature <= 0:
            raise ValueError('Domain-key temperature must be positive.')
        scores = self.domain_scores(
            features,
            num_domains=num_domains,
            neighbors=neighbors,
        )
        probabilities = F.softmax(scores / temperature, dim=-1)
        if hard:
            route = probabilities.argmax(dim=-1)
            probabilities = F.one_hot(
                route, num_classes=probabilities.shape[-1]
            ).to(probabilities.dtype)
        return probabilities

    def sample(self, task_id: int, n: int, device='cpu'):
        st = self.stats[int(task_id)]
        mean = st['mean'].to(device)
        std = st['var'].sqrt().to(device)
        return mean + torch.randn(n, mean.numel(), device=device) * std

    def state_dict(self):
        return {
            'eps': self.eps,
            'stats': {
                int(k): {
                    'mean': v['mean'].cpu(),
                    'var': v['var'].cpu(),
                    'count': int(v['count']),
                }
                for k, v in self.stats.items()
            },
            'feature_keys': {
                int(k): value.float().cpu()
                for k, value in self.feature_keys.items()
            },
            'feature_key_space': self.feature_key_space,
            'class_keys': {
                int(d): {int(c): keys.cpu() for c, keys in classes.items()}
                for d, classes in self.class_keys.items()
            },
        }

    def load_state_dict(self, state):
        self.eps = float(state.get('eps', 1e-6))
        self.stats = {int(k): v for k, v in state['stats'].items()}
        self.feature_keys = {
            int(k): value.float().cpu()
            for k, value in state.get('feature_keys', {}).items()
        }
        self.feature_key_space = str(
            state.get('feature_key_space', 'raw')
        ).strip().lower()
        self.class_keys = {
            int(d): {int(c): keys.float().cpu() for c, keys in classes.items()}
            for d, classes in state.get('class_keys', {}).items()
        }
