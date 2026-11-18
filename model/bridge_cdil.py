from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import DomainEvidenceHead, ExpandableCosineClassifier
from .feature_adapter import ResidualFeatureAdapter
from .operator import DomainOperator
from .style_router import StyleRouter


class BridgeCDILModel(nn.Module):
    def __init__(self, feature_dim: int, style_dim: int, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.classifier = ExpandableCosineClassifier(
            dim=feature_dim,
            num_classes=0,
            scale=cfg['classifier'].get('scale', 20.0),
        )
        self.router = StyleRouter(
            input_dim=style_dim,
            hidden_dim=cfg['router'].get('hidden_dim', 256),
            num_domains=1,
        )
        adapter_cfg = cfg.get('feature_adapter', {}) or {}
        self.feature_adapter_enabled = bool(adapter_cfg.get('enabled', False))
        self.feature_adapter_dim = int(adapter_cfg.get('bottleneck_dim', 128))
        self.feature_adapter_stages = int(adapter_cfg.get('stages', 1))
        evidence_cfg = cfg.get('domain_evidence', {}) or {}
        self.domain_evidence_mode = str(evidence_cfg.get('mode', 'cosine')).lower()
        self.domain_evidence_hidden_dim = int(evidence_cfg.get('hidden_dim', 0))
        self.domain_evidence_dropout = float(evidence_cfg.get('dropout', 0.0))
        self.feature_adapters = nn.ModuleList()
        self.domain_evidence_heads = nn.ModuleList()
        self.operators: list[DomainOperator] = []

    def add_feature_adapter(self, task_id: int):
        if not self.feature_adapter_enabled:
            return
        if int(task_id) != len(self.feature_adapters):
            raise ValueError(
                f'Feature adapters must be appended in task order. Got task {task_id}.'
            )
        self.feature_adapters.append(
            ResidualFeatureAdapter(
                self.classifier.dim,
                bottleneck_dim=self.feature_adapter_dim,
                stages=self.feature_adapter_stages,
            ).to(self.classifier.weight.device)
        )

    def ensure_feature_adapters(self, num_domains: int):
        while self.feature_adapter_enabled and len(self.feature_adapters) < num_domains:
            self.add_feature_adapter(len(self.feature_adapters))

    @property
    def domain_evidence_enabled(self):
        cfg = self.cfg.get('domain_evidence', {}) or {}
        return bool(cfg.get('enabled', False))

    def _append_domain_evidence_head(self, task_id: int, class_ids):
        if int(task_id) != len(self.domain_evidence_heads):
            raise ValueError(
                'Domain evidence heads must be appended in task order. '
                f'Got task {task_id}.'
            )
        self.domain_evidence_heads.append(
            DomainEvidenceHead(
                self.classifier.dim,
                class_ids,
                scale=float(
                    (self.cfg.get('domain_evidence', {}) or {}).get(
                        'scale', self.classifier.scale
                    )
                ),
                blend_weight=float(
                    (self.cfg.get('domain_evidence', {}) or {}).get(
                        'blend_weight', 0.0
                    )
                ),
                hidden_dim=self.domain_evidence_hidden_dim,
                dropout=self.domain_evidence_dropout,
                mode=self.domain_evidence_mode,
            ).to(self.classifier.weight.device)
        )

    def set_domain_evidence_arch(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
        mode: str = 'cosine',
    ):
        """Select checkpointed evidence architecture before heads are created."""
        if len(self.domain_evidence_heads):
            raise RuntimeError('Evidence architecture cannot change after head creation.')
        self.domain_evidence_hidden_dim = int(hidden_dim)
        self.domain_evidence_dropout = float(dropout)
        self.domain_evidence_mode = str(mode).strip().lower()

    def add_domain_evidence_head(self, task_id: int, class_ids):
        if not self.domain_evidence_enabled:
            return
        self._append_domain_evidence_head(task_id, class_ids)

    def ensure_domain_evidence_heads(self, domain_class_ids):
        """Materialize checkpointed heads before strict state loading."""
        for task_id, class_ids in enumerate(domain_class_ids or []):
            if task_id < len(self.domain_evidence_heads):
                continue
            # A checkpoint may be inspected with an ablation configuration that
            # sets domain_evidence.enabled=false.  The parameters must still be
            # materialized for strict loading, even though inference ignores
            # their residual contribution.
            self._append_domain_evidence_head(task_id, class_ids)

    def domain_evidence_class_ids(self):
        return [head.class_ids.detach().cpu().tolist() for head in self.domain_evidence_heads]

    def freeze_domain_evidence_heads(self):
        for head in self.domain_evidence_heads:
            head.eval()
            for parameter in head.parameters():
                parameter.requires_grad = False

    def freeze_feature_adapters(self):
        for adapter in self.feature_adapters:
            adapter.eval()
            for parameter in adapter.parameters():
                parameter.requires_grad = False

    def adapt_known_domain(self, features: torch.Tensor, task_id: int):
        if not self.feature_adapter_enabled:
            return F.normalize(features, dim=-1)
        return self.feature_adapters[int(task_id)](features)

    def native_candidate_features(self, features: torch.Tensor):
        """Encode a sample in every frozen domain-native feature space."""
        return torch.stack(
            [
                self.adapt_known_domain(features, task_id)
                for task_id in range(len(self.operators))
            ],
            dim=1,
        )

    def add_operator(self, op: DomainOperator):
        if op.task_id != len(self.operators):
            raise ValueError(f'Operators must be appended in task order. Got task {op.task_id}.')
        self.operators.append(op)
        self.router.expand(len(self.operators))

    def canonicalize_known_domain(self, features: torch.Tensor, task_id: int):
        adapted = self.adapt_known_domain(features, task_id)
        return F.normalize(self.operators[task_id].apply(adapted), dim=-1)

    def candidate_features(
        self,
        features: torch.Tensor,
        native_candidates: torch.Tensor | None = None,
    ):
        if native_candidates is None:
            native_candidates = self.native_candidate_features(features)
        expected = (features.shape[0], len(self.operators), self.classifier.dim)
        if native_candidates.shape != expected:
            raise ValueError(
                f'native_candidates must have shape {expected}, '
                f'got {tuple(native_candidates.shape)}.'
            )
        return torch.stack(
            [
                F.normalize(
                    op.apply(native_candidates[:, task_id]), dim=-1
                )
                for task_id, op in enumerate(self.operators)
            ],
            dim=1,
        )

    def canonicalize_auto(self, features: torch.Tensor, styles: torch.Tensor):
        if len(self.operators) == 1:
            return self.canonicalize_known_domain(features, 0), torch.ones(
                features.shape[0], 1, device=features.device
            )
        logits = self.router(styles)
        temperature = float(self.cfg['router'].get('temperature', 0.35))
        alpha = F.softmax(logits / temperature, dim=-1)
        if self.cfg['router'].get('hard_inference', False):
            idx = alpha.argmax(-1)
            hard = F.one_hot(idx, num_classes=alpha.shape[-1]).to(alpha.dtype)
            alpha = hard
        candidates = self.candidate_features(features)
        z = (candidates * alpha.unsqueeze(-1)).sum(dim=1)
        z = F.normalize(z, dim=-1)
        return z, alpha

    def forward_known_domain(self, features, task_id: int):
        task_id = int(task_id)
        native = self.adapt_known_domain(features, task_id)
        z = F.normalize(self.operators[task_id].apply(native), dim=-1)
        global_logits = self.classifier(z)
        if (
            not self.domain_evidence_enabled
            or task_id >= len(self.domain_evidence_heads)
        ):
            return global_logits
        head = self.domain_evidence_heads[task_id]
        blend_weight = head.blend_weight.to(
            device=global_logits.device, dtype=global_logits.dtype
        ).clamp(0.0, 1.0 - 1e-6)
        if float(blend_weight) <= 0:
            return global_logits
        local_log_prob = torch.full_like(global_logits, float('-inf'))
        local_log_prob[:, head.class_ids] = F.log_softmax(head(native), dim=-1)
        return torch.logaddexp(
            F.log_softmax(global_logits, dim=-1) + torch.log1p(-blend_weight),
            local_log_prob + blend_weight.log(),
        )

    def forward_domain_mixture(
        self,
        features,
        domain_probabilities,
        native_candidates: torch.Tensor | None = None,
        class_domain_scores: torch.Tensor | None = None,
    ):
        """Marginalize calibrated class posteriors under a domain prior.

        Comparing raw maxima from different adapters is not probabilistically
        meaningful and caused the first domain to dominate.  Normalizing each
        expert over classes before marginalization removes that path-scale bias.
        """
        if native_candidates is None:
            native_candidates = self.native_candidate_features(features)
        candidates = self.candidate_features(
            features, native_candidates=native_candidates
        )
        if native_candidates.shape != candidates.shape:
            raise ValueError(
                'native_candidates must have shape {}, got {}.'.format(
                    tuple(candidates.shape), tuple(native_candidates.shape)
                )
            )
        candidate_logits = torch.stack(
            [
                self.classifier(candidates[:, domain_id])
                for domain_id in range(candidates.shape[1])
            ],
            dim=1,
        )
        if domain_probabilities.shape != candidate_logits.shape[:2]:
            raise ValueError(
                'domain_probabilities must have shape {}, got {}.'.format(
                    tuple(candidate_logits.shape[:2]),
                    tuple(domain_probabilities.shape),
                )
            )
        class_log_prob = F.log_softmax(candidate_logits, dim=-1)
        # A shared classifier provides cross-domain semantic consistency, but
        # it has to discriminate labels that never coexist in a given domain.
        # The frozen local evidence head contributes only a residual posterior
        # for labels observed in that domain; probability mixing (rather than
        # a hard support mask) preserves recovery from routing uncertainty.
        if (
            self.domain_evidence_enabled
            and len(self.domain_evidence_heads) == candidates.shape[1]
        ):
            local_log_prob = []
            blend_weights = []
            for domain_id, head in enumerate(self.domain_evidence_heads):
                local_logits = head(native_candidates[:, domain_id])
                full = torch.full_like(
                    candidate_logits[:, domain_id], float('-inf')
                )
                full[:, head.class_ids] = F.log_softmax(local_logits, dim=-1)
                local_log_prob.append(full)
                blend_weights.append(
                    head.blend_weight.to(
                        device=class_log_prob.device, dtype=class_log_prob.dtype
                    ).clamp(0.0, 1.0 - 1e-6)
                )
            local_log_prob = torch.stack(local_log_prob, dim=1)
            blend_weights = torch.stack(blend_weights).view(1, -1, 1)
            class_log_prob = torch.logaddexp(
                class_log_prob + torch.log1p(-blend_weights),
                local_log_prob + blend_weights.log(),
            )
        domain_log_prob = domain_probabilities.clamp_min(1e-12).log()
        if class_domain_scores is not None:
            if class_domain_scores.shape != class_log_prob.shape:
                raise ValueError('Semantic scores must have [batch, domains, classes].')
            router_cfg = self.cfg.get('router', {}) or {}
            weight = float(router_cfg.get('semantic_weight', 0.5))
            temperature = float(router_cfg.get('semantic_temperature', 0.05))
            if not 0.0 <= weight <= 1.0 or temperature <= 0.0:
                raise ValueError('Invalid semantic routing weight/temperature.')
            if weight > 0 and class_log_prob.shape[1] > 1:
                support = torch.isfinite(class_domain_scores)
                # Equal prior per semantic class: recurring labels should not
                # gain extra votes simply because more domains contain them.
                occurrences = support.sum(dim=1).clamp_min(1).to(class_log_prob.dtype)
                semantic = class_domain_scores / temperature
                semantic = semantic - semantic.flatten(1).amax(1)[:, None, None]
                joint = (
                    class_log_prob
                    + (1.0 - weight) * domain_log_prob.unsqueeze(-1)
                    + weight * semantic
                    - occurrences.log().unsqueeze(1)
                )
                normalizer = torch.logsumexp(joint.flatten(1), dim=-1)
                joint = joint - normalizer[:, None, None]
                logits = torch.logsumexp(joint, dim=1)
                # Returned domain probabilities are inferred joint marginals,
                # not the true test domain. Domain IDs remain metric-only.
                alpha = torch.logsumexp(joint, dim=-1).exp()
                return logits, alpha
        logits = torch.logsumexp(
            class_log_prob + domain_log_prob.unsqueeze(-1), dim=1
        )
        return logits, domain_probabilities

    def forward_auto(self, features, styles):
        inference_mode = str(
            self.cfg.get('router', {}).get('inference_mode', 'learned')
        ).lower()
        if inference_mode in {'density', 'density_mixture', 'domain_keys', 'semantic_mixture'}:
            raise RuntimeError(
                'Density routing requires the external domain-key bank. Use '
                'DomainStyleBank.domain_probabilities followed by '
                'forward_domain_mixture (the trainer/test path does this automatically).'
            )
        if inference_mode == 'classwise_max':
            candidates = self.candidate_features(features)
            candidate_logits = torch.stack(
                [
                    self.classifier(candidates[:, domain_id])
                    for domain_id in range(candidates.shape[1])
                ],
                dim=1,
            )
            logits = candidate_logits.max(dim=1).values
            route = candidate_logits.max(dim=-1).values.argmax(dim=-1)
            alpha = F.one_hot(
                route, num_classes=candidates.shape[1]
            ).to(candidate_logits.dtype)
            return logits, alpha
        z, alpha = self.canonicalize_auto(features, styles)
        return self.classifier(z), alpha
