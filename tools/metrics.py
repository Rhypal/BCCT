from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch


def classification_metrics(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.cpu().long()
    target = target.cpu().long()
    correct = (pred == target)
    return {
        'accuracy': float(correct.float().mean().item()) if target.numel() else 0.0,
        'correct': int(correct.sum().item()),
        'total': int(target.numel()),
    }


def per_task_accuracy(pred: torch.Tensor, target: torch.Tensor, task_ids: torch.Tensor):
    out = {}
    for tid in sorted(task_ids.unique().tolist()):
        m = task_ids == tid
        out[int(tid)] = classification_metrics(pred[m], target[m])['accuracy']
    return out


def per_class_accuracy(pred: torch.Tensor, target: torch.Tensor):
    out = {}
    for cid in sorted(target.unique().tolist()):
        m = target == cid
        out[int(cid)] = classification_metrics(pred[m], target[m])['accuracy']
    return out


def recurring_novel_harmonic(
    pred: torch.Tensor,
    target: torch.Tensor,
    task_ids: torch.Tensor,
    class_groups: Mapping[int, Mapping[str, Sequence[int]]],
):
    """Accuracy balance between recurring and novel class occurrences.

    A class occurrence is assigned according to its status when its task first
    enters the stream: a class already seen in an earlier task is recurring;
    otherwise it is novel. ``H-RN`` is the harmonic mean specified by the
    project evaluation protocol::

        H-RN = 2 * Hrec * Hnov / (Hrec + Hnov)

    Accuracies are sample-level, matching the project's Last/Avg accuracy
    convention. H-RN is undefined until both groups contain test samples.
    """
    pred = pred.cpu().long().reshape(-1)
    target = target.cpu().long().reshape(-1)
    task_ids = task_ids.cpu().long().reshape(-1)
    if not (pred.numel() == target.numel() == task_ids.numel()):
        raise ValueError('Prediction, target, and task-ID lengths do not match.')

    recurring_mask = torch.zeros(target.numel(), dtype=torch.bool)
    novel_mask = torch.zeros(target.numel(), dtype=torch.bool)
    for raw_task_id, groups in class_groups.items():
        task_id = int(raw_task_id)
        task_mask = task_ids == task_id
        if not bool(task_mask.any()):
            continue
        for group_name, destination in (
            ('recurring_ids', recurring_mask),
            ('novel_ids', novel_mask),
        ):
            class_ids = [int(value) for value in groups.get(group_name, [])]
            if not class_ids:
                continue
            ids = torch.tensor(class_ids, dtype=torch.long)
            class_mask = (target.unsqueeze(1) == ids.unsqueeze(0)).any(dim=1)
            destination |= task_mask & class_mask

    overlap = recurring_mask & novel_mask
    if bool(overlap.any()):
        raise ValueError('A test sample was assigned to both recurring and novel groups.')
    covered = recurring_mask | novel_mask
    if int(covered.sum().item()) != target.numel():
        missing = int(target.numel() - covered.sum().item())
        raise ValueError(
            f'{missing} test samples have no recurring/novel group assignment.'
        )

    recurring_count = int(recurring_mask.sum().item())
    novel_count = int(novel_mask.sum().item())
    h_rec = (
        classification_metrics(pred[recurring_mask], target[recurring_mask])['accuracy']
        if recurring_count else None
    )
    h_nov = (
        classification_metrics(pred[novel_mask], target[novel_mask])['accuracy']
        if novel_count else None
    )
    if h_rec is None or h_nov is None:
        h_rn = None
    else:
        denominator = h_rec + h_nov
        h_rn = 0.0 if denominator <= 0.0 else 2.0 * h_rec * h_nov / denominator
    return {
        'h_rec': h_rec,
        'h_nov': h_nov,
        'h_rn': h_rn,
        'recurring_count': recurring_count,
        'novel_count': novel_count,
    }


def continual_forgetting(
    task_accuracy_history: Sequence[Mapping[int, float]],
    current_task_id: int | None = None,
):
    """Compute task-wise and average forgetting from an accuracy matrix.

    ``task_accuracy_history[t][i]`` is the accuracy on task ``i`` after the
    model has finished training task ``t``. Following the standard continual
    learning definition, forgetting for an old task is its best historical
    accuracy minus its accuracy at the current stage. The newly learned task
    is excluded because it has not had an opportunity to be forgotten yet.

    The historical maximum includes the current value. Consequently, improved
    backward transfer produces zero forgetting instead of a negative value,
    matching the metric used by the DUCT evaluation code cited by DSPCL.
    """
    if not task_accuracy_history:
        return 0.0, {}

    normalized_history = [
        {int(task_id): float(accuracy) for task_id, accuracy in row.items()}
        for row in task_accuracy_history
    ]
    current = normalized_history[-1]
    if current_task_id is None:
        current_task_id = max(current) if current else -1
    current_task_id = int(current_task_id)

    per_task = {}
    for task_id in sorted(task_id for task_id in current if task_id < current_task_id):
        observations = [
            row[task_id] for row in normalized_history if task_id in row
        ]
        best_accuracy = max(observations)
        per_task[task_id] = max(0.0, best_accuracy - current[task_id])

    average = sum(per_task.values()) / len(per_task) if per_task else 0.0
    return average, per_task
