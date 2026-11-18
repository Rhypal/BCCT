# BCCT

**Bridge-Consensus Canonical Transport for Cross-Domain Class-Incremental Remote Sensing Scene Classification**

At each stage, the model learns from the current domain's training images and predicts over all classes seen so far without test-time domain identities. The framework has three components:

- **Bridge-Consensus Canonical Transport (BCCT)** fits one affine map to the paired current and historical means and covariances of recurring classes. It estimates the map in a subspace identified from those discrepancies and applies it to all incoming classes.
- **Distribution-Informed Continual Consolidation (DICC)** uses mergeable low-rank class statistics for feature replay and covariance-aware margin regularization as the classifier expands.
- **Task-ID-Free Dual-Space Inference** fuses canonical and domain-native probabilities within each domain path, then combines class evidence across paths using soft routing.

## Installation

From the repository root, run:

```bash
python -m pip install -r requirements.txt
python tests/smoke_test.py
```

The smoke test uses synthetic tensors and needs no datasets or pretrained weights. Training selects CUDA when available and otherwise uses the CPU.

## Data preparation

Obtain the five datasets from their original providers. The paper uses one dataset per stage in this order:

| Stage | Dataset | Local root | Alias mapping key |
| --- | --- | --- | --- |
| 1 | AID | `data/AID` | `AID` |
| 2 | NWPU-RESISC45 | `data/NWPU` | `NWPU-RESISC45` |
| 3 | PatternNet | `data/PatternNet` | `PatternNet` |
| 4 | RSI-CB256 | `data/RSI` | `RSI-CB256` |
| 5 | UC Merced Land Use | `data/UCM` | `UCMerced_LandUse` |

Each local root must contain `train/<canonical_class>/` and `test/<canonical_class>/` folders. To create the class-wise 80:20 splits and apply the alias mapping, run the following for AID and repeat for the other datasets using the corresponding root and mapping key:

```bash
python dataset/split.py --src /path/to/raw/AID --dst data/AID --dataset AID --aliases-file configs/class_aliases.yaml --seed 42 --test-ratio 0.2
```

The five-stage protocol covers 91 global classes. Check the discovered classes and their recurring or novel status before training:

```bash
python scripts/inspect_tasks.py --config configs/paper.yaml
```

## Training and evaluation

[`configs/paper.yaml`](configs/paper.yaml) records the settings saved with the final checkpoint for the main results, with local paths replaced by portable defaults. The frozen backbone is ImageNet-21K-pretrained ViT-B/16 using CLS features. The config uses timm to load the pretrained weights. For offline use, set `backbone.vit_b16_in21k.weights_path` to a local compatible checkpoint. The legacy `configs/default.yaml` contains local paths and different ablation settings.

```bash
python train.py --config configs/paper.yaml
python tests/test.py --config configs/paper.yaml --checkpoint output/bcct_paper/task_04_UCM.pt
```

Training writes each stage's checkpoint, `metrics.txt`, and `class_registry.json` to `output/bcct_paper/`. Evaluation covers every domain seen by the loaded checkpoint, with predictions over the cumulative class set. To resume from an intermediate checkpoint, pass `--resume` and its path to `train.py`.

## Reported results

| Setting | Final accuracy | Average incremental accuracy | Average forgetting | H-RN |
| --- | ---: | ---: | ---: | ---: |
| BCCT on the five-stage sequence | 95.71% | 95.32% | 0.98% | 95.64% |

H-RN is the harmonic mean of final accuracies on recurring and novel class occurrences. These are the manuscript results for the sequence above.
