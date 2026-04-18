# Documentation Index

Reference material for the BIM/CAD transformer pipeline.

## Concept guides

| File | Topic |
|---|---|
| [`ground_truth.md`](ground_truth.md) | What "ground truth" means for each task, with toy examples |
| [`dataset.md`](dataset.md) | How the synthetic CAD/BIM floorplans are generated, sample contents, distribution |
| [`vit.md`](vit.md) | Vision Transformer classifier — intuition, architecture, tiny examples |
| [`detr.md`](detr.md) | DETR-lite detector — queries, Hungarian matcher, loss |
| [`mask2former.md`](mask2former.md) | Mask2Former-lite segmenter — mask queries, dot-product masks |
| [`comparison.md`](comparison.md) | Side-by-side model comparison and analogies |

## Engineering guides

| File | Topic |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | End-to-end data flow, module layout, extension points |
| [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) | CLI walkthrough, hyperparameters, expected curves, troubleshooting |
| [`EXPLANATIONS.md`](EXPLANATIONS.md) | Single-file combined version of the concept guides |

## Suggested reading order

New to the project? Read in this order:

1. [`dataset.md`](dataset.md) — understand what the models eat
2. [`ground_truth.md`](ground_truth.md) — understand what they aim to output
3. [`vit.md`](vit.md) → [`detr.md`](detr.md) → [`mask2former.md`](mask2former.md) — one model at a time
4. [`comparison.md`](comparison.md) — tie them together
5. [`ARCHITECTURE.md`](ARCHITECTURE.md) — zoom out to code organisation
6. [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) — run it yourself
