# FastFlow

An unofficial pytorch implementation of [_FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows_](https://arxiv.org/abs/2111.07677) (Jiawei Yu et al.).

As the paper doesn't give all implementation details, it's kinda difficult to reproduce its result. A very close AUROC is achieved in this repo. But there are still some confusions:
- [ ] Unmatched A.D parameter [#]()
- [ ] Unmentioned but critical LayerNorm [#]()

_Really appreciate the inspiring [discussion](https://github.com/AlessioGalluccio/FastFlow/issues/14) with the community. Please feel free to comment by raising new issues or PRs._

## Installation

```bash
pip install -r requirements.txt
```

## Data
We use [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) to verify the performance.

The dataset is organized in the following structure:
```
mvtec
|- bottle
|  |- train
|  |- test
|  |- ground_truth
|- cable
|  |- train
|  |- test
|  |- ground_truth
...
```
## Training

## Evaluation


## Performance

