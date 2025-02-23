# Universal Phase Retrieval Transformer for Single-Pattern Structured Light Three-Dimensional Imaging

## Dependencies
- Python 3.9
- PyTorch 1.10.0

```
pip install -r requirements.txt
python setup.py develop
```
## Datasets


|  Dataset  | FP1000   | FP672   | FP147   | FP1523   | Total   |
|  ----  | ----  | ----  | ----  | ----  | ----  | 
|  Train | 720  | 540  | 108  | 1200  | 2568  |
|  Validation | 80  | 60  | 12  | 150  | 302  |
|  Test | 200  | 72  | 27  | 173 | 472  |
|  Direction | Horizontal  | Horizontal  | Vertical  | Horizontal  | -  |
|  Frequency | 48  | 100  | 32  | 80  | -  |

Refer to the related references in the manuscript for the complete datasets.

## Implementation
### Train

```shell
#Universal Type
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt options/train/UPRT/UPRT.yml --launcher pytorch
```
### Test
```shell
#Universal Type
python scripts/test.py
```

## Citation
```
@article{song2025universal,
  title={Universal phase retrieval transformer for single-pattern structured light three-dimensional imaging},
  author={Song, Jianwen and Liu, Kai and Sowmya, Arcot and Sun, Changming},
  journal={Optics and Lasers in Engineering},
  year={2025}
}
```

## Acknowledgement
The codes are based on BasicSR. We thank the developers and contributors for their efforts in developing and maintaining the framework.