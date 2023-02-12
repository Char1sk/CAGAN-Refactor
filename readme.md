# Refactor of CA-GAN

[[Paper@IEEE]](https://ieeexplore.ieee.org/document/9025751)  [[Project@Github]](https://github.com/fei-hdu/ca-gan/)  [[Paper@arxiv\]](https://arxiv.org/abs/1712.00899)  [[Project Page]](https://fei-hdu.github.io/ca-gan/

## Changes

- network code
- option/config
- tensorboard

## Folders & Files

### Codes

- data: MyDataset
- loss: Loss Functions
- models: Networks
- options: Arg Options
- utils: Utilities
- train.py: Training

### Others

- Logs: Log folder for TensorBoard
- Saves: Generated images & Saved models
- Scripts: Shell scripts to run Python scripts

压缩结果文件：Not bad, loss
计算展示Train和Test的结果：Train和Test周期性计算FID：Done
试一下loss比例
噪点问题和IO取值范围：0~1，train吊test捞

Log莫名丢失：--samples_per_plugin scalars=100000000
输出范围
数据清理
