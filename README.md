# LeafTip Bench

Wheat500 和 Rice1700 独立训练、独立验证。支持 PET、SAHCC、CGTipNet、P2RLoss（全标注）、P2PNet、CLTR，默认在单张 CUDA 卡上顺序运行十二组。

| 模型 | 实际训练入口 | 默认 batch size |
|---|---|---:|
| PET | PET/main.py | 8 |
| SAHCC | sah-leaf-counting/train.py | 40 |
| CGTipNet | tipnova/train.py --variant cgtipnet | 40 |
| P2RLoss | p2rloss/main.py --label 100 | 16 |
| P2PNet | 官方网络 + 叶尖训练适配器 | 8 |
| CLTR | 官方公开实现 + 叶尖训练适配器 | 4 |

PET 的 8 是保守起点，并非架构上限。SAHCC 的 40 来自此前 RiceW3999 实验。默认 50 epochs、seed 2024、每轮验证；显存不足时降低对应模型的 batch size，使用新 run-name。各模型保留自己的训练增强、学习率与损失；这不是严格的同训练器消融。

## 1. 准备已有仓库和权重

默认兄弟目录：`PET/`、`sah-leaf-counting/`、`tipnova/`、`p2rloss/`、`leaftip-bench/`。其他父目录用 `--repos-root`。TipNova 使用支持 `--variant cgtipnet` 的 `main` 分支；PET 要包含 RiceW3999 数据入口、自动续训和 `best_metrics.json` 输出。不要修改仓库源码；运行时记录各仓库提交号。依赖使用原训练环境，调度器自身只用标准库。

PET 的本地 ImageNet 权重仍放在 `PET/pretrained/vgg16_bn-6c64b313.pth`；SAHCC、CGTipNet 和 P2RLoss 通过 `--weights` 共用同一权重。可以把它指向 PET 已有的文件，无需复制。

P2RLoss 使用支持 RiceW3999 加载器、自动续训和 `best_metrics.json` 的版本。两套数据均以 `--label 100` 独立训练，跳过半监督伪标签分支。

### P2PNet 和 CLTR

在 `leaftip-bench` 的父目录运行：

```bash
git clone https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet.git
git -C CrowdCounting-P2PNet checkout --detach 5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455
git clone https://github.com/dk-liang/CLTR.git
git -C CLTR checkout --detach ab1b5e58eec4851cc629379abeeaceff908b48f4
```

适配器在现有 CUDA PyTorch 环境运行，需要匹配的 torchvision，以及 NumPy、Pillow、SciPy、tqdm。不运行旧仓库的人群数据预处理或多卡启动脚本。P2PNet 用 `--weights` 的 VGG16-BN；CLTR 需要 [ResNet-50 ImageNet V1 权重](https://download.pytorch.org/models/resnet50-0676ba61.pth)，放到本仓库 `pretrained/resnet50-0676ba61.pth` 或用 `--cltr-weights` 指定。不在训练期间联网下载权重。

两者复用 `tipnova/main` 的点数据加载器：在线 512×512 裁块训练，整图验证。P2PNet 沿用官方架构与匹配代价，修正公开代码中回归损失键名不一致的问题。CLTR 沿用官方公开实现的 ResNet-50、Conditional DETR、点邻域距离监督与匹配代码；公开实现和论文版有差异（如未加入 FPN），结果应标为公开实现的叶尖适配，不是论文设置复现。CLTR 默认 500 个查询，可用 `--cltr-queries` 调整；训练中如果标注数超过查询数则报错。两者保存最佳模型、逐图指标及每轮可续跑检查点。

```bash
python batch_train.py --dry-run --models p2pnet cltr
python batch_train.py --run-name point_baselines --models p2pnet cltr \\
  --weights ../PET/pretrained/vgg16_bn-6c64b313.pth \\
  --cltr-weights pretrained/resnet50-0676ba61.pth
```

显存不足时用 `--p2pnet-batch-size`、`--cltr-batch-size` 分别调整，并换新 run-name。

## 2. 准备数据

已有独立数据集可直接放在 `data/Wheat500` 和 `data/Rice1700`，或用 `--data-root` 指向这两个目录的父目录。每个数据集需要 `JPEGImages/`、`PointAnnotations/`、`ImageSets/{train,val}.txt`。清单每行完整文件主名；标注每行原图像素 `x y`；空标注可为空文件。没有独立测试集时，结果只叫验证集结果。

如果你只有合并版 RiceW3999，可以自动按 `wheat500__` 和 `ricetip__` 分出两套原划分。只生成清单并链接原图和标注目录，原始数据不动，已有目标不覆盖。

```bash
python prepare_subsets.py --source ../sah-leaf-counting/data/RiceW3999
```

## 3. 批量跑

```bash
python batch_train.py --dry-run
python batch_train.py --run-name ep50 --weights ../PET/pretrained/vgg16_bn-6c64b313.pth
```

只跑一组：追加 `--datasets Wheat500 --models pet`。可分别通过 `--pet-batch-size`、`--sah-batch-size`、`--cgtipnet-batch-size`、`--p2rloss-batch-size` 调整。重跑同名实验用相同参数加 `--resume`，完成的任务跳过，未完成的由原训练器从最近完整 epoch 恢复。不要在同一个 run-name 上并发启动多个调度器。

`--models CGTipNet` 和 `--models cgtipnet` 等效；`--models tipfuse`、`--tipfuse-batch-size` 是兼容别名。新实验用 `cgtipnet` 作为结果标识；已有 `tipfuse_seed*` 运行如需续跑，请继续使用原调度器版本，或另取 run-name 开新实验。

每组目录含 `config.json`、`console.log` 和训练产物。汇总在 `outputs/<run-name>/results.csv`，每组一行。失败记录在表内并继续后续任务，最终返回非零退出码。选最佳权重用各入口的验证 MAE；汇总 MAE、RMSE 与一对一匹配 F1@4/8/16px。复用现有合并集训练权重不能替代这两套数据的独立训练。

调度、续跑及结果汇总用模拟训练器验证；真实 CUDA 训练和显存未实测。
