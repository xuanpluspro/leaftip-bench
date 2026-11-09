# LeafTip Bench

Wheat500 和 Rice1700 独立训练、独立验证。支持 PET、SAHCC、CGTipNet、P2RLoss（全标注），默认在单张 CUDA 卡上顺序运行八组。P2PNet、CSRNet 暂未接入。

| 模型 | 实际训练入口 | 默认 batch size |
|---|---|---:|
| PET | PET/main.py | 8 |
| SAHCC | sah-leaf-counting/train.py | 40 |
| CGTipNet | tipnova/train.py --variant cgtipnet | 40 |
| P2RLoss | p2rloss/main.py --label 100 | 16 |

PET 的 8 是保守起点，并非架构上限。SAHCC 的 40 来自此前 RiceW3999 实验。默认 50 epochs、seed 2024、每轮验证；显存不足时降低对应模型的 batch size，使用新 run-name。各模型保留自己的训练增强、学习率与损失；这不是严格的同训练器消融。

## 1. 准备已有仓库和权重

默认兄弟目录：`PET/`、`sah-leaf-counting/`、`tipnova/`、`p2rloss/`、`leaftip-bench/`。其他父目录用 `--repos-root`。TipNova 使用支持 `--variant cgtipnet` 的 `main` 分支；PET 要包含 RiceW3999 数据入口、自动续训和 `best_metrics.json` 输出。不要修改仓库源码；运行时记录各仓库提交号。依赖使用原训练环境，调度器自身只用标准库。

PET 的本地 ImageNet 权重仍放在 `PET/pretrained/vgg16_bn-6c64b313.pth`；SAHCC、CGTipNet 和 P2RLoss 通过 `--weights` 共用同一权重。可以把它指向 PET 已有的文件，无需复制。

P2RLoss 使用支持 RiceW3999 加载器、自动续训和 `best_metrics.json` 的版本。两套数据均以 `--label 100` 独立训练，跳过半监督伪标签分支。

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
