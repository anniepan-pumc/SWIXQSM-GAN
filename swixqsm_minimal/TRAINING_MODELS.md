# 返修重训：选择四种模型及评审消融

2026-10-17。此入口供作者上传后进行新的对照训练。原项目、原生成器源文件及既有权重保留；本次仅用临时合成影像验证程序，没有训练真实受试者模型，也没有新增性能结论。评审逐条对应及相位对照的条件见 [REVIEWER_BASELINES.md](REVIEWER_BASELINES.md)。

已有 primary cohort 目录可用新增 [SERVER_TRAINING.md](SERVER_TRAINING.md) 自动生成 manifest、保留历史1773例test，并用 `run_server --config ... --model <名称>` 一键准备和训练；配置模板为 [server_training.example.json](server_training.example.json)。下文保留直接使用已准备 manifest 的单项训练接口。

## 选择实验

在包含 `swixqsm_minimal/` 的目录运行：

```bash
python -m swixqsm_minimal.train --list-models
python -m swixqsm_minimal.train --model pix2pix --help
```

| `--model` | 生成器 / 判别器 | 默认生成器损失 |
|---|---|---|
| `swixqsm_gan` | 原 UNetSmall，32/64/128/256、跳跃连接；QSM-only 全局判别器修正版 | `L_adv + 0.2 L1` |
| `fcn_gan_l1` | 同宽全卷积编码器—解码器，去掉跳跃连接；与 SWIX 相同判别器 | `L_adv + 0.2 L1` |
| `plain_unet` | 与 SWIX 完全相同的 UNetSmall；无判别器 | `L1` |
| `pix2pix` | 8 层下采样、base64 的 U-Net；条件 70×70 PatchGAN | `L_adv + 100 L1` |
| `swixqsm_l1_ablation` | 与 SWIX 相同生成器、初始化和工厂；判别器不执行、不更新 | `0.2 L1`，严格关闭对抗项 |
| `swixqsm_lambda1` | 与 SWIX 相同 | `L_adv + 1 L1` |
| `swixqsm_lambda10` | 与 SWIX 相同 | `L_adv + 10 L1` |
| `swixqsm_lambda100` | 与 SWIX 相同 | `L_adv + 100 L1` |
| `swixqsm_c2` | 与 SWIX 相同（返修 C2） | `0.1 L_adv + 100 L1` |

`--model all` 按表中前四项顺序训练；`--model reviewer_suite` 顺序运行全部八项神经网络实验，每项单独目录。零图、均值图、逐体素线性回归通过后述独立入口运行，不消耗神经网络训练轮次。运行套件时不能覆盖各自损失系数；单项实验可用 `--lambda-l1`、`--lambda-adv` 显式设置并留档。严格消融的系数固定。

FCN 是作者确认的“去除跳跃连接”架构对照，不冒称某篇未指定 FCN 论文的复现。它改变生成器结构，与 SWIX 使用相同判别器及损失。严格对抗消融与普通 plain U-Net 分开，避免同时改变 L1 系数而混淆对抗项作用。

## 首先固定真实 train / val / test

原 N_test 人在初期留出且未用于选择权重，仍保留为 test。新验证集只从原 5083 人训练池中划出；目录名 `validate` 不能证明其用途，历史上存放 N_test 人的目录仍须标记 `test`。

先准备真实 manifest，含基础六列和 `global_subject_id`、真实 `qsm_units`、QC/参考记录；可从 [qc_manifest.example.csv](qc_manifest.example.csv) 修改。所有访视使用同一真实参与者身份。为适配完整独立性审计，同一人还应统一 `subject_id`，不同采集由路径和 `case_id` 区分。

若目前仅有 `train/test`，可运行：

```bash
python -m swixqsm_minimal.split_training \
  --manifest /prepared/original_train_test.csv \
  --val-fraction 0.1 --seed 42 \
  --output /prepared/retraining_split
```

输出 `manifest.csv` 和 `split_record.json`。按全局参与者分组，所有访视及队列别名整组分配；仅更改原 train 行的 split，保留 test 成员及其数据声明。此步骤只读元数据，不打开影像。比例按参与者数向上取整，不承诺病例行数恰为 90/10；已存在 val 时拒绝重复划分。**所有模型使用这一份固定名单**。

新名单确定后，按 [DATA_QC.md](DATA_QC.md) 生成 QC 报告，并按 [TRAINING_AUDIT.md](TRAINING_AUDIT.md) 做随机初始化身份审计。例如：

```bash
python -m swixqsm_minimal.qc \
  --manifest /prepared/retraining_split/manifest.csv \
  --contract /prepared/data_contract.json \
  --output /qc/retraining --split train val

python -m swixqsm_minimal.audit_splits \
  --manifest /prepared/retraining_split/manifest.csv \
  --output /audit/retraining_random
```

QC 须有真实人工配准/参考审核；模板 pending 不会被当成通过。独立性审计会读取全表 SWI 做重复内容检查，这是准备阶段的审计，不是利用 test 误差选模。训练入口只加载 train/val 的影像，不打开 test 文件或用 test 指标调参。修改名单后须重新生成匹配的报告。

## 上传后训练

Python ≥3.10，依赖见 [requirements.txt](requirements.txt)。以下数据路径、单位、尺度及训练预算都是**语法示例**；须用实际已核实标签确定固定尺度，再在全部实验共用。`scale=0.1 ppm` 不代表历史预测已经完成单位核实。

```bash
python -m pip install -r swixqsm_minimal/requirements.txt

python -m swixqsm_minimal.train \
  --model swixqsm_gan \
  --manifest /prepared/retraining_split/manifest.csv \
  --output /experiments/retraining/swixqsm_gan_seed42 \
  --qc-report /qc/retraining/report.json \
  --independence-report /audit/retraining_random/report.json \
  --units ppm --qsm-scale 0.1 --qsm-offset 0 \
  --size 416 --batch-size 4 --epochs 100 \
  --lr 0.0001 --lr-disc 0.0001 --lr-policy cosine \
  --warmup-epochs 0 --early-stopping-patience 0 \
  --seed 42 --device cuda
```

将 `--model` 改为 `fcn_gan_l1`、`plain_unet`、`pix2pix` 或其他表中名称，并使用新的 output 即可。改为 `all` 或 `reviewer_suite` 时 output 是套件父目录，程序为各项创建子目录。单模型单设备训练；也可在不同 GPU 上分别运行单项并显式设置 `--device cuda:0` 等，不能复用同一输出目录。

默认 FP32、Adam(β₁=0.5, β₂=0.999)、余弦学习率、100 轮、batch4、无 warmup、无早停。这里给出统一起始协议，不代表已优化。请在看 test 前固定预算、权重搜索范围和随机种子；若做多种子重复，可统一使用预先约定的种子及独立目录，报告参与者级结果与训练重复的差异。pix2pix 参数更多；显存不足时应先为全部方法确定共同可行 batch，不能默默把它缩小并继续宣称批次条件完全相同。本入口没有 AMP、梯度累积或 DDP。

可选 `--lr-policy plateau` 使用 `ReduceLROnPlateau`：每完整轮监测新 val 的参与者等权物理 MAE，不使用 test。参数为 `--plateau-factor .5 --plateau-patience 5 --plateau-threshold 0 --min-lr 1e-6`；threshold 使用绝对值模式，单位为训练 `--units`，默认 0。按 PyTorch 规则，超过允许的不改善轮数后降低；不是每 5 轮固定降低。G/D 各自保存状态，D 仅在本轮真实更新后调度；warmup→GAN 时 G 重置 plateau 历史并保留当前 lr。历史记录本轮和下一轮 lr，以及实际 scheduler 事件，完整轮恢复保持该状态。旧 `constant/cosine` 仍可选择，默认不变。

当前服务器的可选起点为 G lr=`2e-4`、D lr=`1e-4`、3 轮 L1 warmup、plateau、最多50轮、阶段内 val 早停 patience=15，见 [SERVER_PXQ.md](SERVER_PXQ.md)。这是待验证的较高初始学习率方案，不保证更快收敛或更高精度；GAN 稳定性仍须结合损失、梯度及验证曲线判断。新的比较方案须共同固定归一化、学习率及选模规则，避免看 test 后调节。

默认各模型都**不做数据增强**，并在配置中明确记录；随机采样不等于方向或对比度增强。这没有完成评审提出的方向/TE 泛化实验。若后续加入增强，应独立预定义、在各方法一致应用并重新验证物理变换，不能把任意旋转视为有正确磁场方向标签的物理样本。

未完成 QC 或独立性证据时，仅可显式用相应 `--allow-unverified-data "实际原因"` / `--allow-unverified-independence "实际原因"` 做诊断，所有结果保留 unverified；该选项不代替真实审核。正式结果应补齐证据。

## 损失、结构与公平性边界

所有生成器输出均为不裁剪的线性 QSM。标签按 `y=(q_canonical-offset)/scale` 变换；推理按相同常数反变换。L1 在 `mask>0` 的有效体素中求平均；G 对抗项为假图目标为真的 BCEWithLogits，D 损失为真/假 BCE 均值的半和。D/G 分开更新，每批 G 只前向一次。真/假 QSM 以及条件 SWI 在判别器输入使用同一有效 mask。

表中 L1 系数作用于**固定归一化后的 QSM**。其物理误差等效系数为 `lambda_l1 / scale`。因此本包的 `0.2` 不能在未核实历史数据尺度时被宣称为旧论文在 ppb 标签空间使用 λ=0.2 的优化轨迹的复现；比较 `0.2/1/10/100` 时必须固定单位、scale、offset 和 mask，只用 val 选择。

SWIX 和 FCN 的全局 D 保留原 QSM-only 3×3 卷积/InstanceNorm 主干及空间展开分类器，将历史的双输出 ReLU 分类头修正为单个无激活 logit，并适配配置的面内尺寸。它不是此前 minimal 的条件 patch 判别器，也不是旧判别器权重逐项复现。论文应表述为修正版训练，并披露输出头、定量输出和预处理变化。

pix2pix 参考 [原论文](https://arxiv.org/abs/1611.07004) 与 [官方网络实现](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py)：base64 的 `unet_256` 拓扑，BN，三层 0.5 dropout，条件 70×70 PatchGAN，`lambda_L1=100`。为有正负值的 QSM，将 Tanh 改为线性头，使用共同脑区 mask 和固定归一化。输入先统一 padding 到 `--size`，pix2pix 内部再右/下补至 256 的倍数并切回原尺寸，例如 416→512→416；没有插值或压缩 MRI。应称为适配 QSM 的标准 pix2pix 对照，并说明这些修改。

默认 `--adversarial-mode valid_patch` 排除名义卷积覆盖中完全无有效体素的 patch；所有有效输出位置统一归一化。pix2pix 的 BatchNorm 会耦合批次和空间位置，70×70 是卷积感受野，不代表严格局部依赖或脑外梯度完全隔离。SWIX 全局 D 每个非空切片只有一个 logit，使用整片而非 patch 支持。`all_patches` 可预定义为额外损失对照，不能将这些不同 D 设计当作完全相同的对抗机制。

各实验使用共同初始化规则：卷积/线性权重 normal(0,0.02)、BN normal(1,0.02)、bias=0。G、D、采样使用分离的随机流；同 seed 下 SWIX、plain U-Net 和严格对抗消融的初始 G 完全相同。记录实际初始 state hash、网络参数数、逐轮批次索引 hash 与更新次数。FCN/pix2pix 架构不同，不能要求参数张量相同。原 UNetSmall 保留了不参与 forward 的旧 deconv 注册参数，配置分别报告已注册和实际使用参数数。

## 选权重、断点续训和日志

每轮仅在 val 上计算物理单位参与者等权 MAE，同人多访视先合并有效体素；不会按切片数把参与者当成独立重复。保存：

- `config.json`、`manifest_resolved.json`：来源、初始化、架构、损失、采样、单位、QC 和协议。
- `history.jsonl`：逐轮 MAE、参与者误差、损失、有效支持比例、物理输出分布样本、读取统计与实际批次 hash。
- `best.pt`：全程最低 val MAE，可能属于 warmup；`best_gan.pt`：实际对抗训练后的最佳权重；`best_l1.pt`、`best_warmup.pt`：相应阶段存在时保存。
- `last.pt`、`selection.json`、`run_result.json`：最后完整轮、真实更新计数及停止原因。

GAN 方法对照应使用 `best_gan.pt`，L1 方法使用 `best_l1.pt`，在研究方案中预先确定。默认 `early-stopping-patience=0` 表示固定总轮次内验证选优；设正整数时在 L1 或 GAN 各自阶段计算连续未改善轮数，warmup 不参与早停计数。G 学习率每完整轮推进，D 只在有实际 D 更新的轮推进，D 余弦周期为总轮数减 warmup。不得根据 test 决定是否继续训练。

```bash
python -m swixqsm_minimal.train \
  --resume /experiments/retraining/swixqsm_gan_seed42/last.pt \
  --device cuda
```

仅支持本入口的 `last.pt` 在完整轮边界恢复；恢复优化器、调度器、随机状态、历史和选优状态。必须保留原 manifest、代码、运行库版本、输入文件及实验目录，不能改变总轮数、损失或数据名单，不能绕过已触发的早停。程序会核对版本、源代码、manifest、已消费输入的 size/mtime 和历史一致性；已核实 QC 另提供内容 hash，size/mtime 本身不是内容证明。

可预先用 `--epochs 100 --max-epochs-this-run 10` 分段执行前十轮，再以上述方式继续原 100 轮预算。重启某个套件时应逐个恢复其子目录的 last.pt；套件入口拒绝覆盖，未开始项可单独运行到新目录。本机验证了相同 CPU 环境下连续/分段的权重数值一致；跨设备及不同 GPU 内核不承诺位级重放。仅支持 CPU/CUDA。非有限值会终止本轮，不保存半个优化步；写盘中断造成 history/checkpoint 不一致时明确拒绝恢复，不自动拼接记录。

## 三个简单基线与统一评价

新划分后，均值图和逐体素回归必须**仅用新 train 重新拟合**，不能复用以全部原训练池拟合、因此包含新 val 的 artifact。二者要求真正共同解剖空间，不能用相同矩阵尺寸冒充配准。详见 [EVALUATION.md](EVALUATION.md)。

```bash
python -m swixqsm_minimal.fit_baselines \
  --manifest /prepared/retraining_split/manifest.csv \
  --output /experiments/retraining/simple_baselines/train_baselines.npz \
  --qc-report /qc/retraining/report.json \
  --independence-report /audit/retraining_random/report.json \
  --units ppb --reference-id actual_reference_id --space-id prepared_common_space
```

零图是物理值 0，评价时直接生成，无需训练。每个神经网络使用对应冻结权重推理；入口从 checkpoint 自动识别生成器架构：

```bash
python -m swixqsm_minimal.test.infer \
  --checkpoint /experiments/retraining/swixqsm_gan_seed42/best_gan.pt \
  --swi /prepared/test_case/SWI.nii.gz \
  --swi-mask /prepared/test_case/input_brain_mask.nii.gz \
  --output /predictions/swixqsm_gan/test_case_QSM.nii.gz \
  --device cuda
```

在对应评价 manifest 添加真实 `prediction_path` 后，用 `evaluate --split test` 逐模型评价；所有方法保持相同参与者、mask、参考零点和预定 data_range，报告 MAE/RMSE/NMSE/SSIM/PSNR/HFEN、bias 和已审核 ROI 一致性。正式执行前也须完成 test 的独立图像 QC。不得根据 test 表现反选 lambda、epoch、mask 或预处理。具体完整命令和 ROI 条件见 [EVALUATION.md](EVALUATION.md) 与 [ROI_EVALUATION.md](ROI_EVALUATION.md)。

## 程序验证范围

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m unittest discover -s swixqsm_minimal/test -t . -v
```

新增测试使用真实宽度的四种网络与临时合成 NIfTI，覆盖有限梯度、正负/超范围输出、损失与梯度隔离、416 网格、训练—保存—自动识别推理、test 文件不被训练读取、同人访视合并、严格消融、warmup 调度、验证早停及完整轮恢复。它们验证软件行为，不代表真实 MRI 训练收敛、临床分割质量或模型性能。实际本机版本与完整检查结果随上传包的验证记录提供。
