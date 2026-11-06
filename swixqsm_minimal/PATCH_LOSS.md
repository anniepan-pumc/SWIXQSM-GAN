# 有效patch对抗损失与训练诊断

2026-10-17。第四项补充有效patch损失及可追溯诊断，不改变生成器、判别器的网络结构或参数形状，不修改既有权重，不执行真实病例训练。新的损失选项只影响以后实际调用训练时的优化；它不会追溯改变固定模型的输出。

作者已明确：held-out test participants预先独立划为test，训练没有val，且仅在权重固定后评价。本次沿用这一事实，不要求为增加诊断补划val或重新训练。固定模型的选权记录及原文§2.4更正见 [TRAINING_AUDIT.md](TRAINING_AUDIT.md)。

## 与评审的对应及证据边界

| 评审条目 | 直接要求 | 本次能提供的支持 |
|---|---|---|
| R1 major 3 | 解释细节减少；如保留静脉、微出血等主张，补小结构/病灶量化 | 增加物理分布、误差及训练诊断，支持观察细节损失；没有真实ROI/病灶标注就不能声称完成病灶级验证。 |
| R1 major 6 | 报告mean bias、limits of agreement及区域ppb误差，区分相关和一致性 | 记录有效域整体bias与分布；整体bias不是ROI Bland–Altman或区域一致性的替代。 |
| R3 major 1 | 明确指标，比较零图/均值/线性基线，展示误差图和体素scatter | 沿用统一物理评价，增加每case/method分布及零图NMSE参照；分布JSON本身不是完整图表或全部评审分析。 |
| R3 major 8 | L1-only、标准pix2pix、对抗项消融；说明lambda、epochs、停止规则及augmentation | 使未来对抗损失范围与实际损失比可追溯；没有既存独立对照权重时，不能通过诊断制造这些实验。 |

“背景patch稀释对抗项”是**由当前minimal源码推导的机制**，不是审稿人直接提出的指控，也不是历史论文权重存在该问题的证据。原实现对D的真/假输入使用同一mask，L1也已限制在有效体素，这些行为正确。但BCE默认对所有logit取平均，完全没有有效组织的感受野仍进入分母；这些位置没有通向有效生成组织的梯度，同时仍参与D的纯背景分类。有效组织占比和padding因此可能影响对抗项的归一化。

这一机制不能单独证明模型塌缩、既有结果过平滑的原因、标签泄漏或小病灶一定改善。当前minimal的条件PatchGAN也不等于原稿所述的历史判别器，不能把本次修改视为原模型逐项复现。

## 有效patch的确切规则

实现位于 `util/patch_loss.py`。`patch_support(mask, discriminator)`读取实际 `D.net` 的Conv2d核、stride、padding和dilation，逐层传播正值支持域：一个输出位置的卷积输入中只要有任意正支持，就将其标为有效。LeakyReLU不改变结构支持范围。

- mask正值表示有效成员，0表示无效；patch权重是二元membership，不是概率或覆盖比例加权。
- 每层显式zero padding均为无效支持，不把人工padding当组织。
- 有效与无效组织混合的边界patch仍纳入，不要求整个感受野都在脑内。
- 只有整个实际感受野均无正支持的patch被排除；不按logit中心点取mask，不缩放mask近似支持。
- 只支持明确的Sequential Conv2d/LeakyReLU拓扑；增加未知空间层、全局层或归一化层会拒绝，不能继续沿用旧感受野声明。

当前D为三层 `Conv2d(k=4,s=2,p=1)` 后接 `Conv2d(k=3,s=1,p=1)`，理论感受野38×38，logit步长8。以输入像素中心索引0、1、…计，首个logit中心为3.5，名义输入索引范围为[-15,22]；第j个位置平移8j。边界真正可达的输入支持由逐层传播确定，名义范围不意味着图外存在有效组织。报告按height、width顺序记录这些参数。

设有效logit集合为S，所选patch数为N，则：

```text
BCE_S(z,t) = sum(j in S, BCEWithLogits(z_j,t_j)) / N
L_D = 0.5 * (BCE_S(D(real),1) + BCE_S(D(fake_detached),0))
L_G = masked_L1(pred,target) + actual_adv_weight * BCE_S(D(fake_for_G),1)
```

D real、D detached fake和G的对抗项使用**相同支持**。分母是整个batch内所选patch总数，不是先对每slice求均值再等权平均；有效patch较多的slice贡献较多。整batch无有效patch时明确报错，不用epsilon伪造分母。`patch_bce`仅对选中logit计算BCE，因此被排除位置的值与梯度不参与该损失。

未来训练CLI的 `--adversarial-mode`：

| 选项 | 行为 |
|---|---|
| `valid_patch`（默认） | 按上述正支持规则排除全背景感受野，再以有效patch总数归一。 |
| `all_patches` | 显式保留本包原先对全部logit取平均的损失范围，供受控损失对照；包括纯背景patch。 |

`all_patches`只保留这一损失范围选择，不恢复历史判别器、数据变换、初始化、优化器或训练轨迹，不能据此声称复现历史模型。输入仍使用同一有效mask，L1公式没有改变。更换patch归一化会改变未来对抗梯度的尺度，应记录实际模式，不能宣称同一个数值lambda在两种模式下具有相同作用。

## 后续训练日志怎样解释

`train`在每个训练步收集损失，每epoch写入 `history.jsonl` 的 `training_diagnostics`，并随checkpoint和 `selection.json`保存。`--diagnostic-batches`默认4，仅控制每epoch最多保留多少个训练batch的**物理分布样本**；设0关闭分布采样，逐步损失日志仍然收集。

每个字段保存 `sum`、`count`、`mean`、`status`和 `reduction`，epoch均值由累计sum/count计算。不同损失使用各自的真实分母，不能把字段均值互相直接相加当作同一个epoch总目标：

| 字段 | 分母与含义 |
|---|---|
| `l1` | 各step有效体素的mask总量；本包数据加载器将有效mask转为二元。L1在归一化QSM目标空间计算。 |
| `adv`、`weighted_adv` | 实际执行对抗步的所选patch数；分别为原始BCE与乘实际adv_weight后的BCE。 |
| `d_real`、`d_fake`、`d_total` | 实际执行D步的所选patch数；total为0.5×(real+fake)。 |
| `g_total` | 每batch一个标量的平均，与实际step目标一致。 |
| `l1_batch`、`weighted_adv_batch` | 各batch损失标量的平均，用于计算实际加权项/L1的数值比。 |
| `valid_patch_fraction` | 实际有效patch总数/全部logit数，与选择哪种loss模式分开记录。 |
| `loss_patch_fraction` | loss实际纳入patch数/全部logit数；all_patches时为1。 |
| `valid_voxel_fraction` | 正mask体素数/输入batch元素总数。 |

L1和送入D的QSM采用既定归一化目标空间；原始BCE是无量纲损失，不是ppb误差。训练目标仍为 `masked_L1 + actual_adv_weight * adversarial_BCE`，不机械套用原稿 `L_adv + 0.2 L1` 或标准pix2pix的系数。

warmup/L1-only阶段未执行的raw adv和D损失，count为0、mean为null、status为not_observed，不能报作“观察到0损失”。`weighted_adv_batch`在这些阶段记录实际加入G目标的贡献为0；这与声称已计算原始对抗BCE不同。

`weighted_adv_to_l1`是**epoch内batch平均加权对抗损失 / batch平均L1**，不是逐step比值的平均，也不是不同分母池化后的raw损失之比。L1为0时返回null及原因。该比值只表示数值损失规模，不表示梯度贡献、梯度方向或有效训练强度；程序不据此自动调lambda，不用test诊断改变选权。

日志中的 `d_real_probability`和 `d_fake_probability`是D更新前的sigmoid分数，`d_fake_for_g_probability`来自D更新后、计算G项时的D。这些是不同时间点的判别分数，不能视为同一个固定D上的直接前后对照，也不自动是经过校准的真实概率。

### 物理分布样本

分布采样复用已有G forward产生的**更新前预测**，不会为日志再做一次train-mode forward，因此不额外改变BatchNorm统计或引入一次Dropout采样。先用既定固定scale/offset反归一化到物理QSM单位，再统计 `mask_path`的正值有效域；这里是参考可靠性/覆盖支持域，不是整个输入脑区。

样本来自每epoch已shuffle训练loader的前若干batch，含batch索引和subject IDs，不保证代表全队列。每份记录独立保存：

- prediction与target的mean、population SD（ddof=0）、p01/p50/p99、min/max；分位数为本次供入数组的精确线性插值结果。
- 有效体素数、bias、MAE、RMSE、prediction/target SD比。
- 未中心化target能量、误差平方和、标准NMSE及物理零图NMSE参照；target能量非零时零图NMSE为1。

**不平均batch分位数，不把这些样本写成epoch或队列分位数**。target SD为0、能量为0、非有限值或数值溢出均有独立状态和null，不能把未定义统计写为0。低SD比本身不能证明mode collapse；NMSE使用未中心化能量，也不能直接称为解释方差。

### 来源传播与选模

`training_objective`记录本run目标和真实D拓扑/支持/reduction，`training_diagnostics`记录对应epoch诊断。推理sidecar分别写入 `TrainingObjective`、`TrainingDiagnostics`；评价保留原预测的 `PredictionTrainingObjective`、`PredictionTrainingDiagnostics`。旧来源缺失时保持None/JSON null，不依据当前默认模式倒填历史训练信息。

本次不改变后续新run的val物理受试者等权MAE选模、阶段best或固定epochs规则；阶段含义见 [TRAINING_AUDIT.md](TRAINING_AUDIT.md)。这些新run规则不应用来改写当前历史模型“无val、固定权重后test”的实际流程。

## 当前固定模型可直接补充的诊断

在已有评价命令增加 `--distribution-diagnostics`，会生成 `distribution_diagnostics.json`。它使用已保存的物理预测和reference，在各case共同有效评价mask内，对每种方法分别计算上述分布。输出逐case/method记录，不是患者多访视池化分位数或全队列分位数；既有主要指标汇总规则见 [EVALUATION.md](EVALUATION.md)。

```bash
python -m swixqsm_minimal.evaluate \
  --manifest /prepared/frozen_evaluation_manifest.csv --split test \
  --qc-report /qc/final/report.json \
  --allow-unverified-independence "历史模型尚无本包格式的完整身份与初始化暴露审计；模型选择来源另见selection-record" \
  --selection-record /records/actual_model_selection.json \
  --distribution-diagnostics \
  --output /experiments/evaluation/frozen_distribution \
  --units ppb --reference-id actual_reference_id \
  --save-predictions
```

示例要求真实预测路径、单位/完整参考来源、通过的数据QC及真实冻结权重选择记录；模板见 [selection_record.example.json](selection_record.example.json)。历史预测若明确为未核实QC，应按 [DATA_QC.md](DATA_QC.md)使用显式未核实路径，不能补造已核实来源。`--selection-record`核对真实权重字节与每份预测的来源hash，未提供日志时仍仅为声明层级。

本命令不做G/D forward、不调用优化器、不要求val、不选模型、不调参数。可继续增加已经准备好的训练集基线，所有方法使用同一评价mask和单位/参考；相应共同空间条件仍需满足。没有 `--data-range` 时PSNR/SSIM保持未定义，不从test分布自动估计范围。输出采用临时目录完成后再发布，失败不留下部分成功的结果目录。

### 仅供后续选择执行的训练示例

以下命令本次**未执行**，也不是本轮返修前提。若未来做新训练，需真实train/val、QC与身份/初始化审计；旧权重初始化另遵循 [TRAINING_AUDIT.md](TRAINING_AUDIT.md)。

```bash
python -m swixqsm_minimal.train \
  --manifest /prepared/new_experiment_manifest.csv --output /experiments/new_valid_patch \
  --qc-report /qc/new_experiment/report.json \
  --independence-report /audit/random_initialization/report.json \
  --units ppm --qsm-scale 0.1 --size 416 --batch-size 4 \
  --epochs 20 --warmup-epochs 2 --adv-weight 0.01 \
  --adversarial-mode valid_patch --diagnostic-batches 4 \
  --seed 42 --device cuda
```

显式 `--adversarial-mode all_patches`可作为以后受控的损失范围对照，但须保持身份划分、初始化、数值协议、预算及其他设置可比；不能用当前固定模型事后声称已完成此消融。

## 仍需真实证据的结论

合成检查可验证感受野支持、边界、归一分母、被排除logit的梯度、日志状态与来源传播；这些不证明真实模型性能提升。固定预测诊断也不能恢复历史D loss、实际对抗更新轨迹或曾使用的lambda。

没有真实ROI/病灶标注及相应设计，本次不新增小结构保留、微出血/钙化鉴别或区域一致性结论；整体分布、HFEN和低MAE均不能代替这些证据。不声称有效patch修正已改善GAN、减少幻觉或优于L1/pix2pix。第5项采样/I/O优化亦不在本次范围。
