# QSM归一化、物理尺度恢复与评价

本页说明新实验的可逆归一化流程。这里的命令和数值均为示例；本次功能仅经合成数据测试，没有拟合真实数据的归一化参数、训练真实模型或确认历史预测的物理单位。新参数不能倒套到归一化历史不明的旧权重或旧预测上。

## 旧流程与新流程

| 情况 | 参数与处理 | 可以得出的结论 |
|---|---|---|
| 原项目旧实现或仅有state_dict的权重 | 若使用PIL/uint8、裁剪、逐例target min-max，或没有保存实际转换参数，信息可能已丢失 | 不能根据输出范围、文件名或新训练集统计猜测历史反变换；当前可见源码也不能证明历史实际执行协议 |
| 原minimal手动固定参数 | 显式指定`--qsm-scale`、可选`--qsm-offset`和单位；默认offset为0 | 转换可逆，但手填数值本身不证明参数来自训练集 |
| 新增train-only参数文件 | 仅用最终train成员的可靠QSM体素拟合一套参数，保存输入、划分、单位和审核来源 | 后续train/val/test及推理共用同一套参数；不在验证或测试标签上重新拟合 |

运行时的QSM变换仍是固定仿射变换，生成器使用线性输出。QSM全程以浮点值处理，不经过PIL或uint8，不做target裁剪，不要求归一化值位于`[0,1]`或`[-1,1]`。这不改变网络结构，也不保证预测已经准确。

SWI输入另用输入脑mask内的0.5/99.5百分位缩放并截断到`[0,1]`。这是只依赖SWI的输入处理，不能用来恢复QSM尺度；不能把参考QSM可靠mask充当部署时的SWI输入mask。

## 先核实单位和参考零点

每行标签用manifest的`qsm_units`声明实际输入单位，只接受`ppm`或`ppb`；没有该列时需通过`--qsm-units`或已核实QC记录提供输入单位。`--units`指定参数与训练输出使用的共同单位，不能用它重解释已声明的输入单位。

```text
1 ppm = 1000 ppb
q_canonical = convert_units(q_loaded, input_units -> parameter_units)
y = (q_canonical - b) / s
q_prediction = y_prediction * s + b
```

`s`必须为有限正数，`b`为有限数，两者都以参数文件的单位表示。先逆变换到参数单位，再按请求转换输出单位。NIfTI的`mm`空间单位、`scl_slope`/`scl_inter`和文件名不能证明磁化率单位或参考约定。

QSM参考组织、参考区域、参考目标值及重建定义须独立核实。`b`只是可逆数值居中，不是重新选择物理参考零点；减去训练均值也不使原来不同参考的队列变得可比。单位或参考仍不明确时，保留unverified及原因，不能因拟合成功而写成ppm/ppb精度认证。参考、配准和审核要求见[DATA_QC.md](DATA_QC.md)。

## 默认：参与者等权的训练集mean/std

`fit_normalization`默认`--method mean_std`：

```text
offset b = μ_train
scale  s = σ_train
```

对每名参与者，将其所有train访视的可靠体素合并，得到该人的体素均值`μ_i`和总体方差`v_i`。参与者之间等权：

```text
μ_train = (1/N) * sum_i μ_i
σ_train² = (1/N) * sum_i [v_i + (μ_i - μ_train)²]
```

这包含参与者内与参与者间变异，**不是仅计算各参与者均值的标准差**。同人多访视共用一个参与者权重，访视内体素在该人合并支持域内按体素等权；不把切片、较大的体积或额外访视当成额外独立参与者。有完整`global_subject_id`时按该标识分组，否则使用`subject_id`；标识真实性仍需独立审计。

统计只使用`split=train`且`mask_path > 0`的体素；可靠mask须非空、非负、与准备后的图像同格，并包含在SWI输入脑mask内。图像以float32解码、单位转换后的统计以float64累积，方差采用总体定义`ddof=0`。空支持域、非有限值、非正scale等情况会拒绝拟合。

可选`--method zero_rms`保持数值零点：

```text
offset b = 0
scale  s = sqrt(σ_train² + μ_train²)
```

这是同一参与者等权分布的均方根尺度，不是标准差，也不是最大值。两种方法都须在查看测试结果前确定；不能按test误差择优切换。

## 1. 在最终train上拟合参数文件

以下示例假定manifest已声明每行`qsm_units`，并已有对应QC与独立性报告：

```bash
python -m swixqsm_minimal.fit_normalization \
  --manifest /prepared/manifest.csv \
  --units ppm \
  --method mean_std \
  --output /experiments/qsm_normalization.json \
  --qc-report /audit/qc/report.json \
  --independence-report /audit/independence/report.json
```

如CSV没有`qsm_units`列，可另加真实的`--qsm-units ppb`，表示标签先从ppb转换为上例的ppm；这与输出参数单位不同。参考ID可用`--reference-id ACTUAL_REFERENCE_ID`作声明/一致性检查，不执行重新参考。

质量与独立性门槛沿用`fit_baselines`：必须选择`--qc-report`或`--allow-unverified-data "实际缺证原因"`；独立性报告可选，缺失时明确记为unverified，也可显式用`--allow-unverified-independence "实际缺证原因"`记录。两个维度各自保留状态，任何一个通过均不能替代另一个。

拟合仅打开或哈希train的输入文件，不解码val/test影像或使用其target统计；完整manifest的身份/划分元数据仍用于一致性核查。每份输出JSON必须是新文件，不覆盖旧文件。

参数文件绑定train真实绝对输入路径、单位与文件SHA。应在**最终训练环境、最终划分及最终mask路径**上拟合；本机拟合后将数据移动到服务器不同路径时，应在服务器重新拟合并审计，不能只手改JSON。这是参数统计，不是模型训练；读取和文件hash仍会产生I/O。

## 2. 训练入口共用参数文件

原minimal接口和四模型入口均支持`--normalization-file`。它替代手动`--qsm-scale`/`--qsm-offset`，不能同时传这些手动参数；`--units`必须与文件中的共同单位完全一致。

```bash
python -m swixqsm_minimal.train \
  --model plain_unet \
  --manifest /prepared/manifest.csv \
  --normalization-file /experiments/qsm_normalization.json \
  --units ppm \
  --output /experiments/plain_unet_mean_std \
  --qc-report /audit/qc/report.json \
  --independence-report /audit/independence/report.json
```

不带`--model`的原训练接口使用同一组归一化参数；模型选择与其余训练选项见[TRAINING_MODELS.md](TRAINING_MODELS.md)。训练重新核对train成员、路径、单位和源文件hash，并保留参数文件来源。输入单位fallback可从拟合文件/QC继承；显式`--qsm-units`不能与已记录的输入解释冲突。验证标签应用相同参数，不重新估计均值或尺度。

改变scale会改变归一化L1与对抗项的相对权重：`L1_normalized = MAE_physical / scale`，因此配方中`lambda_l1 * L1_normalized`对应的物理L1系数为`lambda_l1 / scale`。数值居中/缩放也会改变判别器输入分布，不能据此声称自动保持旧损失平衡。比较实验应固定并记录共同归一化参数及损失系数，在预定的验证流程中检查损失、梯度及物理误差表现；不能按test结果调整。

服务器首次尚无参数JSON时，先按现有配置完成`run_server --prepare-only`，再针对其最终prepared manifest运行上述fit，随后直接调用训练CLI。不要原地改已绑定的prepared/resume配置后宣称可直接续跑。服务器配置的`training.normalization_file`适用于事先已有、且绑定同一train输入路径的参数文件；它不会自动拟合参数。详细准备顺序见[SERVER_TRAINING.md](SERVER_TRAINING.md)。

## 3. 验证/测试自动恢复物理尺度

```bash
python -m swixqsm_minimal.validate \
  --checkpoint /experiments/plain_unet_mean_std/best.pt \
  --manifest /prepared/manifest.csv \
  --split test \
  --output /evaluation/test_mae_ppb.json \
  --device cpu \
  --output-units ppb \
  --qc-report /audit/qc/report.json \
  --independence-report /audit/independence/report.json
```

`validate`从checkpoint自动读取归一化，调用既有验证引擎，对网络预测和已归一化标签都先逆变换，再计算可靠支持域内的物理MAE与`per_subject_mae`，最后对参与者等权平均。默认输出checkpoint单位，上例显式转换为ppb。没有`qsm_units`列时仍需输入单位fallback或QC解释；可另用`--reference-id`及`--expected-phase l1|warmup|gan`检查对应来源/阶段。此入口只报告上述MAE，不代替完整多指标评价。test不参与重新拟合参数或选模。

## 4. 推理自动逆变换一次，保存已恢复的数据

```bash
python -m swixqsm_minimal.test.infer \
  --checkpoint /experiments/plain_unet_mean_std/best.pt \
  --swi /input/swi.nii.gz \
  --swi-mask /input/swi_brain_mask.nii.gz \
  --output /predictions/qsm_ppb.nii.gz \
  --output-units ppb \
  --device cpu
```

推理只需要SWI及其输入脑mask。它自动用checkpoint参数执行`q=y*s+b`，再按需转换单位，保存float32 NIfTI及JSON sidecar；默认单位与checkpoint相同。脑外设为物理数值0，这是输出mask约定，不是重新设定QSM参考。

sidecar的`ValueSpace="physical_qsm"`以及`InverseNormalization.applied=true`表明文件已恢复尺度；还记录参数单位、输出单位、scale、offset、公式及checkpoint SHA。**该输出直接用于`evaluate`，不能再次乘scale或加offset。** `evaluate`读取物理尺度文件进行单位一致化和指标计算，不执行第二次QSM反归一化，详见[EVALUATION.md](EVALUATION.md)。

## 5. 仅对明确的原始网络输出单独恢复尺度

若自行导出了逆变换前的浮点网络输出，并能确认它使用对应checkpoint的归一化，可运行：

```bash
python -m swixqsm_minimal.restore_scale \
  --checkpoint /experiments/plain_unet_mean_std/best.pt \
  --input /predictions/raw_network_output.nii.gz \
  --input-value-space network_normalized \
  --swi-mask /input/swi_brain_mask.nii.gz \
  --output /predictions/qsm_restored_ppb.nii.gz \
  --output-units ppb
```

`--input-value-space`是对输入来源的显式声明，不能凭数值范围判断后随意填写。工具拒绝已标为physical、已执行inverse、单位或checkpoint hash冲突的输入，也拒绝缺少可信归一化元数据的旧权重。它不修复PIL/uint8或裁剪丢失的信息。正常`test.infer`的输出不需要再调用此工具。

## 数值例子与需保存的记录

以下仅演示公式，参数不是实际训练集拟合结果：

```python
import numpy as np
from swixqsm_minimal.util.normalization import QSMNormalization

norm = QSMNormalization(scale=0.1, offset=0.02, units="ppm")
q_ppb = np.array([-180.0, 20.0, 120.0], dtype=np.float32)
y = norm.normalize(q_ppb, input_units="ppb")  # 约为[-2, 0, 1]
q_restored_ppb = norm.denormalize(y, output_units="ppb")
# 约为[-180, 20, 120]；不裁剪负值，也不限制y的范围。
```

保留完整fit JSON及其hash、最终manifest/划分与输入hash、QC/独立性报告、参考声明和实际命令。fit文件使用`swixqsm.fitted_normalization` schema，包含方法、参与者/访视/体素计数、等权统计、train身份hash及每个输入来源。运行时`normalization`记录版本、QSM固定仿射参数、SWI处理及线性输出；checkpoint另保存参数来源，预测sidecar记录实际逆变换。

这些记录支持新实验复查，不补造旧权重的归一化历史。旧权重实际单位、训练预处理或参考仍未知时，继续保留未知状态；加载相同结构的state_dict也不能使新参数自动成为旧模型的正确反变换。
