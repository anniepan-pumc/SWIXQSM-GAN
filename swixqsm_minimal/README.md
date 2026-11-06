# SWIXQSM minimal：定量评价与返修训练对照

创建日期：2026-09-06。

本目录是独立开发副本，运行时不导入原项目的 `model/`、`data/`、`utils/` 或原训练/推理脚本。今后的单目标修正、三目标和跨层技术开发在此目录或另建分支进行；原版及原权重保留。这里是**修正后的迁移基线**，不等于历史 GAN 的逐项实验复现。

## 新增：可选择的返修重训入口

Training uses a local NIfTI case layout you provide via `--data-root` / manifest CSV; site-specific server paths are not included in this public package.

固定归一化参数的计算、训练接入及验证/测试反归一化见 [NORMALIZATION.md](NORMALIZATION.md)。新增 `fit_normalization` 只读取 train 影像，生成可审计参数 JSON；训练可用 `--normalization-file` 替代手工 `--qsm-scale/--qsm-offset`，新 checkpoint 自带参数，推理输出已恢复尺度的 float32 QSM。此工具不推断原始权重缺失的尺度。

作者现已另行授权编写四模型重训代码。运行 `python -m swixqsm_minimal.train --list-models` 查看选择；`--model swixqsm_gan/fcn_gan_l1/plain_unet/pix2pix` 分别训练，`--model all` 顺序运行四项，`--model reviewer_suite` 另加入严格对抗消融与 L1 权重敏感性。具体命令、物理尺度、结构修改及完整轮断点续训见 [TRAINING_MODELS.md](TRAINING_MODELS.md)，评审对应与相位对照条件见 [REVIEWER_BASELINES.md](REVIEWER_BASELINES.md)。

结合现有 `swi_qsm_data/train/<ID>`、`swi_qsm_data/validate/<ID>` 的自动清单与服务器一键入口见 `(removed from public package)`。`build_manifest` 兼容平铺或 `QSM/` 子目录，保留完整缺项；`run_server --config ... --model reviewer_suite` 统一准备、审计和顺序训练。历史 validate 始终作为 test；可按作者确认的已去颅骨 SWI 生成有来源记录的输入 mask，不复用 QSM mask。

这一新入口使用显式独立的模型工厂和训练引擎，不改旧生成器源文件或既有权重；推理按 checkpoint 元数据识别网络。新验证集只从原训练池按参与者划出，保留原 N_test 人 test。简单基线继续复用既有物理评价入口，并须在新 train 上重新拟合。本机仅运行合成功能测试，正式模型由作者上传后训练。

不带 `--model` 的原训练接口仍保留；下方有关 `--gen-checkpoint`、`--adv-weight`、`--freeze-bn` 的命令属于该接口，其系数约定和判别器与新四模型比较不同，不应混用参数或声称复现同一训练协议。

## 原最小接口与已有评价功能范围

- 单 SWI → 单 QSM，2D UNetSmall；保留原生成器层名称、形状、Dropout、BatchNorm 和上采样方式，严格加载旧生成器 state_dict。
- 新训练使用线性输出，支持有正负数的定量 QSM。旧 Sigmoid 无权重，因此 strict load 成功并不意味着新旧输出行为相同。
- 新的条件 PatchGAN，输出无激活 logits；旧判别器权重不兼容，不加载。支持纯 L1 热身及 `--adv-weight 0` 纯回归基线。
- float32 数据；只 padding，不 PIL、不 uint8、不拉伸、不裁剪、不自动配准。按实际有效层生成索引。
- 明确的患者级 train/val/test 划分检查；训练只访问 train/val 图像，test 不参与选模。
- 每个患者汇总有效体素 MAE，再对患者等权平均选模；同人多个访视合并体素统计，不把切片当独立患者。
- 最佳/末次训练快照、解析后的 manifest、配置、权重来源哈希与验证日志；推理保存原网格、物理尺度 NIfTI。
- 独立物理单位评价，以及零图、train-only逐体素平均图和线性回归基线；详见 [EVALUATION.md](EVALUATION.md)。
- 配对数据的方向/spacing、QSM参考记录、人工配准审核及文件哈希绑定；详见 [DATA_QC.md](DATA_QC.md)。
- 独立身份/预训练成员审计、当前运行阶段分别选模，以及固定既有权重的测试选模记录；详见 `(removed from public package)`。既有模型无验证集不自动要求重新训练。
- 按判别器真实感受野排除全背景patch，记录训练损失/有效比例及物理输出分布；既有固定预测可单独生成只读分布诊断，详见[PATCH_LOSS.md](PATCH_LOSS.md)。
- 固定预测的ROI区域物理误差、覆盖/审核及缺失记录，参与者层面的bias/LoA、Pearson、ICC(A,1)及配对bootstrap CI；详见[ROI_EVALUATION.md](ROI_EVALUATION.md)。不需要验证集或重训，也不自动生成标签。

此前六项补充保持既有权重和物理指标定义。第四项为后续GAN训练增加有效patch损失归一与诊断；masked L1公式保留。QC检查准备后的图像与记录，不执行重建、配准、重采样或轴重排；报告也不证明真实组织已准确配准。本次新增 `split_training` 和四模型入口的完整轮恢复；原接口的 `--gen-checkpoint` 仍仅加载 G 初始化新实验。回放比例 sampler、混合精度、梯度累积、多 GPU 并行、三目标或 z 轴模块尚未包含。采用普通单优化步。

## 文件布局

| 文件 | 职责 |
|---|---|
| `model/generator.py` | 独立复制的旧 UNetSmall 及其必要组件 |
| `model/discriminator.py` | 新条件 PatchGAN |
| `util/data.py` | manifest 校验、有效切片与浮点加载 |
| `util/normalization.py` | 共用可逆 QSM 变换、单位转换、SWI 归一化与 checkpoint 元数据校验 |
| `fit_normalization.py` / `util/normalization_workflow.py` | train-only 固定 offset/scale 拟合、单位继承、训练名单及文件哈希绑定 |
| `validate.py` / `restore_scale.py` | 新 checkpoint 的 val/test 物理 MAE，以及显式原始网络输出的一次性尺度恢复 |
| `util/nifti.py` | 清理来源强度 header，保存物理 QSM 与 JSON sidecar |
| `util/engine.py` | 严格权重加载、独立 G/D 更新及物理单位验证 |
| `train.py` | 最小训练入口及实验快照 |
| `train_models.py` / `model/training_models.py` | 四种可选网络、严格消融、L1权重配方及完整轮恢复 |
| `util/comparison_engine.py` / `util/model_loading.py` | 显式比较损失、诊断及按元数据加载生成器 |
| `split_training.py` | 只从原训练池按全局参与者划出 val，保留原 test |
| `build_manifest.py` / `prepare_swi_masks.py` | 目录候选、原始名单核对、缺失报告及明确启用的 SWI-only 支持域准备 |
| `run_server.py` / `server_training.example.json` | 配置化服务器准备与模型选择入口，保留执行计划和来源哈希 |
| `fit_baselines.py` / `evaluate.py` | 训练集基线拟合与统一物理单位评价入口 |
| `util/baselines.py` / `util/metrics.py` / `util/evaluation.py` | 基线、指标及受试者级汇总 |
| `qc.py` / `util/quality.py` / `util/quality_gate.py` | 几何、审核、文件哈希报告及入口核验 |
| `util/reference.py` / `util/qc_preview.py` | QSM参考记录/数值检查与有界图像预览 |
| `audit_splits.py` / `util/independence.py` | 全局身份、解码SWI重复及预训练暴露审计 |
| `util/training_state.py` / `util/selection_record.py` | 新训练阶段选模、既有冻结权重的选择记录 |
| `util/patch_loss.py` / `util/training_diagnostics.py` | 实际感受野支持、有效BCE、分母明确的损失及有界分布诊断 |
| `test/infer.py` | 仅依赖 SWI/输入脑掩膜的新模型推理入口 |
| `test/test_smoke.py` | 合成数据、梯度和端到端回归检查 |
| `test/test_normalization.py` | 数值往返、单位、非零偏移、NIfTI scaling、元数据兼容性 |
| `util/sampling.py`、`util/volume_cache.py` | 小组卷内切片打散、每实例有卷数/字节上限的准备后缓存 |
| `benchmark_io.py` | 有界只读读取次数/耗时/逐切片数值对比，不运行模型 |
| `util/regions.py`、`util/agreement.py` | 同网格真实ROI提取、完整访视合并、参与者区域一致性及CI |

顶层 `data.py`、`models.py`、`engine.py`、`infer.py`、`test_smoke.py` 仅保留兼容导入/入口，不保存第二份实现。主目录结构按 `train.py` / `model/` / `util/` / `test/` 分工。

生成器保留 `UpSample.deconv` 等原版未参与 forward 的参数，目的在于旧 state_dict 严格兼容，不将其计作新增技术。层宽仍为 32/64/128/256。

## 先准备数据

新实验请复制 [qc_manifest.example.csv](qc_manifest.example.csv)，填写真实路径、global_subject_id、参考来源和审核记录，并按 [data_contract.example.json](data_contract.example.json) 声明方向、间距和参考政策。详细流程见 [DATA_QC.md](DATA_QC.md)。示例的人工状态为pending，须完成真实审核后记录审核人/日期，再生成最终报告。契约中的范围和容差只是语法示例。

支持绝对路径，或相对 CSV 所在目录的路径。训练至少需要 train 和 val 两组。下列六列是基础必需字段，QC另需参考和审核字段；旧 `manifest.example.csv` 仅提供基础格式：

```csv
subject_id,split,swi_path,qsm_path,swi_mask_path,mask_path
```

1. 每行一个采集体积；患者标识在所有队列/访视中一致。同一患者不能跨 split，同一个输入路径不能重复。改名副本/硬链接的身份泄漏仍需人工审计；程序不能推断实际患者身份。
2. 四个 NIfTI 必须都是 finite 3D，并已配准到同一 shape/affine。代码检查网格，但不能判断实际配准是否正确。
3. `swi_mask_path` 是从输入 SWI 或部署时可获取数据生成的脑 mask；归一化和推理都使用它，不能用仅参考 QSM 才能生成的 mask 代替。
4. `mask_path` 是参考重建可靠性与采集覆盖的有效支持域，必须非空且包含在输入脑 mask 中；仅用于训练/验证。参考 mask 不传入生成器，判别器对真/假图施加相同支持域。
5. 提前核实所有 QSM 的单位、参考零点与重建定义。用 CSV 列 `qsm_units` 声明每行输入标签单位（ppm/ppb），列存在时禁止空值。缺此列时，QC需显式给 `--qsm-units`，消费端继承该解释，再转换至训练 `--units` 指定的共同单位；输出单位不能重解释已核实的输入单位。单位转换不会解决不同参考零点/重建方法的差异。
6. 切片轴固定为 NIfTI 第三轴；输入须采用统一且适合轴位训练的方向/协议。网络面内顺序为 `[Y,X]`，保持旧版 `permute(2,1,0)` 的约定；没有自动 canonical 重定向。

SWI 按全体积输入脑区的 0.5/99.5 百分位缩放并截断至 [0,1]，脑外为 0。这是仅用于输入的有损处理，**不用于 QSM 的反归一化**。其每例 low/high 写入预测 JSON，以便追溯。

QSM 的统一公式为：

```text
q_canonical = convert_units(q_loaded, qsm_units -> --units)
y = (q_canonical - offset) / scale
q_pred = y_pred * scale + offset
```

推荐先用 `fit_normalization` 在新 train 可靠体素中计算参数，再以 `--normalization-file` 消费。默认采用受试者等权的 mean/std；另可选择 offset=0 的 RMS。手工模式仍支持 `--qsm-scale` 和 `--qsm-offset`（默认 0）。参数均使用 `--units`；输入单位取自 `qsm_units`、拟合记录或 QC 声明，亦可用 `--qsm-units` 显式补充。使用固定训练集尺度，不做每患者/每切片 target min-max、不裁剪 QSM、不使用 SWI 范围或真实 QSM 百分位还原。预测可小于 0 或超出训练范围。offset 是可逆数值居中，不代表重新选择物理参考组织。

例如 `units=ppm, scale=0.1, offset=0.02`：`q=-0.18 ppm -> y=-2 -> q=-0.18 ppm`。`-180 ppb` 的标签先显式转换为 `-0.18 ppm`。训练、验证和推理都调用 `QSMNormalization`，防止多个路径各自实现不同公式。

每份新 checkpoint 保存结构化 `normalization`：版本、QSM units/scale/offset、是否裁剪、SWI 策略、输出激活。缺失/不支持/冲突的元数据会阻止直接推理，不会根据预测范围猜尺度。旧最小包中含完整且可识别的固定尺度 config 可明确迁移为 offset=0；仅有原版 state_dict 的权重只能用来初始化新训练。

这次修复保证数据变换可逆及输出单位可追溯，并不保证网络预测无误差。原版若已把 QSM 经 uint8、裁剪或无参数记录的逐例归一化处理，丢失的定量信息无法通过事后乘一个系数恢复，应以正确物理标签重新后训练。旧输入/输出协议与新协议不同；仅加载旧权重不等于完成量纲校准。

`--size` 为 padding 后正方形边长，至少 32 且为 8 的倍数，必须容纳全部准备后图像的面内尺寸。默认 416；只在右/下补零，保存时切回原尺寸。若出现尺寸超限，应增大 size 或先按研究协议准备共同网格，不能直接压缩整脑。

## 生成并使用数据QC报告

```bash
python -m swixqsm_minimal.qc \
  --manifest /prepared/qc_manifest.csv \
  --contract /prepared/data_contract.json \
  --output /qc/initial \
  --split train val test
```

pending、rejected或硬检查失败的病例会保留失败报告，不能作为通过记录使用。预览只展示有界样本；应在完整三维图像中核对配准和参考ROI。完成实际审核后更新manifest，用新目录生成 `/qc/final/report.json`。报告绑定输入图像、mask、参考/审核声明和证据文件；更改这些内容后须重新生成。

训练、基线拟合、评价CLI必须二选一：`--qc-report /qc/final/report.json`，或显式 `--allow-unverified-data "实际未核实原因"`。后一项仅供历史数据排查，状态及原因会记录为unverified，不是QC通过。训练只重查train/val，基线拟合只重查train，评价只重查所选split。参考身份包含id、方法、区域和目标ppb；仅同名id不足以证明参考一致。

新训练还须提供 `--independence-report`，或显式 `--allow-unverified-independence "实际缺失证据"`，与几何QC分别记录。审计报告必须对应实际随机/预训练初始化；旧权重需要完整、可追溯的成员历史，不能仅凭权重hash宣称未见测试者。新增global ID或修改审核字段后须重新生成相关报告。

当前已有5083例训练、1773例预先分离测试且无验证集的模型，可直接使用既有固定预测进行评价，填写[selection_record.example.json](selection_record.example.json)并给evaluate传 `--selection-record`；不调用train，不补造验证集，也不按新预处理重新解释旧权重。该记录核对选模声明、权重及预测来源，不能代替身份/预训练完整性审计。

现有固定预测的评价命令可另加 `--distribution-diagnostics`，生成逐例/方法的distribution_diagnostics.json，包含有效支持域内物理mean/SD、p01/p50/p99、bias/MAE/RMSE及零分母状态；不执行模型forward或优化器，也不能恢复旧训练的D loss。

## 环境与运行

Python >=3.10。依赖 PyTorch、NumPy、nibabel 和 SciPy（用于QC及物理空间指标）；不依赖 torchvision、torchmetrics、JAX、SimpleITK、PIL 或 torchinfo。先使用已有兼容环境，必要时安装本目录 `requirements.txt`。版本范围不等于已验证全部组合，实际实验配置中记录版本。

以下命令在 CVMed 根目录运行。参数中的路径和 QSM 尺度是示例，必须替换为完成 QC 后的实际数据与尺度。不会自动向正在重建的目录写入内容。

```bash
python -m swixqsm_minimal.train \
  --manifest /prepared/qc_manifest.csv \
  --output /experiments/swixqsm_g1_l1 \
  --qc-report /qc/final/report.json \
  --independence-report /audits/legacy_init/report.json \
  --gen-checkpoint /checkpoints/legacy_gen.pt \
  --reset-head \
  --units ppm --qsm-scale 0.1 --qsm-offset 0 \
  --size 416 --batch-size 4 --epochs 20 \
  --lr 1e-5 --adv-weight 0 --freeze-bn --device cuda
```

`--reset-head` 用于重置与新尺度不匹配的最终卷积；中间特征仍由旧 G 初始化。省略则保留所有已加载层。若不传 `--gen-checkpoint`，则为同架构随机初始化。

上述是后续新实验示例，legacy_init报告需用同一 `--checkpoint` 和真实 `--history` 生成；当前冻结模型的返修评价不要求执行训练命令。

在完成纯回归流程验证后可新建 GAN 实验目录，使用 `--adv-weight 0.01 --warmup-epochs 2`：前两轮仅 L1，之后为 `L_masked-L1 + 0.01 * L_adv`。这些是起点，不代表已优化超参数。学习率固定；`--freeze-bn` 固定 BN 运行统计但仍训练其 affine 参数。小批次是否使用该选项应与对照保持一致。

后续训练的 `--adversarial-mode valid_patch`为默认策略：实际感受野内有有效体素才计入，D真/假与G共用权重，按整批选中patch总数归一。显式 `all_patches`可保留原全patch均值作将来对照。每轮保存损失及有效patch比例；`--diagnostic-batches 4`默认只记录前4个打散训练批次的物理分布，设0关闭分布采样，损失日志仍保留。分位数不冒称全epoch，loss数值比例不等同于梯度贡献，不据此在测试集调参。

新实验分别保存best_l1.pt、best_warmup.pt、best_gan.pt（仅存在实际执行的阶段）及最后/全程最佳权重。best.pt仍可属于warmup；history、checkpoint和selection.json记录真实current-run阶段及G/D更新次数。adv-weight>0而epochs<=warmup-epochs会提前拒绝。阶段名只指本次运行，不推断来源权重过去是否接受过GAN训练。

如果从上一轮新模型的 `best.pt` 开始，`--gen-checkpoint` 同样可以读取其 generator，但 D/optimizer 会重置，需使用新 output 目录并记录是新阶段而非精确续训。

```bash
python -m swixqsm_minimal.test.infer \
  --checkpoint /experiments/swixqsm_g1_l1/best.pt \
  --swi /prepared/case/SWI.nii.gz \
  --swi-mask /prepared/case/swi_brain_mask.nii.gz \
  --output /predictions/case_QSM.nii.gz \
  --device cuda
```

带已核实数据契约的新checkpoint会自动检查SWI及脑mask的方向、spacing和几何，并继承完整QSM参考定义；SWI-only输入只记为geometry_verified，不能验证跨采集配准或输出参考ROI的数值。没有已核实契约的legacy/unverified checkpoint需显式添加 `--allow-unverified-data "实际原因"`；该状态保留在预测JSON中。

推理仍只接受带可识别 normalization 的本包训练快照；opt-out不能将任意旧state_dict变成已校准模型。输出 float32，保留 SWI shape/affine，脑外置零。加载 NIfTI 时由 nibabel 应用输入 slope/intercept **一次**；保存时使用 slope=1/intercept=0，重置 SWI 的 intensity intent/calibration 范围，防止源图显示信息误导定量显示。生成同名 JSON，明确 Units、Normalization、DataReference、DataQC、输入百分位参数和 checkpoint SHA256。图像、JSON 或实验目录已存在时拒绝覆盖。

## 验证与后续扩展

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s swixqsm_minimal/test -t . -v
```

测试使用临时合成 NIfTI，不读取受试者图像或启动正式训练。包括正负与超范围 QSM、非零 offset、ppm/ppb 转换、患者泄漏拒绝、网格错误拒绝、G/D 梯度隔离、两轮 L1/GAN 训练、checkpoint 加载及推理几何恢复。额外检验非默认 NIfTI slope/intercept 和已知输出 `-2 -> -0.18 ppm` 的数值链，以及QC哈希失效、人工状态、参考身份和训练→推理→基线→评价接入；这些测试不构成真实病例配准或模型性能验证。

后续新训练默认 `--sampling-mode volume_grouped --volume-group-size 2`，先随机subject/访视，再在不超过两卷的小组内打散切片，保留每组尾部小批次，每轮每有效切片一次。`--sampling-mode slice_random`可显式使用全局随机切片作对照。两者批次组合及优化轨迹可能不同，不能当作历史训练的精确重放；公平对照还应报告实际更新步数。数据采样和loader使用独立随机状态。

每个train/val实例各自最多缓存 `--cache-volumes 2 --cache-mib 256`；字节仅计准备后三数组驻留空间，不是总内存上限。仅用有效mask建索引，首访才解码整卷并做原有预处理。索引的header检查与首访内容检查不替代前置QC。配置保存sampling_policy/indexing_io，每轮history/checkpoint/selection记录training_io，区分初始化读取、遍历decode、cache命中/逐出/超预算绕过。当前显式num_workers=0；没有创建磁盘影像缓存。详见 [SAMPLING_IO.md](SAMPLING_IO.md)，含本次样例限制、数值验收与真实小样本I/O结果。

后续先锁定此基线和对照配置，再增加 `models_z.py`、新的损失或独立三目标模块。不要回写原来的 `model/unet.py`、`data/data_loader.py`、`utils/train_gan.py` 等文件。生成器源文件哈希见 `legacy_provenance.json`。
