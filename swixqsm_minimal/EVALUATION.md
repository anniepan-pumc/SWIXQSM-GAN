# 物理单位评价与训练集基线

新增于2026-10-17。入口为 `python -m swixqsm_minimal.fit_baselines` 和 `python -m swixqsm_minimal.evaluate`，在CVMed根目录运行。物理评价不改变生成器、判别器或checkpoint选模规则；推理增加数据契约检查与参考来源记录。后续有效patch训练损失与诊断单列于[PATCH_LOSS.md](PATCH_LOSS.md)，不改变这里的物理指标定义或已有权重。

训练、拟合和评价入口现需 `--qc-report`，或显式 `--allow-unverified-data "实际未核实原因"`。QC流程见 [DATA_QC.md](DATA_QC.md)，模板为 [qc_manifest.example.csv](qc_manifest.example.csv) 和 [data_contract.example.json](data_contract.example.json)。报告核查准备后的几何、参考记录、证据和人工审核声明，不执行配准；相同网格或一份通过报告均不能独立证明真实组织配准正确。本次QC接入不改变网络或损失。

已有模型可在冻结权重后直接评价，manifest只含train/test也支持，不要求补建验证集或重新训练。若测试者未参与选权重/调参，可用 `--selection-record /prepared/selection_record.json` 记录这一实际流程，并核对冻结权重及各预测的checkpoint SHA256。记录模板及步骤见[TRAINING_AUDIT.md](TRAINING_AUDIT.md)。程序不从文件名推断历史阶段，也不把该选模声明等同于完整身份/预训练成员核实。

`--independence-report`是另外的可选审计，要求预测和基线来源也对应同一身份/划分记录；旧输出无完整来源时可显式 `--allow-unverified-independence "实际缺失记录"`，或不提供该项并保留默认unverified。它不改变数值指标，但结果不能据此自动宣称完整独立性已由程序验证。若需核对新checkpoint当前运行阶段，可传 `--expected-prediction-phase l1/warmup/gan`（三选一）；GAN须记录实际对抗更新，旧权重没有记录时不猜测。

可选 `--distribution-diagnostics` 在现有评价检查全部成立后，额外输出distribution_diagnostics.json。每例、每方法在同一有效mask内报告物理均值/总体SD、p01/p50/p99、min/max、bias/MAE/RMSE、预测与参考SD比及NMSE；空分母返回null及原因。逐例精确分位数不合成全队列分位数，不把访视当独立人做统计。该入口不训练、不修改预测、不需要validation集，沿用固定权重的1773人test流程；不能从结果反算历史D损失或推导区域/小病灶效能。

## 数据与声明

第六项新增可选 `--roi-spec`，对既有固定预测补区域物理误差、参与者ROI均值的bias/LoA、Pearson和绝对一致性ICC(A,1)及配对bootstrap CI，不需要val或训练。默认不启用时，原有指标/汇总数值保留。须提供真实解剖label图及独立审核/来源；空标签、覆盖不足或未审核保留null与排除人数。输入规范、完整访视规则、两固定方法统计定义及可用命令见[ROI_EVALUATION.md](ROI_EVALUATION.md)，可从[roi_spec.example.json](roi_spec.example.json)复制语法后替换实际字典。

新实验从 [qc_manifest.example.csv](qc_manifest.example.csv) 填写输入QC字段，再加入预测列；[evaluation_manifest.example.csv](evaluation_manifest.example.csv) 展示基础评价列，本身不足以提供完整QC证据。两者兼容训练manifest的六个必需字段。评价列如下，QC额外字段见 [DATA_QC.md](DATA_QC.md)：

| 字段 | 含义 |
|---|---|
| `qsm_units` | 每行reference的真实单位，ppm或ppb；缺此列时在QC声明 `--qsm-units` 并继承，legacy评价则需显式给该参数 |
| `qsm_reference` | 实际QSM零参考定义，与 `--reference-id` 一致；不能用数值offset代替物理参考 |
| `space_id` | 已准备并QC的共同解剖空间，与 `--space-id` 一致 |
| `prediction_path` | 可选，已完成反归一化的物理QSM NIfTI；所选split内必须全有或全无 |
| `prediction_units` | 预测图单位，也可由其同名JSON的 `Units` 或 `--prediction-units` 声明；多来源冲突会拒绝 |
| `prediction_reference` | 预测图零参考，也可由JSON的 `ReferenceID` 或 `--prediction-reference-id` 声明；须与reference一致 |
| `prediction_reference_method`、`prediction_reference_region`、`prediction_reference_target_ppb` | 严格评价旧预测时与prediction_reference共同声明完整参考身份，或由JSON的 `DataReference` 提供 |
| `cohort`、`case_id` | 可选队列和采集标识；未给case_id时自动生成，写入解析后的manifest |
| `roi_labels_path`、`roi_label_source` | 可选ROI功能的真实整数解剖图路径及实际来源；空路径记录缺失，已声明但损坏/不匹配者拒绝 |
| `roi_review`、`roi_reviewer`、`roi_review_date` | 与图像QC分开的ROI审核声明，accepted需真实审核者及YYYY-MM-DD日期；其余状态不纳区域统计 |

示例中的 `actual_reference_id`、`prepared_common_space` 是需要替换的标签，不代表目前任何队列已经采用某种参考或完成配准。程序核对声明一致性，不替作者判断实际参考组织或解剖对齐是否正确。

同一case的SWI、reference、两个mask、预测图必须已在同一shape/affine及空间单位上。评价不配准、不重采样、不裁剪。`mask_path`为共同有效评价区，非空且包含于输入脑mask。任何方法都使用同一个评价mask。

NIfTI必须声明空间单位。旧图header为unknown时，已核实流程需在QC契约填写spatial_units_if_unknown并继承该假设；legacy评价可用 `--spatial-units mm` 显式声明。假设均留档，冲突会拒绝，不能据数值猜mm。HFEN按毫米计算，非正交/shear网格需先在上游准备。头信息一致仍不能证明真实解剖配准。

`--units` 只控制输出的共同物理单位，默认ppb。reference根据qsm_units转换，prediction根据自己的单位转换。无qsm_units列时，已核实流程继承QC报告的输入单位fallback，运行时冲突声明会拒绝；输出ppb不会把输入ppm重解释为ppb。不会用reference百分位、预测范围或SWI范围拟合强度校准，也不改变磁化率零参考。

## 拟合训练集平均图与线性基线

```bash
python -m swixqsm_minimal.fit_baselines \
  --manifest /prepared/qc_manifest.csv \
  --output /experiments/baselines/train_baselines.npz \
  --qc-report /qc/final/report.json \
  --units ppb \
  --reference-id actual_reference_id \
  --space-id prepared_common_space
```

拟合仅加载、核验、统计及内容hash `split=train` 的影像，val/test影像不参与。manifest读取仍检查全表必需路径是否存在。artifact含系数、支持域、单位、完整参考定义、空间声明、SWI预处理、训练身份/路径、数据hash及data_qc来源，使用无pickle的NPZ+JSON保存，拒绝覆盖。严格评价要求基线artifact也有已核实来源；旧artifact需按报告重拟合，或明确选择unverified评价。

三种基线如下：

- `zero`。物理值恒为0，评价时直接生成，不需拟合。与神经网络归一化值0不同，例如offset=0.02ppm时，网络归一化0对应20ppb，不能当作零图。
- `train_mean`。同一解剖位置的训练集参考QSM加权均值图。
- `linear`。每个体素独立拟合 `QSM(v)=intercept(v)+slope(v)*SWI_normalized(v)` 的带截距加权OLS。输入沿用本包仅依赖SWI脑mask的0.5/99.5百分位缩放；target只换单位，不裁剪或逐例归一化。输入无方差时该位置退化为训练均值。

同一训练受试者有多访视时，每访视权重为1/该人训练访视数，再在每个体素的有效访视内归一化。完整覆盖的人具有相同总权重，缺覆盖者只贡献其有效访视。

**平均图和逐体素回归需要预先准备的共同解剖空间。** 各人native图不能只补成同矩阵就当成同一解剖位置。所有train及待评价图的网格必须匹配artifact，声明space也必须一致。原生空间上的模型预测可单独评价；若要与空间基线比较，应先把所有方法按同一预定流程准备到共同评价空间。

训练未覆盖的体素不提供已验证基线值。评价mask只要包含此类体素，程序拒绝运行，不自动取交集。应在研究方案中确定共同有效域，不能看测试误差后再缩小mask。程序还检查artifact训练subject ID/输入路径与所选评价行不重叠；身份别名、改名副本和历史模型暴露仍需外部审计。

## 在相同测试者上评价

```bash
python -m swixqsm_minimal.evaluate \
  --manifest /prepared/qc_manifest.csv \
  --split test \
  --qc-report /qc/final/report.json \
  --baselines /experiments/baselines/train_baselines.npz \
  --output /experiments/evaluation/model_and_baselines \
  --units ppb \
  --reference-id actual_reference_id \
  --space-id prepared_common_space \
  --data-range 500 \
  --save-predictions
```

500ppb只是参数语法示例，不是自动推荐的强度范围。PSNR/SSIM的data_range应在训练数据或研究方案中预先确定，并在所有模型/队列间保持一致；计算前不按该范围裁剪。改用ppm输出时，等价范围为0.5ppm。未给data_range时，其余误差仍计算，PSNR/SSIM为null并记录原因，不从预测或测试reference自动估计范围。

没有prediction_path时，只评价零图与提供的训练基线。只评价已有模型与零图时，可以不传 `--baselines`，不要求跨受试者共同atlas空间，但同一case各图仍需正确配准、同网格和同参考。

严格评价核对完整参考身份 `{id,method,region,target_ppb}`，同名id但方法、区域或目标不同仍拒绝。经已核实训练的新checkpoint推理会检查训练契约的方向/spacing并继承参考，预测JSON带DataReference。旧预测若缺该对象，须根据真实来源补齐CSV中的prediction_reference、prediction_reference_method、prediction_reference_region、prediction_reference_target_ppb；仅给ReferenceID或 `--prediction-reference-id` 不足以宣称已核实。记录声明不会替预测重新选参考。若JSON声明SpaceID，也会交叉检查。

旧checkpoint的推理需显式 `--allow-unverified-data "实际原因"`，且仍需可识别的normalization元数据。严格评价拒绝明确标记unverified、failed、pending、rejected或未知QC状态的预测；如为历史排查，应将评价的 `--qc-report` 替换为 `--allow-unverified-data "实际原因"`，保留未核实状态。原预测的PredictionDataQC及来源与EvaluationDataQC分别留档，未记录的旧来源标为not_recorded，不会因评价另存而提升为verified。既有模型的checkpoint是否接触过测试者仍需外部审计。

## 指标定义

以下reference为y，prediction为p，M为有效mask。单位统一后按float64计算，不受脑外padding数值影响。

| 指标 | 定义/单位 |
|---|---|
| MAE | mean_M abs(p−y)，ppm或ppb |
| RMSE / MSE | sqrt(mean_M (p−y)^2) / mean_M (p−y)^2 |
| bias | mean_M(p−y)，方向为预测减参考 |
| NMSE | sum_M (p−y)^2 / sum_M y^2；非零reference能量时物理零图等于1 |
| PSNR | 20log10(data_range)−10log10(MSE)，dB |
| SSIM | axis2逐轴位二维均匀窗口，默认7×7，sample covariance、K1=0.01、K2=0.03；只平均完整窗口都在mask内的中心，保持二维邻域 |
| HFEN RMSE | 3D物理LoG误差的RMSE，默认sigma=1.5mm、4sigma有限支持；单位为ppm/mm²或ppb/mm² |
| relative HFEN | norm(LoG(p−y))/norm(LoG(y))，无量纲 |

HFEN采用毫米到各轴体素尺度的换算、按spacing平方缩放的二阶导，并显式校正离散核零直流响应，避免厚层/小sigma时把纯固定偏置计为高频误差。此定义和实现参数会留档，不可直接与旧脚本中另一种HFEN数值混用。滤波只在完整核支持都处于mask内的中心评价，因此SSIM/HFEN的有效中心数与点误差体素数分别报告。

NMSE参考能量为0、relative HFEN参考高频能量为0、空间窗口支持不足、未提供data_range等情况返回null及明确status。完美预测的PSNR表示为null + status=infinite，不写非法JSON Infinity。汇总保留每项定义/未定义人数与原因；有限值均值不是包含无限值的总体PSNR均值。

## 输出与统计单位

- `per_case.csv`。每次采集各方法的指标、单位、参考声明、有效体素/窗口数、状态，以及该case的spacing和核参数。
- `per_subject.csv`。先在同一人内将各访视的误差和参考能量等充分统计量合并，再计算该人的MAE/RMSE/NMSE等；不把访视当独立受试者。
- `summary.json`。各方法整体/分队列的受试者等权均值与样本SD、定义/未定义人数及原因、命令配置、版本、manifest/artifact/QC报告hash、完整参考身份、独立性/固定权重选模来源和指标约定。单人SD为null。
- `manifest_resolved.json`。实际所选split和解析后的路径/采集ID。
- 可选 `distribution_diagnostics.json`。每例/方法的物理输出分布及误差、选模记录和预测来源；同名输出只描述当前评价，不填造旧权重的TrainingObjective/TrainingDiagnostics。
- 可选 `roi_per_case.csv`、`roi_per_subject.csv`、`roi_summary.csv`、`roi_agreement.json`。每区域/方法/队列的物理误差、配对均值、覆盖/排除原因、参与者n及一致性/CI；含独立的标签/spec来源hash，不将其当作解剖QC通过。
- 可选 `predictions/` 与 `saved_predictions.csv`。物理QSM float32及JSON，记录单位、DataReference、原预测QC/来源、评价QC、评价mask路径和方法，便于图像QC。脑外/未训练覆盖的0是padding，评价有效范围以mask为准。

已有输出目录会拒绝覆盖。评价先在临时目录完成，全部成功后才交付结果目录；报错不会留下看似完整的半套评价。

基础指标提供统一的图像/体素级评价与非神经网络基线，可选ROI功能增加区域定量一致性；这些均不提供组织学或临床有效性结论，也不能据此追认历史Table2。真实受试者结果需要在QC合格且来源齐备的数据上运行后报告。

## 数值验收

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m unittest discover -s swixqsm_minimal/test -t . -v
```

新增测试涵盖identity、物理零图、偏置/单位缩放、脑外扰动、SSIM空间邻域、HFEN毫米尺度/厚层常数偏置、空分母、训练集均值/已知逐体素线性关系、混合单位、测试数据不参与拟合、重复访视权重、训练覆盖不足/网格/声明/身份冲突，以及QC报告绑定、完整参考定义、legacy状态保留和训练→推理→基线→评价接入。测试只使用临时合成数据。
