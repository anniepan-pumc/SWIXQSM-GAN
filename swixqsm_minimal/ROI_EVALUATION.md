# 冻结模型的ROI区域定量一致性

2026-10-17。第六项在现有物理评价上增加ROI提取、区域误差和参与者层面的一致性统计，不修改网络、既有权重或预测，不运行训练。没有提供真实ROI标签时，不生成虚构标注，也不报告真实队列结果。

作者已明确held-out test participants预先独立划为test，训练没有val，且仅在权重固定后评价。本入口沿用这一事实，不要求补建validation集或重新训练。固定权重选择记录见 [TRAINING_AUDIT.md](TRAINING_AUDIT.md)。

## 2026-10-19补充：先算候选ROI、后补审核

默认 `--roi-review-policy accepted` 保持原有严格规则：只有真实记录为accepted、审核者及日期完整的ROI进入数值统计；原有ppm/ppb评价和哈希门槛不变。若作者明确延后审核，可另行选择 `--roi-review-policy deferred`，同时必须给非空 `--allow-unverified-data` 原因，不能与 `--qc-report` 合用。此时pending、not_recorded、unreviewed及缺失审核记录的真实整数标签可以作为研究候选计算；rejected仍排除，缺标签、缺来源、缺组成label及不足覆盖仍保留null/原因。它不会补造accepted、审核者或日期。原记录为accepted却缺合法审核者/日期仍视为配置错误。

`--units stored --qsm-units stored` 只允许stored与stored一致，反归一化后保留原NIfTI解码数值；不转换ppm/ppb、不重新参考、不恢复未知物理单位。stored ROI只在上述显式deferred模式下可用，解剖训练平均/线性基线仍不支持stored。`reference-id`为未核实的数值参考声明时，应明确命名，例如 `unverified_original_reference`；它不是已经核实的组织零参考。

可用独立 `--roi-manifest /prepared/roi_test.csv` 附加标签，避免改变冻结evaluation manifest与checkpoint/独立性报告的身份哈希。CSV必须对当前所选split的**全部case_id一一对应**，缺标签者保留行并留空路径，不得通过删行选择好病例。只接受 `case_id,roi_labels_path,roi_label_source,roi_review,roi_reviewer,roi_review_date`，另允许 `roi_review_status` 作为roi_review别名；两者同时存在须一致。输入影像、身份、split、单位及prediction列不可经此CSV修改。相对标签路径以该ROI CSV目录为基准。覆盖信息只交给ROI评价器，核心评价rows不变；CSV从同一份bytes解析/计算SHA，并在完成时复核未变。CSV与spec哈希写入summary/provenance。

```bash
python -m swixqsm_minimal.evaluate \
  --manifest /prepared/frozen_test_predictions/evaluation_manifest.csv \
  --split test --output /results/model_A_test_roi_candidate \
  --units stored --qsm-units stored \
  --reference-id unverified_original_reference \
  --allow-unverified-data "单位、零参考及ROI/配准人工审核延后；仅研究候选数值比较" \
  --independence-report /prepared/independence/report.json \
  --roi-manifest /prepared/roi_test.csv --roi-spec /prepared/actual_nuclear_roi_spec.json \
  --roi-review-policy deferred --roi-bootstrap-reps 2000
```

区域名可在同一spec中明确写RN_L、RN_R、RN_bilateral、SN_L、SN_R及whole等，每项使用实际字典中的整数labels；工具不猜核团编号，whole也不自动等于全脑。左右合并时每个组成label仍须出现。工具不生成标签、不注册/重采样、不改模型或权重。

CSV/JSON保留原 `roi_review_status`，并写入 `roi_review_policy`、`roi_qc_status`、`QCStatus`、`UnitStatus`、`value_space`、`research_only`和 `formal_accuracy_claim=false`。deferred的 `status=ok` **仅表示数值与预设支持域条件满足**，不是通过解剖审核。stored的UnitStatus为 `unverified_original_storage`，ValueSpace为 `restored_stored_values`。ICC和r为无量纲；bias/LoA和误差仍为stored原尺度，不能称为ppb误差或物理校准。所有人数不足/常量/零能量及bootstrap不足情况沿用原null/status政策。

`difference`与 `absolute_mean_difference` 是ROI均值的差及其绝对值；`mae/rmse/bias/nmse` 是同ROI支持域内的体素误差。汇总分别保存 `roi_mean_participant_summary` 与 `voxel_error_participant_summary`，CSV对应 `roi_mean_*_participant_mean/sd` 与 `voxel_*_participant_mean/sd`，禁止将两类MAE混写。重复访视先在同参与者内池化，所有预期访视符合条件后才入该ROI的参与者统计。

每case/region记录实际reference数组、评价网格、共同mask与ROI交集的SHA（`reference_array_sha256`、`evaluation_grid_sha256`、`evaluation_support_sha256`、`roi_support_sha256`），另保留labels/spec/ROI清单哈希。reference数组统一以实际评价值的little-endian float64 C-order bytes绑定，mask以bool C-order bytes绑定，shape/affine另由grid hash绑定；它们不是原文件SHA，也不证明解剖正确。跨模型比较须核对同人、同标签/spec、同reference、同网格和同支持域，不以误差结果挑选标签或病例。

## 为什么优先补ROI

直接核对两份原Word后，相关条目为：

| 评审条目 | 要求 |
|---|---|
| R1 major 6 | 报mean bias、limits of agreement及区域ppb误差；区分相关和绝对一致性。 |
| R1 minor 4 | 解释Shunyi 48/87的ROI入选原因，以及与未入选者的差异。 |
| R2 major 6 | 增加red nucleus（RN）、substantia nigra（SN）、dentate nucleus（DN）及已发现微出血的证据。 |
| R3 major 3 | 增加Bland–Altman、ICC及CI、各队列零参考，并减弱ppb校准等过强表述。 |
| R3 major 9 | 将ROI一致性扩到所有队列，增加RN/SN/DN；另要求reference临床关联、明确cohort/n及读片设计。 |

原文ROI方法为T1-MPRAGE的FreeSurfer recon-all分割、注册到QSM空间，左右同名区域按体素数加权合并。§3.3只报告Shunyi **48人**的putamen、caudate、pallidum、thalamus、amygdala及whole-brain分析，没有报告UKBheld-out test set、CSVD83人或全部Shunyi87人的ROI结果，也没有已完成RN/SN/DN结果。

原文putamen r=0.880、ICC=0.488；caudate r=0.632、ICC=0.564；whole-brain r=−0.275、ICC=−0.019。r较高不能替代绝对一致性，也不能仅按r把putamen称为ICC最好的区域。原文OLS/CV R²不替代Bland–Altman或ICC CI；本工具不会用test回归拟合去校正预测，消除原本应报告的偏差。

前五项分别提供物理单位与简单基线、数据QC、身份/权重来源、有效patch与诊断、采样/I/O工具。它们是ROI评价的基础，整体mask的bias、分布、HFEN及合成数值验收都不能自动证明区域一致性。真实标签、病例覆盖及物理参考仍需作者提供。

## 准备真实label图与区域定义

在既有评价manifest中增加以下列。路径可沿用manifest的相对路径约定；其余四角色、预测、单位、完整QSM参考及来源要求见 [EVALUATION.md](EVALUATION.md)。

| 列 | 要求 |
|---|---|
| `roi_labels_path` | 已在该case评价网格上的真实3D整数label图；可空，但明确记为缺失。 |
| `roi_label_source` | 实际atlas/分割版本、处理与来源记录，不以“FreeSurfer”或一个文件名猜测实际标签。 |
| `roi_review` | accepted、pending、rejected，或缺失；只有真实完成审核后才可填accepted。 |
| `roi_reviewer` | accepted时须非空，记录实际审核者。 |
| `roi_review_date` | accepted时须为合法YYYY-MM-DD日期。 |

ROI图必须与该case现有reference/预测/共同mask同网格、同空间单位，有限、3D、非负整数；0为背景。工具不自动配准、重采样、四舍五入浮点概率图或猜测atlas。标签应来自独立解剖资料，不能根据生成QSM的结果挑选ROI来取得更好指标。相同affine也不能独立证明组织已准确配准，人工审核不能由hash自动代替。

配套 `--roi-spec` JSON描述统一label语义，可从[roi_spec.example.json](roi_spec.example.json)复制格式后按实际字典替换。例如：

```json
{
  "schema": "swixqsm.roi_spec",
  "version": 1,
  "id": "REPLACE_WITH_ACTUAL_ROI_SPEC_VERSION",
  "definition": "REPLACE_WITH_ACTUAL_LABEL_DICTIONARY_VERSION_AND_BILATERAL_DEFINITION",
  "label_source": "REPLACE_WITH_ACTUAL_ATLAS_AND_SEGMENTATION_SOURCE",
  "regions": [
    {"name": "example_bilateral_region", "labels": [101, 201]}
  ],
  "min_coverage": 1,
  "min_voxels": 1
}
```

101/201只是格式示例，不能当作本研究任何核团的真实label编号。须按实际字典替换名称、ID、来源和labels；不能以示例凑出RN/SN/DN。region名称须唯一，每个region可合并多个不重复的正整数label（最大2³¹−1）；这些label都是该区域的**必需组成**。不同region允许重叠，但它们的测量不是独立重复，不能据此扩大样本量。

例如双侧region=[左、右]时，只有左侧label出现并不足以代表双侧。输出每个组成label的 `label_voxel_counts`和 `missing_labels`；缺一部分记 `missing_component_label`，全部未出现记 `empty_label`，都不进入该区域有效参与者统计。若要分别报告左右，应明确设为两个region，而非缺一侧时静默退化。

### 共同有效域与阈值

每个region使用 `ROI ∩ 共同有效评价mask`，所有方法用同一支持。报告原label体素数 `n_label_voxels`、交集体素数 `n_valid_voxels`及coverage=n_valid/n_label。默认min_coverage=1、min_voxels=1；这只是明确的完整覆盖门槛及语法起点，不代表1个体素足以建立真实核团的可靠性。min_coverage须在(0,1]内，min_voxels须为正整数；阈值必须按研究方案预先确定，不能看test误差后放宽。

coverage分母只有当前labelmap内的体素；100%不能证明FOV外的解剖结构已经覆盖，也不能证明标签正确或真实配准通过。厚层、小ROI、标签破碎及采集截断还需真实QC。

空标签路径、缺来源、未accepted或不满足预定覆盖/体素阈值时，保留该case/region的null、status与原因，不补0、不悄悄只留下好病例。主要状态包括missing_labelmap、missing_label_source、review_not_accepted、empty_label、missing_component_label、insufficient_voxels、insufficient_coverage。已声明路径却文件损坏/不存在、网格或label数值不合法，以及accepted却缺合法审核者/日期等配置错误会拒绝本次评价，不能伪装成普通缺失。有效物理QSM=0仍参与计算；排除背景label=0不等于排除磁化率数值0。

输入数据QC通过不会自动令ROI审核通过。ROI来源、文件与spec的hash用于绑定使用了什么，不能证明atlas、身份或人工审核声明是真实的。新增roi列会改变既有manifest元数据，旧QC/独立性报告可能因此失配；应对实际更新后的manifest重新生成只读报告，保留已有严格门槛，不通过绕过检查迁就旧hash，也不需要重训模型。

## 统计单位与指标

每case/方法/region提取reference与prediction的物理均值、差值、pair_mean，以及区域内voxel MAE/RMSE/bias/NMSE。方向始终为prediction−reference；pair_mean=(prediction+reference)/2供Bland–Altman/scatter使用。

**区域均值之差与区域内voxel误差不同。** 一个区域内正负误差可能互相抵消，不能因区域均值接近就宣称每voxel准确；两组量都保留。NMSE沿用未中心化reference能量定义，零分母为null及原因。

### 多访视与身份

对同一method/participant/region，manifest中该participant所有预期访视都须ok才纳入参与者一致性统计；任一缺失或失败则记 `incomplete_visits`，不能只留下较好或唯一可用访视。完整者合并各访视的体素充分统计量，再以合并有效ROI体素计算reference/prediction均值；参与者作为n仅计一次。

这一估计对象是**上述多访视体素池化测量的一致性**，不是单scan准确度或纵向重复性；较大有效体积的访视在池化均值中贡献更多。若研究目标是特定时间点，应在预先声明的manifest选取规则中确定，不能事后按结果挑访视。

如所选split有任一global_subject_id，就须在该次ROI评价全部行完整提供，且subject_id/global_subject_id一对一；不提供时明确以声明的subject_id为参与者。一个subject_id不能跨cohort，字符串映射检查仍不证明真实身份。overall及by-cohort都按region/method完整列出有效n、缺失/失败/不完整人数和原因；详细原因为case状态和participant的visit_status_counts。不能用混合队列的高相关掩盖单队列偏差。

### 跨参与者的一致性

| 统计 | 定义及解释 |
|---|---|
| Mean bias | 参与者ROI均值差d=prediction−reference的均值，物理单位同输出。 |
| Bland–Altman LoA | mean(d)±1.96×sample SD(d)，样本SD使用ddof=1。是差值分布界限，不是mean bias的置信区间。 |
| Pearson r | 参与者ROI均值的线性关联；不是绝对一致性。常量/人数不足时返回未定义。 |
| ICC(A,1) | 两个固定方法的single-measurement绝对一致性，计入方法间系统性偏差；不等于只看一致趋势的consistency ICC。负值保留，不截到0。 |
| 95%CI | 以成对participant为单位的percentile bootstrap；reference和prediction的同人配对一起重抽。 |

ICC使用 `(MSR−MSE)/(MSR+(k−1)MSE+(k/n)(MSC−MSE))`，k=2，MSR为参与者行均方、MSC为方法列均方、MSE为两因素加性模型残差均方。本工具要求n≥3才报告ICC，这是保守报告政策，不是数学最小样本数。Pearson在任一方法常量时未定义；两列相同常量的ICC零分母为null，两列各常量但数值不同的ICC按公式为0，明确不具有参与者区分能力，不能视作可靠性的正面证据。

LoA的正态差值、近似同方差等假设尚未被程序自动证明，应结合difference对pair_mean的分布检查。存在比例偏差或异方差时不能把一个全局LoA当作任意值域都适用。均值bias接近0也不能掩盖很宽的LoA。

bootstrap使用局部随机生成器与记录的seed，不把voxel、左右region或不同方法拆成独立人。本工具n<3不给任何bootstrap CI上下界，是保守报告政策；n=1的bias、n=2的SD/LoA仍可保留为描述点值。已定义点值的区间标 `insufficient_n_for_bootstrap_policy`，点值本身未定义时优先保留该原因。

每个指标仍公开实际有效/未定义rep数与原因，未定义rep不重新抽到“有效”为止。若只有部分rep可定义，CI标partial_defined，是条件于可定义rep的摘要，不能称为全部重抽都有效或保证名义95%覆盖。少于两个可定义rep不给区间；点估计未定义时也不因部分rep可定义而补出CI。人数不足、常量、零分母等保留null/status，不凭epsilon或截断制造ICC/CI。

本次区间是每region/metric的pointwise CI，不是多ROI同时置信区间，也不是全家族错误控制。工具不开展新的临床假设检验或自动FDR；原稿reference临床关联及多重比较方案仍是单独任务。

统计定义依据 [Koo与Li的ICC报告原文](https://doi.org/10.3348/kjr.2016.17.5.623)、[Bland与Altman的方法原文](https://www-users.york.ac.uk/~mb55/meas/ba.htm)，成对percentile bootstrap的约定可对照 [SciPy官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)。本实现的bootstrap区间不能被描述为ICC解析F区间，或未实现的BCa区间。

## 调用与输出

默认不启用ROI，原有图像指标与输出不因增加此功能改变。只有传入 `--roi-spec`才增加ROI评价，`--roi-bootstrap-reps`默认2000、`--roi-bootstrap-seed`默认42；reps=0只关闭bootstrap CI，仍计算可定义的点估计。

在CVMed根目录、使用CVMed环境执行。例如在已有冻结模型评价命令上增加：

```bash
python -m swixqsm_minimal.evaluate \
  --manifest /prepared/frozen_evaluation_with_actual_roi.csv --split test \
  --qc-report /qc/with_actual_roi/report.json \
  --allow-unverified-independence "历史初始化成员尚未按本包完整审计；固定权重选择记录另附" \
  --selection-record /records/actual_model_selection.json \
  --roi-spec /prepared/actual_roi_spec.json \
  --roi-bootstrap-reps 2000 --roi-bootstrap-seed 42 \
  --output /experiments/evaluation/frozen_roi \
  --units ppb --reference-id actual_reference_id
```

示例路径均须指向真实来源；不能把此前外部样例的QSM_mask当解剖ROI，不能用未知QSM单位或缺独立输入mask的数据凑manifest。完整参考和预测来源仍按 [EVALUATION.md](EVALUATION.md)核对。现有selection-record核验冻结权重及预测hash，不等于自动核实全部历史身份；显式未核实来源仍如实保留。

通过既有检查后增加：

- `roi_per_case.csv`：每case/method/region的审核/覆盖/状态、label组成、物理均值与voxel误差。
- `roi_per_subject.csv`：所有预期访视的完整性、参与者池化值和误差；不把缺失访视静默丢掉。
- `roi_summary.csv`：overall及分cohort的每region/method人数、区域误差与一致性结果，含未定义情况。
- `roi_agreement.json`：完整统计定义、spec/输入来源、阈值、缺失说明、bootstrap状态与CI，便于追溯。

这些输出提供Bland–Altman/scatter所需配对量，不表示已经为真实队列完成图表或解释。现有输出目录仍拒绝覆盖，全部评价经临时目录完成后才发布；ROI输入/统计错误不能留下看似完整的半套新结果。

## 完成范围与剩余路线

本次只用临时已知合成数据检查接口、数值与状态。真实ROI尚未提供，不报告UKB/Shunyi/CSVD的新增结果，不声称RN/SN/DN、微出血或钙化已验证，也不把代码完成当作审稿问题已全部解决。

建议继续按以下顺序处理已有冻结预测：

1. 先复核原48人的真实标签、单位参考和配对表，补bias/LoA/ICC CI；再检查UKBheld-out test set、Shunyi87、CSVD83可用标签及真实排除流程。48/87选择与未入选者差异需原始记录和资料，不能由工具猜测。
2. 有真实适用label后扩RN/SN/DN；常规FreeSurfer名称不保证这些结构已分割。小结构、部分容积和覆盖需单独解释。
3. 做已有协议/方向分层及多平面连续性展示；当前配准后affine不自动等于原始采集B0方向，需真实元数据。
4. 有临床表时补同参与者、同region的reference临床关联；有真实病灶标注/CT或reference确认及读片设计时补lesion与读者评价，资料不足则限缩原文主张。
5. L1-only、标准pix2pix、lambda/对抗消融或新模型如无已有可用权重，需要另获用户允许开展新训练；本轮不将重训作为ROI评价前提，也不宣称这些对照已完成。

主文方法、cohort/n、48人选择、图号/轴单位、参考声明、临床与伦理内容仍需按真实资料修改；它们没有因为新增ROI代码而自动完成。
