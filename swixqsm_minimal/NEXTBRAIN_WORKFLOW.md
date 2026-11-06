# NextBrain 核团定量与八模型比较

本流程不训练模型，不修改模型结构。`stored` 表示原存储数值，不能写成 ppm/ppb；数值零图也不代表零磁化率。解剖、配准和临床访视 QC 待补。计算成功及哈希吻合均不等于 QC 接受。

## 一、两种评价对象

1. `evaluate` + `compare_roi_models`：reference、prediction、ROI **在同一评价网格**。输出 ROI 均值绝对差、体素 MAE/RMSE/bias/NMSE，以及按受试者计算的 ICC(A,1)、Pearson r、Bland–Altman bias/LoA 和可定义的配对 bootstrap CI。各 ROI、各队列分别计算；左右侧不充当独立受试者。
2. `evaluate_native_nuclei`：reference 在 GRE/QSM 原生网格，prediction 在 SWI 原生网格，各自用配准后的核团取均值。**只比较 ROI 均值，不计算跨网格体素 MAE、PSNR、SSIM 或 HFEN。** 小规模、待 QC 试跑的 CI 默认关闭，ICC 在人数不足时留空。该输出不能与同格体素指标混成一个性能排名。

固定 NextBrain AllenAtlasLUT 定义：红核 RN=385；黑质网状部 SNr=310，致密部 SNc=352，整块黑质 SN=310∪352；齿状核 DN=721。每个核团分别输出 L/R/bilateral，共 15 个 ROI。whole SN 必须有两个亚核，bilateral 必须有双侧；缺失分量、FOV 缺失、可靠域覆盖不足均显式保留并排除对应主汇总。标签定义不能套用 FreeSurfer LUT。

## 二、现有外部试跑的读取与重算

本次病例固定为顺义5例和CSVD5例。已有历史预测绑定 `gen_new_19.pt` 的 SHA256 `83c4e0b7aa451b499a59c03836e0d3ce92d7715352b90313afa7a60e044a955c`，输入 profile 固定，训练时实际 normalization 仍未证实。因此历史单模型结果与未来八模型新训练结果分别记录。

本地继续处理入口在工作区 `outputs/nextbrain_continue_20261019`：

- `segmentation_manifest.csv/json`：10例双侧生成状态及实际来源，包含复用的旧完整结果。
- `processing_status.csv/json`：配准和定量进度，缺失病例保留。
- `latest_quantification.json`：当前有效量化目录；只有 `complete=true` 才是10例全部完成。
- `quant_10_cases_v3/native_roi_per_case.csv`：最终预计300行=10人×15ROI×(历史模型+零图)。这是模型/ROI行数，不是300人。
- `quant_10_cases_v3/native_roi_summary.csv`：分队列、分ROI及左右侧汇总，含定义人数、排除人数、原尺度 ROI 均值误差与数值零图比较；当前只有一个历史模型。
- `quant_10_cases_v3/report.json`：原始文件与预测哈希、固定权重、单位/QC限制；没有推断缺失的物理换算。

以下为从项目根目录重算已完成10例的命令；仍在运行时应使用 `latest_quantification.json` 所指的实际完成清单：

```bash
python -m swixqsm_minimal.evaluate_native_nuclei \
  --cases outputs/nextbrain_continue_20261019/quant_10_cases_v3.json \
  --source-root "$PWD" \
  --output outputs/nextbrain_continue_20261019/recomputed_native_roi_v1 \
  --allow-unverified-data 'QC后补；reference及prediction均为未核实原存储尺度，参考零点未核实'
```

`cases` 是 JSON 数组，每项有 `case_id/cohort/subject_id/registration_report/prediction_provenance`，可附 `model/expected_checkpoint_sha256`。同一病例的所有模型必须共用完全相同的 reference、配准和标签支持；不同模型不能贴标签复用同一权重。预测记录必须包含 prediction 路径和 SHA256、checkpoint SHA256、cohort 和 subject_id。不得把8模型缺失的预测用历史模型复制补齐。

## 三、接入服务器同格八模型评价

ROI 清单与原冻结评价清单独立，**不要往已冻结的训练/测试清单补列**。每个选定 split 的 ROI CSV 必须覆盖该 split 每一例，`case_id` 一一对应；没有标签者保留空路径，不删除人。最少列：

```csv
case_id,roi_labels_path,roi_label_source,roi_review,roi_reviewer,roi_review_date
example,labels/example_nuclei.nii.gz,真实分割与配准来源,pending,,
missing_example,,,pending,,
```

可从哈希绑定、scanner-identity v2 注册完成记录生成 label/spec/ROI CSV：

```bash
python -m swixqsm_minimal.prepare_nuclei_roi \
  --cases /absolute/path/registration_cases.json \
  --evaluation-manifest /absolute/path/frozen_manifest.csv \
  --split test --target gre --grid-column qsm_path \
  --source-root /absolute/path/project \
  --output /absolute/path/new_roi_binding_test
```

该工具只重编码已有整数标签：右侧目标ID增加subjT000，左侧ID保留，其余标签置零。左右原图先按真实仿射映射到目标网格，**不能按数组位置直接合并不同裁剪网格**。工具不做新配准、不改 header、不重采样强度；身份、实际目标文件 SHA、空间单位和网格不一致会拒绝。SWI 网格单独绑定用 `--target swi --grid-column swi_path`。生成的 overlay 路径相对 overlay CSV，移动标签目录时应保持相对结构。

在最新版服务器包 `settings.json` 加入以下字段，并填写真实服务器路径：

```json
"roi_evaluation": {
  "enabled": true,
  "spec": "/absolute/path/new_roi_binding_test/roi_spec.json",
  "manifests": {
    "test": "/absolute/path/new_roi_binding_test/roi_manifest.csv"
  },
  "review_policy": "deferred",
  "bootstrap_reps": 2000,
  "bootstrap_seed": 42
}
```

只有准备了 val 的真实标签和完整 ROI CSV 后才添加 `manifests.val`；没有标签的 split 不产生虚构结果。该设置要求主流程本身 `quality_mode=deferred` 且有非空的未核实原因。八模型常规命令保持：

```bash
bash run.sh evaluate-test --model reviewer_suite --label final
bash run.sh compare-test --model reviewer_suite --label final
```

启用 ROI 后，`compare-test` 先生成原全图4张CSV，再生成 `comparison_roi/test_final/` 下的4张核团比较 CSV：

- `all_models_roi_per_case.csv`
- `all_models_roi_per_subject.csv`
- `roi_model_metrics_long.csv`
- `roi_model_metrics_wide.csv`

只需重做核团汇总且评价已完成时用 `bash run.sh compare-roi-test --model reviewer_suite --label final`。汇总要求8个真实评价目录，严格比较case/region/ROI支持域/reference及网格哈希，不默默删掉缺少的模型或病例。

**已有旧包训练/预测结果时**：保留原包和冻结产物；不要覆盖原训练代码后绕过 `prepared` 源码审计。可用新包的独立 `evaluate` 入口读取旧包生成的冻结 prediction manifest，新增评价目录；传递原有 independence report 和质量原因，再运行比较器。不要重写原训练/测试划分，也不要改权重或用测试结果选权重。

```bash
python -m swixqsm_minimal.evaluate \
  --manifest /absolute/path/predictions/MODEL_test_final/evaluation_manifest.csv \
  --output /absolute/path/evaluation/MODEL_test_nuclei_v1 \
  --split test --units stored --reference-id unverified_original_reference \
  --prediction-name MODEL --expected-prediction-phase gan \
  --data-range ACTUAL_FIXED_TRAIN_ONLY_RANGE \
  --independence-report /absolute/path/prepared/independence/report.json \
  --allow-unverified-data 'QC和单位核实后补；原存储尺度' \
  --roi-spec /absolute/path/roi_spec.json \
  --roi-manifest /absolute/path/roi_manifest.csv \
  --roi-review-policy deferred
```

L1 模型用 `--expected-prediction-phase l1`；`ACTUAL_FIXED_TRAIN_ONLY_RANGE` 取原运行的 `metric_range.json.data_range`，不能在测试集拟合。为8个模型各自评价后，用 `python -m swixqsm_minimal.compare_roi_models --output NEW_DIR --evaluation NAME=DIR ...` 传入8个真实目录。

## 四、外部顺义/CSVD：新八模型权重就绪后，无需受限队列影像即可推理

新入口只支持含实际 normalization、输出激活和训练阶段记录的**新训练 checkpoint**。历史无预处理记录的 `gen_new_19.pt` 不适用。不会为了得到8行比较而把旧权重套用新的 normalization。

准备 CSV：

```csv
model,checkpoint_path,expected_phase
swixqsm_gan,/absolute/run/models/swixqsm_gan/best_gan.pt,gan
fcn_gan_l1,/absolute/run/models/fcn_gan_l1/best_gan.pt,gan
plain_unet,/absolute/run/models/plain_unet/best_l1.pt,l1
pix2pix,/absolute/run/models/pix2pix/best_gan.pt,gan
swixqsm_l1_ablation,/absolute/run/models/swixqsm_l1_ablation/best_l1.pt,l1
swixqsm_lambda1,/absolute/run/models/swixqsm_lambda1/best_gan.pt,gan
swixqsm_lambda10,/absolute/run/models/swixqsm_lambda10/best_gan.pt,gan
swixqsm_lambda100,/absolute/run/models/swixqsm_lambda100/best_gan.pt,gan
```

```bash
python -m swixqsm_minimal.predict_native_nuclei \
  --cases /absolute/path/quant_10_cases_v3.json \
  --checkpoints /absolute/path/eight_checkpoints.csv \
  --source-root /absolute/path/project \
  --output /absolute/path/external_eight_models_nuclei_v1 \
  --device cuda --num-threads 4 \
  --allow-unverified-data '外部顺义及CSVD试跑；QC后补，原存储尺度和参考零点未核实'
```

按模型加载一次权重，逐例 SWI 推理；brain mask 使用同病例 SuperSynth T1 brain 到 SWI 的固定映射，显式标记其配准待 QC，推理不读取 reference QSM 强度。按 checkpoint 记录反归一化一次，无每例 reference 拟合、clipping 或强度校准。模型名称必须匹配 checkpoint 的真实 recipe，checkpoint bytes 必须互不相同。当前 native reference 单位未核实，因此该入口要求 stored checkpoint；physical checkpoint 无法靠猜测转换到未知 reference 单位。

输出：`metrics/native_roi_per_case.csv` 和 `metrics/native_roi_summary.csv`，包含8模型+独立数值零图；10例全部可定义时逐例表1350行=10×15×9。各队列各5人，不能把15个ROI当成独立人数。缺失/部分 ROI 和 QC 待审核状态仍保留。失败写 `run_status.json`，保留实际已生成的预测，不把部分成功标为全部完成。

这一路线仅检验新模型在外部数据的候选数值表现；临床访视关系、外部训练重叠、配准、分割边界、磁化率单位和参考零点仍需要后续审核，不能据此直接作临床或物理准确度结论。
