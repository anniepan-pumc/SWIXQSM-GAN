# 配准、方向/间距与QSM参考的数据QC

2026-10-17新增，入口为 `python -m swixqsm_minimal.qc`。检查既有配对与元数据，不执行配准、重采样、轴重排、重新参考或模型结构修改。

## 工作流程

1. 根据真实训练协议复制并填写 `data_contract.example.json`。允许范围应预先确定，不能根据测试表现反复放宽来取得通过。
2. 复制 `qc_manifest.example.csv`，或在现有manifest补字段。示例人工审核为pending，不虚构已审阅结论。
3. 生成报告及预览，完成真实逐例审核，记录审核人/日期，再用新输出目录生成最终报告。
4. 训练、基线拟合和评价使用 `--qc-report /qc/final/report.json`。启动时只重查所消费case，不在每切片/epoch重新hash。

QC与独立性分开记录。新训练另需身份/初始化审计报告或显式未核实理由，见[TRAINING_AUDIT.md](TRAINING_AUDIT.md)；现有冻结模型的评价不要求补建validation或重新训练。

```bash
python -m swixqsm_minimal.qc \
  --manifest /prepared/qc_manifest.csv \
  --contract /prepared/data_contract.json \
  --output /qc/initial \
  --split train val test \
  --preview-limit 12
```

含失败/pending/rejected病例时仍生成报告和预览，退出码为2；全部通过为0。选定病例的缺失/损坏图像和错误单位记入失败报告；manifest结构、身份划分或契约本身无效则直接拒绝。preview-limit仅限制展示数量，所有case仍会检查并记录审核状态，不会给未看过的病例自动填accepted。可以按split分批检查或增加展示上限；配准审核应在完整三维图像中进行。

## 几何与切片方向

geometry必须显式声明：axis_codes（例如RAS）；slice_axis=2，且第三轴为S/I；spacing_mm_min/max（三轴毫米上下界）；max_obliquity_deg（当前切片法向相对RAS superior的急角上限，0到90且不含90）。可选spatial_units_if_unknown仅用于header空间单位unknown的明确假设，示例未默认假设mm。

示例中的间距、15度和参考容差仅展示语法，未由本项目真实数据验证。还检查有限3D数据、非奇异正交affine、shear、qform/sform冲突、header voxel size与affine长度、各配对图/mask的物理网格和空间单位。发现问题需上游处理，不偷偷翻转或缩放影像。

报告角度是当前网格角度，不是原始头位/B₀角。模板配准后的affine不能证明原采集方向，相同shape/affine不能证明组织配准成功。

## 配准审核及辅助证据

每行registration_review为accepted、rejected或pending。accepted须有非空registration_reviewer及合法registration_review_date（YYYY-MM-DD）；registration_notes可记录脑室、基底节、病灶等核对内容。自动几何通过不代替人工审核，人工accepted不能覆盖几何或参考硬失败。

| 可选列 | 内容与限制 |
|---|---|
| registration_image_path | 已配准到SWI网格的GRE magnitude等辅助解剖图，供预览；同样检查几何和hash |
| qsm_brain_mask_path / qsm_brain_mask_source | 参考采集的独立脑mask及来源；不能用SWI mask或可靠性mask冒充独立边界，同文件/硬链接会拒绝 |
| registration_landmarks_path | 配准后独立确认的对应解剖标志，JSON明确使用RAS世界坐标mm |
| registration_transform_path | 实际变换文件，只记录hash不执行；兼容列名transform_path |

提供独立脑mask时报告Dice、对称平均/最大表面距离mm；这些不能证明内部核团和小病灶对齐，也不能靠复制mask取得成功证据。现有mask_path是可靠性/覆盖域，不用于推造独立配准准确度。

标志点格式为 `{"coordinate_system":"ras_mm","fixed":[[x,y,z],...],"moving":[[x,y,z],...]}`。两组为同名、配准后的对应点，数量相同且非空；报告n、RMS及最大距离。voxel索引或LPS不能直接填入，应先按实际affine和单位转换；标志对应与独立性仍需审核。

如registration契约增加min_mask_dice或max_landmark_error_mm，阈值需预先确定，缺证据或超阈值即失败。未设阈值时只记录辅助指标，不自动等同成功。SWI与QSM对比度和符号不同，直接强度相关不作为自动通过标准。

preview.html展示有界正交数组切面，优先用辅助解剖图，否则用QSM，各图独立窗宽仅用于显示，轮廓仅来自独立QSM脑mask。它不是放射学自动重定向，也不代替完整三维审核。

## QSM参考零点

reference契约有六字段：id、method（mean/median）、region、target_ppb、tolerance_ppb、require_mask。每行需要：

- qsm_reference、qsm_reference_method、qsm_reference_region，与契约一致。
- qsm_reference_source，实际重建pipeline及参考步骤/可追溯来源，不能用“ppb”替代参考定义。
- qsm_reference_mask_path，真实参考ROI；require_mask=true时必须提供。
- 可选qsm_reconstruction_mask_path，更广的重建可靠域；否则使用现有mask_path。

参考mask须同网格、非负非空，并包含于输入脑mask和重建可靠域，防止脑外padding零值假通过。CSF可能不在分析脑实质mask内，此时需实际重建可靠mask，不能删参考ROI或假造可靠性。

检查只将既有QSM换为ppb，计算ROI均值/中位数与目标的差，再按容差判断；不自动减均值、截断或调offset。接近0不能证明mask真的是CSF/白质，也不能证明来源正确，仍需来源记录与审核。

没有参考mask且require_mask=false时，只记录declared_only，明确未数值核验。完整参考身份为 `{id,method,region,target_ppb}`，随checkpoint、baseline artifact及预测/评价JSON传播，独立于Normalization的scale/offset。同名id但方法/区域/目标不同仍拒绝；容差和require_mask属于检查政策。

## 报告绑定及入口

report.json保存每例几何、审核、辅助指标、参考统计与来源。规范化契约、四主输入及所有提供的QC证据文件SHA256、case身份/元数据均被绑定。更换影像、mask、参考声明或审核记录后须重生成报告。

prediction_*列不绑定到输入QC，因此同一输入配对可评价不同模型；预测单位、网格、参考身份另行验证。subject、split、case_id等应稳定，不能审核后随意改名。

- train只核验train/val，checkpoint/config保存data_qc/data_reference。
- fit_baselines只核验train，artifact保存相同来源和参考定义。
- evaluate只核验所选split；baseline须有已核实来源，旧artifact需重拟合或走显式legacy路径。
- 新checkpoint推理自动检查SWI/脑mask与训练几何契约，并继承参考定义。SWI-only没有reference QSM，只称geometry_verified，不称验证了跨采集配准或输出ROI恰好为0。

没有qsm_units列时，QC必须给--qsm-units ppm/ppb。消费端继承已核实解释，不能因输出单位为ppb而把原ppm标签误当ppb；显式fallback冲突会拒绝。

```bash
python -m swixqsm_minimal.train \
  --manifest /prepared/qc_manifest.csv --output /experiments/verified_l1 \
  --qc-report /qc/final/report.json \
  --independence-report /audits/random_init/report.json \
  --units ppm --qsm-scale 0.1 --size 416 \
  --epochs 20 --batch-size 4 --adv-weight 0 --device cuda

python -m swixqsm_minimal.fit_baselines \
  --manifest /prepared/qc_manifest.csv --output /experiments/baselines/verified.npz \
  --qc-report /qc/final/report.json \
  --units ppb --reference-id actual_reference_id --space-id prepared_common_space

python -m swixqsm_minimal.evaluate \
  --manifest /prepared/qc_manifest.csv --output /experiments/evaluation/verified \
  --qc-report /qc/final/report.json --baselines /experiments/baselines/verified.npz \
  --units ppb --reference-id actual_reference_id --space-id prepared_common_space
```

未给data-range时PSNR/SSIM未定义，其余误差照常。加入prediction_path即可评价模型，新推理JSON带DataReference。旧预测缺完整定义时，应真实补prediction_reference、prediction_reference_method、prediction_reference_region、prediction_reference_target_ppb，不能仅凭同名id宣称等价。

新入口默认需报告。旧数据/旧checkpoint临时排查可显式用 `--allow-unverified-data "实际原因"`，状态与原因写为unverified，不是QC通过。strict评价拒绝明确标成unverified、failed、pending、rejected或未知QC状态的预测；legacy评价保留原预测的PredictionDataQC、PredictionDataReference及来源，另行记录EvaluationDataQC，不把原推理例外覆盖为verified。未记录的旧来源也明确保留not_recorded。

底层数组/拟合API不代替CLI验证，直接调用者负责相同检查。旧checkpoint不能从state_dict推断参考零点或训练几何。

## 验收范围

合成测试覆盖方向/spacing/shear/qform、毫米表面/标志距离、参考方法/单位/支持域、人工pending/rejected拒绝、过期hash、选定split隔离、单位fallback传播，以及已核实训练→推理→基线→评价全链。本次不执行真实配准或审核，不填造实际病例accepted记录，不替队列选择参考组织或容差。
