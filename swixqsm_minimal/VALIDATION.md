# 最小包验收记录

## 最新：2026-10-17 目录清单与服务器训练入口

完整 **289 项** unittest 通过，耗时74.853秒；76份Python源码通过AST检查。仍为Python 3.10.19、PyTorch 2.9.1、NumPy 1.26.4、nibabel 5.3.3、SciPy 1.15.3，本机CPU。原项目六份来源源码哈希未变，未训练真实模型或验证CUDA性能。

新增目录生成、SWI支持域准备及服务器编排测试，覆盖平铺/明确嵌套文件、缺漏与全部1773测试目录保留、冻结原始成员核对、metadata消歧及同人访视分组、原train内部划分val、单模型/4项/8项派发、CPU仅准备模式、argv子进程与来源记录、生成mask和原始来源变化拒绝复用。真实3例合成NIfTI的独立CLI检查通过，未建神经实验；删除生成的test mask或修改原test有效mask均拒绝复用。

额外只读核对现有数据：5083/1773原始ID名单无交集，本地22/16例均匹配所属原组。7例QSM数值检查支持作者暂定ppb量级；6例有原有效mask者完成SWI-only mask准备探测，逐例排除1–114个不在输入支持内的有效mask体素，均有来源与计数，未改变源文件。这不是全队列、配准QC接受、单位来源证明或性能评价。完整本地预检保留全部1773原test ID，因数据仅为样例及缺有效mask，在读取影像前停止，未启动训练。

运行和真实数据缺项见 [SERVER_TRAINING.md](SERVER_TRAINING.md)，机器记录见 [TRAINING_VALIDATION.json](TRAINING_VALIDATION.json)。下方保留各历史阶段验收记录。

## 最新：2026-10-17 评审四模型重训入口

完整 **242 项** unittest 通过，耗时 44.622 秒。环境为 Python 3.10.19、PyTorch 2.9.1、NumPy 1.26.4、nibabel 5.3.3、SciPy 1.15.3，macOS arm64 / CPU；没有进行真实受试者训练或 CUDA 性能验证。以下较早各轮的版本和测试数量保留为历史记录，不代表最新运行环境。

新增 36 项测试覆盖四种真实宽度模型、无跳连 FCN、QSM 线性输出、pix2pix 名义 70×70 感受野及 BN 限制、显式损失与有限梯度、同 seed 初始化与采样、四模型合成训练/保存/推理、严格对抗消融、warmup 调度及阶段早停、完整轮恢复与冲突拒绝、只从原训练池按全局参与者分割。训练测试故意损坏 held-out test 的 NIfTI，确认该入口不打开其影像；验证合并同一参与者的多个访视。独立探测另验证 416×416 四模型前向/反向，以及 warmup 后连续/分段训练的一致性。

新架构和目标仅用于作者另行授权的 prospective retraining，不宣称旧判别器或历史物理预处理的逐项复现。正式数据、单位、参考、配准和外部有效性须由真实实验核实。入口见 [TRAINING_MODELS.md](TRAINING_MODELS.md)，机器可读记录见 [TRAINING_VALIDATION.json](TRAINING_VALIDATION.json)。

日期：2026-09-06。测试环境：本机 CVMed Python 环境，PyTorch 2.9.1，CPU；未进行真实病例训练，未验证 CUDA/MPS 性能。

- 8 项 unittest 全部通过：可逆目标尺度/偏移/单位、元数据拒绝与旧最小包兼容、SWI/QSM 尺度分离、NIfTI 非默认 slope/intercept、混合单位与物理 MAE，以及原来的数据/梯度/训练推理检查。
- 已知值贯穿新训练快照、独立推理 CLI 与文件保存：网络常数输出 -2，在 scale=0.1、offset=0.02 ppm 时，磁盘 QSM 为 -0.18 ppm（绝对误差容限 1e-7）。这验证还原链路，不代表训练所得预测误差为零。
- 非默认来源 NIfTI scaling 正确加载一次；输出 float32、slope=1/intercept=0，重置来源显示范围并保存 Units/Normalization JSON。
- 合成数据尺寸 32×40×2，包含负 QSM 值与非单位 affine；推理保存后 shape/affine 保持一致且数值 finite。
- 使用本地 `save/20260119_1847/gen_19.pt` 验证旧生成器权重严格加载，71 个 state_dict 键匹配。
- 在原生成器与新副本均使用 `normalize=False`、eval 模式及相同输入时，CPU 前向输出逐值相等。这只验证迁移兼容性，不代表选定该 checkpoint 为最佳临床/科研基线。
- 原训练、推理、生成器、判别器、训练循环与数据加载共六个文件的 SHA256 在操作前后相同；校验值记录在 `legacy_provenance.json`。

代码按 train.py、model/、util/、test/ 组织；顶层旧入口仅做兼容转发。全部 Python 源码通过编译检查，原六个源码文件哈希仍未改变。

正式训练前仍需准备配准合格的 manifest、核实单位/参考零点、选择预训练权重并确认患者划分。此验收不替代真实数据 QC 或模型性能验证。历史 uint8 或裁剪丢失的信息不可通过本修复自动恢复。

## 2026-10-17 统一物理评价与训练基线

在同一CVMed环境、CPU执行完整测试37项，全部通过，耗时9.279秒。环境为Python 3.10、PyTorch 2.9.1、NumPy 2.2.6、nibabel 5.3.3、SciPy 1.15.3。使用临时合成数据，未运行真实病例训练/评价。

- 新增12项指标测试，涵盖物理误差/单位、零图NMSE=1、脑外扰动、空间SSIM、HFEN毫米尺度及厚层常数偏置、空分母和多访视充分统计量。
- 新增10项基线测试，涵盖train-only均值/已知逐体素线性关系、显式单位、重复访视权重、网格/空间/参考/支持域/身份冲突、安全存取和训练数据来源记录。
- 新增7项集成测试，涵盖物理零图、患者等权汇总、预测JSON/CSV/CLI单位与参考冲突、空间单位、四种方法在相同mask上的完整CLI拟合→评价→保存，以及失败不交付半套结果。
- 原8项归一化和训练/推理回归测试仍通过。生成器、判别器、训练入口、训练engine、数据加载、归一化和推理源码的操作前后SHA256一致。
- 新入口的评价范围为物理图像/体素级误差与简单基线。结果仍需真实配准、共同参考与独立受试者资料支持，不追认历史论文Table2数值。

## 2026-10-17 配准、方向/间距与参考零点QC

同一CVMed环境、CPU执行完整测试72项，全部通过，耗时25.712秒；收紧失败/待审核/未知预测来源状态后，另通过1项针对来源拒绝与保留的回归测试。全部使用临时合成数据，未读取真实受试者影像、进行真实配准或启动正式训练。

- 新增18项几何/报告测试、9项参考检查测试及8项入口集成测试。覆盖方向、毫米spacing、倾斜角、shear、qform/sform冲突、非法单位码、独立脑mask表面/标志点距离，以及参考方法/单位/ROI可靠支持域。原37项评价、基线和训练/推理测试仍通过。
- 人工pending/rejected、缺审核人/日期、参考身份冲突、过期文件/元数据hash均被拒绝；没有参考ROI时只记录declared_only。验证不执行重新参考或影像重排。
- 完整已核实训练→推理→基线拟合→评价链通过；推理拒绝不符训练spacing契约的输入。报告只重验消费split，新增预测列不使输入QC失效。输入单位fallback按QC记录传播，不被输出单位重解释。
- 缺失图像、截断QSM payload、不可识别影像、非法单位和预览读取失败能保留逐例failed报告；无效manifest结构/身份划分或契约仍直接拒绝。
- 未核实及失败/待审核/未知预测来源不能在严格评价中当作已核实；legacy路径保留PredictionDataQC、PredictionDataReference、原始文件/sidecar/checkpoint来源，评价来源单列EvaluationDataQC。
- 操作前后SHA256确认本包生成器、判别器、训练engine、归一化和指标实现五个文件均未改变。训练/推理入口仅增加QC核验与来源记录，网络结构、损失和选模规则未改动。全部Python源码通过语法检查，示例契约及CSV字段一致。

真实实验仍须按研究方案填写方向/spacing范围、参考区域/方法/目标及容差，提供实际参考来源/ROI并逐例审核配准。合成验收不证明任何真实病例已经通过QC；使用步骤见[DATA_QC.md](DATA_QC.md)。

## 2026-10-17 第三项与无需重训的固定模型评价

作者补充实际流程：held-out test participants在训练初期分为测试集，训练没有验证集，测试者从未参与权重选择或调参，仅在权重固定后评价。本轮按该事实保留留出测试，不要求重新划分或真实模型重训；原稿§2.4的validation叙述须另按实际日志纠正，尚未修改原Word。

完整113项unittest在同一CVMed环境、CPU全部通过，耗时28.507秒。本轮新增41项，包含21项身份/初始化历史、8项阶段状态、4项真实保存路径集成、7项固定模型选择记录，以及1项无validation的CLI固定模型评价；另扩展既有QC全链测试，覆盖完整来源传播。

- 固定模型评价仅读取既有物理预测和权重字节/hash。无val的train/test manifest正常通过；选模声明、实际checkpoint、预测来源及证据文件绑定，未知/开发期评价不改称heldout，缺日志只记录declaration_only。训练集行不可标heldout，已有validation/test-based历史不可被相反新声明覆盖。
- 全局subject一对一、跨split身份、解码SWI副本、完整/未知祖先暴露记录及checkpoint/history变更均有拒绝或未核实状态；审计允许train/test无val，消费端只重验指定split内容。
- 在临时合成训练中控制两轮val分数为1、2，验证全局best仍归warmup，同时保留GAN阶段best且确有D和对抗G更新。各文件自身分数与全局兼容字段明确区分。未进入GAN的请求提前拒绝；这些机制只预防后续新run，不追认历史权重阶段。
- 原预测的QC、独立性、训练阶段、选模记录和checkpoint来源在另存结果时保留；新评价报告不会自动补齐旧来源。
- 操作前后hash确认生成器、判别器、训练engine、归一化、数据加载、物理指标及底层评价七个实现文件未改。新增内容为审核、记录和阶段保存控制，没有改网络、损失或原有物理公式。

测试仅使用临时合成影像/权重，未启动真实病例训练、重训或真实队列评价，未修改原Word/真实权重。未声称已补齐L1-only/pix2pix真实消融或GAN性能增益。步骤和待作者填入的实际epoch/日志见[TRAINING_AUDIT.md](TRAINING_AUDIT.md)。

## 2026-10-17 第四项有效patch损失与诊断

同一CVMed环境、CPU完整141项unittest全部通过，耗时30.167秒。本轮新增12项感受野/有效BCE、10项分布/聚合诊断及6项训练整合测试；扩展既有阶段保存和无val冻结预测评价测试。仅使用临时合成数据/权重，未启动既有真实模型训练或重训。

- 当前判别器真实Conv链的理论RF为38×38、stride8；逐层正支持传播与独立输入索引追踪一致，涵盖单体素、角落、边界、dilation/stride变化。全背景感受野不计BCE，partial边界保留，D真/假/G共享支持和整batch选中patch分母；全空支持拒绝。
- 被排除logit的扰动不影响有效loss或gradient；增加远处且与stride对齐的padding不稀释有效损失/梯度；全ones支持与旧BCE mean一致。未来默认valid_patch，显式all_patches另留原全patch范围，其有效比例与实际loss分母分别记录。
- 整合检查确认G仅forward一次、BN统计仅更新一次、D梯度隔离及mask外G输出梯度为0。开启/关闭诊断的实际G/D权重更新、BN状态和RNG逐位相同；诊断复用已有更新前预测。非有限输入保护独立于诊断开关，坏输入在模型/优化器更新前拒绝。
- raw L1/adv/D按明确sum/count聚合，实际batch总目标和加权adv/L1比例另记；warmup/L1未执行的adv/D为null/not_observed。比例不当作梯度贡献，不能自动推断模式塌缩或调整lambda。
- 有界训练batch分布先按固定scale/offset还原物理值，只统计有效mask，保留逐batch精确分位数，不伪造全epoch分位数。空分母、非有限有效数据及数值溢出保留null/status；不写JSON NaN/Infinity。
- 冻结评价的distribution_diagnostics.json在已有单位/参考/网格检查后生成逐例/方法物理分布；仅train/test且无val的流程通过，固定权重字节不变。旧来源缺少TrainingObjective/TrainingDiagnostics时保留未知，不通过当前诊断倒填历史记录。
- 生成器、判别器、固定归一化、数据加载、物理指标、阶段选模和既有选模记录七个实现文件的操作前后hash一致。engine仅增加有效patch目标与日志/非有限保护；masked L1及物理验证公式保留。全部Python源码通过语法检查，两份原Word未修改。

本次不改变既有模型输出、不补造历史D损失，不证明有效patch改动已提高真实GAN或小病灶表现；没有完成新的L1/pix2pix训练消融或ROI/病灶级研究。使用步骤见[PATCH_LOSS.md](PATCH_LOSS.md)。

## 2026-10-17 第五项切片采样与整卷缓存

同一CVMed环境、CPU完整173项unittest全部通过，耗时28.156秒；56份Python源文件通过AST语法检查。本轮新增32项（采样8、缓存9、数据I/O与原数值兼容9、benchmark入口5、训练缓存参数提前拒绝1），并扩展原训练保存路径测试。没有真实病例训练/重训、真实模型选模或模型精度比较；训练入口回归仅使用临时合成影像/权重。

- 修复全局随机slice与2卷LRU的反复解码：未来默认按随机subject/访视的小组卷内打散，每有效slice每epoch恰好一次，不跨组拼接尾批。可显式选择slice_random；采样与DataLoader均使用独立随机状态。batch组成/步数可能变化，不声称历史优化轨迹等价，也未新增患者等权重采样。
- 每dataset实例有独立卷数/字节双上限缓存，默认2卷、256MiB准备后数组，不是总RSS限制。train/val不再争用方法装饰器的全局2项缓存；超预算单卷绕过、失败不留半项、序列化冷启动、返回tensor不污染只读缓存均通过。极小预算下仍保持数值，但会反复解码，计数如实记录。
- 建索引仅检查四header和解码有效mask；原SWI/QSM/输入脑mask数值、subset及归一化检查延至首访，前置QC门保留。Io计数区分index、四文件准备、成功预处理与命中/逐出，并在config、history、checkpoint和selection记录；不计上游QC/身份审计或操作系统物理磁盘流量。
- 10卷×8有效slice的合成四角色探测：遍历解码276→40，减少85.5%；新index另解码10个mask及检查40个header，旧构造解码40文件。与保存的修改前minimal loader比较，全部80个有效slice索引及SWI/QSM/mask tensor字节完全相同；另有混合ppm/ppb、非零offset、负QSM、非默认NIfTI slope/intercept、稀疏mask、transpose/padding逐bit回归。
- 只读检查用户提供样例：train22可见目录中10例有SWI/QSM/QSM_mask，但无四角色齐全交集；validate16可见目录的直接角色未见QSM/QSM_mask。未以目录名推断选模val，沿用1773例预先独立test且固定权重后评价的事实。没有拿QSM_mask代替输入脑mask、猜QSM单位或创建正式训练manifest。
- 真实3卷×8slice仅作raw三角色探测：遍历解码45→9，减少80%；wall time约0.9913→0.2106秒，逐slice值hash全部相同，9个压缩源文件操作前后SHA256相同。startup每策略另解码9文件。未测真实四角色预处理/物理误差；OS page cache未受控，不将该单次时间外推成全量训练提速。
- 本轮只修改数据读取与训练采样/日志入口；生成器、判别器、engine、固定归一化、有效patch损失、物理指标、阶段选模、冻结评价及推理实现的操作前后hash均一致。原Word、真实权重和外部影像没有修改。

使用步骤与边界见[SAMPLING_IO.md](SAMPLING_IO.md)，完整计数与哈希见[合成报告](../outputs/SWIXQSM-GAN_priority5_synthetic_io_20261017.json)、[真实raw报告](../outputs/SWIXQSM-GAN_priority5_real_raw_io_20261017.json)及[本轮验收记录](../outputs/SWIXQSM-GAN_priority5_validation_20261017.json)。

## 2026-10-17 第六项固定预测的ROI区域一致性

同一CVMed环境、CPU完整206项unittest全部通过，耗时29.288秒；61份Python源码通过AST语法检查。新增33项（统计13、区域提取/缺失/完整访视/数值保护14、CLI整合6）。全部使用临时合成数据，没有真实ROI队列分析或真实模型训练/重训。新增ROI入口不运行模型forward；原有回归测试仍包含临时合成模型的训练保存与推理。

- 只读回核原Word，下一项对应R1.6的bias/LoA/区域ppb误差、R3.3的ICC CI与R2.6/R3.9的扩展ROI/队列。原文48人ROI与全队列范围区分，未把整体分布或前五项代码完成当作区域/病灶验证。
- evaluate的可选--roi-spec复用同一已核验物理target/prediction和共同mask，接受真实同网格整数label图；不自动分割/配准/重采样/校准。区域review与图像QC分开，label/spec hash绑定来源但不提升审核或身份状态。模板中的label编号仅为格式例子。
- 缺标签、缺来源、未审核、空ROI、双侧缺component及覆盖/体素数不足保留null与原因；记录n_label/n_valid/coverage。坏文件、浮点概率label、复数/RGB、网格/单位不匹配及非法accepted审核配置拒绝。所有方法必须覆盖所有所选case，并使用同一mask。
- 所有预期访视合格才汇总participant ROI，否则incomplete_visits；合格者按体素充分统计量池化，每人仍计一次。globalID存在时完整且与subjectID一对一，同人跨cohort拒绝；overall/cohort均保留有效/排除人数。不能由图内coverage推断FOV外解剖覆盖或实际标签质量。
- 参与者ROI均值输出bias、sample SD、固定95%normal-based LoA、Pearson与两个固定方法single-measurement绝对一致性ICC(A,1)。独立手算例x=[1,2,3,4], y=x+2的ICC=5/11，y=2x的ICC=8/17；反序n3保留ICC=-3。这些防止误用consistency/average ICC或截去负值。
- paired participant percentile bootstrap使用局部RNG；同人配对共同重抽，公开每指标有效/退化rep数与原因，partial_defined为条件摘要；n<3不报告CI，点估计未定义也不补CI。点区间不是LoA、BCa或多ROI同时区间，不报告临床等价或病灶效能。
- 开启ROI前后per_case.csv、per_subject.csv及原summary指标逐字节/逐值一致；ppm/ppb改变有单位量而不改变r/ICC；同输入seed的区域报告复现。无val的冻结选模记录仍通过，合成checkpoint与所有输入字节不变。严格图像QC不会自动升级pending ROI审核；新增ROI列后旧QC元数据hash仍拒绝，重新生成真实更新后报告即可。
- 数值溢出、常量/少人/零能量、缺失和空分母严格JSON安全。ROI错误发生在staging期间，不交付半套目录；已存在的输出拒绝覆盖。生成器、判别器、训练入口/engine、归一化、loss、指标、QC/身份门、冻结选模记录、采样/缓存及推理实现的本轮前后hash均一致。

软件示例见[合成ROI演示](../outputs/SWIXQSM-GAN_priority6_synthetic_roi_demo_20261017/DEMO_ONLY.json)：7次合成采集、6个参与者、两个合成cohort，已知偏差2ppb被还原为约2ppb；r=1但固定偏差降低ICC的独立例子也保留。全部人物、标签、图像和checkpoint均为虚构软件fixture，不能写入真实论文结果。临时输入在演示后清理，可由测试fixture重建。

真实标签/物理预测未齐，不声称已补UKB/Shunyi/CSVD结果、48/87筛选分析、RN/SN/DN、微出血/钙化、临床关联或读片研究。使用见[ROI_EVALUATION.md](ROI_EVALUATION.md)，剩余排序见[优先级更新](../outputs/SWIXQSM-GAN_剩余算法优先级与第六项实现_20261017.md)。
