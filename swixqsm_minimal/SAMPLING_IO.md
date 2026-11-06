# 切片采样、内存缓存与有界I/O探测

2026-10-17。第五项解决小容量整卷缓存与全局随机切片读取之间的配合问题，属于训练可行性与可追溯性改进，不是精度错误或性能提升的证据。不改变网络结构、现有权重、QSM数值变换或损失定义，不执行真实病例重训，也不创建持久化影像缓存。

作者已明确1773人预先独立划为test，训练没有val，且仅在权重固定后评价。样例路径中的 `validate/`只是目录名，不能据此把这些人重新解释为模型选择val。本次探测不选模型、不使用测试结果调参，也不要求重划分或重训；固定模型记录见 [TRAINING_AUDIT.md](TRAINING_AUDIT.md)。

## 原问题与评审关系

原minimal对全部有效slice随机打散；每次cache miss都重新解码SWI、QSM、输入脑mask、参考有效mask，并重新计算仅依赖输入的SWI百分位归一化。旧 `@lru_cache(maxsize=2)`只保留两个准备后的整卷tuple；该装饰器的wrapper跨dataset实例共享容量，key含self，并非每个train/val实例各有两项。

跨很多参与者随机取slice时，同一卷下一次被访问前可能已被逐出缓存，从而重复读取gzip NIfTI与预处理。这个机制可以用调用计数验证；它不等于已经测得全量5083人训练的真实速度，也不能由此认定模型精度下降。

审稿人没有直接指出缓存/I/O bug。它间接支持R3 major 8要求的可完成、公平且预算明确的基线与消融，以及epochs/训练流程报告；不是对R1小结构验证或R3物理指标问题的替代。既有优先级报告的10人×8slice调用计数是早期小型合成观察，不应当作本次新benchmark结果或全队列提速承诺。

## 新的数据加载与实例缓存

`PairedSlices`保留既定float32、SWI归一化、固定可逆QSM变换、有效slice及padding规则；缓存由 `util/volume_cache.py`的 `InstanceVolumeCache`管理。

1. dataset启动时读取所选split每卷四个NIfTI header检查基础形状/网格，只解码有效性mask建立有效slice索引；不为索引预先解码或归一化SWI/QSM。
2. 请求某slice时，准备后的volume命中cache则复用；miss才解码四文件、检查有限值/支持域、按既定方案准备SWI、QSM target和有效mask。输入脑mask的完整数值及有效mask包含关系检查延迟到首次访问，不代表启动时已验证全部输入；独立QC门仍在训练前运行。
3. resident内容是准备后的三个ndarray组成的tuple。cache管理器不写文件、不重采样、不更改dtype、数组flags或调用者数据；dataset owner将自己新准备的三个数组设为只读，`pad_slice`复制后再返回tensor，防止一个slice意外修改缓存影响后续slice。

默认每个dataset实例最多2卷、256MiB，须同时满足两项限制。超过卷数或字节预算时按LRU逐出较早项；单卷自身超过预算时，正常返回loader结果但不驻留（oversized/bypass），不为它清空已有cache。loader失败不插入半成品，也不逐出原有项。

`dataset.io_stats`提供统计快照；`dataset.clear_cache(reset_stats=False)`释放cache引用但保留累计统计，显式true才重置。cache的主要字段包括：

| 字段 | 含义 |
|---|---|
| hits / misses | 准备后volume请求命中/未命中次数。 |
| load_attempts / loads / load_failures | loader尝试/成功/失败；成功bypass也计loads。 |
| evictions | 为满足容量约束逐出的resident项数。 |
| oversized / bypasses | 超预算而不驻留的结果。 |
| current_items / peak_items | 当前及观测峰值resident卷数。 |
| current_bytes / peak_bytes | 当前及观测峰值resident准备数组逻辑字节数。 |
| max_items / max_bytes | 此实例的两项配置上限。 |
| clears / cleared_items / cleared_bytes | 显式清理及释放的cache引用统计；不是向操作系统保证已归还同量RSS。 |

`dataset.io_stats`包含上述嵌套cache快照，以及 `index_header_reads`（索引时四角色header检查）、`index_volume_decodes`（有效mask解码）、`prepared_volume_decodes`（正式准备的四角色解码）、`volume_decodes`（两者合计）、`volume_preparations`（成功完成准备）。decode计尝试，包括失败；prepared decode也会读取header。一次prepared-volume miss不能当成只读一个NIfTI，启动索引开销与正式slice遍历开销须分别看待。这些计数不包含上游QC/独立性扫描，更不是OS实际磁盘读次数。

### 256MiB不是进程内存上限

预算仅累计resident tuple中的 `sum(array.nbytes)`，不包括解码/归一化/百分位临时数组、调用者仍持有的引用、padding与tensor复制、模型/优化器、Python开销、操作系统文件cache或额外的数组底层storage。view按逻辑nbytes计，不保证等于独占物理分配。

train和val等不同实例各有自己的预算；worker进程也不会共享一份全局resident缓存。pickle/deepcopy后的接收实例从cold cache和新统计开始。缓存返回共享数组引用，调用方不得原位修改resident数据；需要改动时应复制。实例内配置、输入或归一化发生变化，应清cache并重新建立对应dataset，不能复用旧准备结果。

本机探测到物理内存总量64GiB，但可用内存未确认；本地剩余磁盘约5.7GiB。这里保留小型内存cache、不创建persistent cache，避免把大批准备数据写入有限磁盘。两项资源观察不构成全量训练可运行或RSS安全的保证。

## volume_grouped采样的确切行为

`util/sampling.py`的 `CacheAwareBatchSampler`只读取当前dataset的 `index`与 `rows`元数据，不触发图像加载，也不读取其他split。每epoch：

1. 使用独立CPU `torch.Generator`，以seed+epoch（mod 2^64）建立可重现顺序，不改变全局torch/Python/NumPy RNG。
2. 随机排列subject，再随机排列该subject的访视；同人访视在volume时间表中连续。
3. 每组最多volume_group_size卷，只在这一组内部打乱有效slice，再按batch_size生成batch。
4. 每组结束时保留自己的partial batch，尾部不跨组拼接。每个dataset有效slice索引在epoch中恰访问一次，不重复、不丢弃。

组边界可能混合不同subject；该方法不是患者等权采样，也不对多访视者降权或过采样。顺序、batch组成与partial batch数量可能改变，因此优化步数、BatchNorm/Dropout经历和梯度轨迹可能不同。**数值预处理相同、每slice一次，不等于历史训练轨迹等效**。

### 后续训练参数

| 参数 | 默认 | 用途 |
|---|---|---|
| `--sampling-mode` | `volume_grouped` | 分组工作集采样；`slice_random`保留全局随机slice模式供对照。 |
| `--cache-volumes` | 2 | 每dataset resident卷数上限。 |
| `--cache-mib` | 256 | 每dataset准备数组逻辑字节上限；1MiB=1024² bytes。 |
| `--volume-group-size` | 2 | 每采样组最多卷数。 |

训练入口使用num_workers=0，使当前进程的访问与cache统计可直接核对；没有新增多worker调优。val沿用既有顺序读取。分组模式要求cache卷数至少覆盖group size；**字节预算也必须能容纳真实工作组**。否则即便卷数限制满足，仍会频繁evict或bypass，需从miss/eviction/oversized统计识别，不能只因用了grouped就宣称每卷仅加载一次。

新run的 `config.json`记录 `sampling_policy`及启动后的 `indexing_io`；每epoch的 `history.jsonl`、checkpoint和 `selection.json`记录 `training_io`。其中train/val分别有 `read_delta`与 `cache_counter_delta`，表示本epoch区间计数；`after`保留累计计数、当前resident及累计峰值。current/peak bytes不是相减后可解释为“本epoch分配量”的计数，不能把累计peak误叫单epoch峰值。

所见样例shape为256×288×48或288×256×48，单个float32整卷为14,155,776 bytes（13.5MiB）。准备后三数组约40.5MiB/卷，两卷约81MiB，仅是resident估算；完整处理峰值明显还包含临时数组。不同队列、更大矩阵或更多层必须按实际nbytes重新判断。

本次不进行真实病例训练或重训；回归检查可以用临时合成微型数据调用训练入口核对保存路径，不产生真实临床实验结果。未来如主动开展新实验，可在已有合规训练命令加入：

```text
--sampling-mode volume_grouped --cache-volumes 2 --cache-mib 256 --volume-group-size 2
```

比较损失或架构时应保持采样/缓存配置及训练预算可追溯，不能让一个方法额外获得不同训练步数却只报告相同epoch。QC、独立性、物理单位与阶段归属规则仍见 [DATA_QC.md](DATA_QC.md)、[TRAINING_AUDIT.md](TRAINING_AUDIT.md)及 [PATCH_LOSS.md](PATCH_LOSS.md)。

## 本次外部样例的有限清单

只读目录为 `/path/to/swi_qsm_data/train/`及 `validate/`。以下是提供的**样例目录**，不是论文完整5083/held-out test participants，也未递归遍历所有子目录。

| 直接可见角色 | train样例 | validate样例 |
|---|---:|---:|
| 直接目录entries | 22 | 21 |
| 非隐藏subject目录 | 22 | 16 |
| 隐藏entries | 0 | 5 |
| SWI.nii.gz | 22 | 16 |
| QSM.nii.gz | 20 | 0 |
| QSM_mask.nii.gz | 10 | 0 |
| brain_mask.nii.gz候选 | 2 | 16 |
| 同subject同时有SWI/QSM/QSM_mask | 10 | 0 |
| 上述三者再加brain_mask候选 | 0 | 0 |

还有部分GRE magnitude、phase及变换文件；表中0只指该直接角色未发现，不证明其他未遍历位置不存在重建结果。brain_mask文件名不证明生成来源或部署时能获得，也不等于已通过QC。

清单阶段仅对2个train与2个validate非隐藏subject检查NIfTI header，未在该阶段解码整卷数组；后续另做下述有限raw探测。所看SWI为float32、RAS、约0.798611×0.798611×3mm、48层；train的三主图shape/affine一致，validate的brain_mask候选也与各自SWI同网格。header一致不证明真实组织配准，RAS也不代表没有obliquity或已知原始B0方向。

所看train的SWI压缩文件约2.67–2.82MB、QSM约2.43–2.53MB、QSM_mask约45.5KB；所看validate的SWI约2.60–2.86MB。压缩文件很小不意味着解码后的驻留内存很小。

QSM的ppm/ppb不能从文件名、dtype、NIfTI的mm空间单位或强度范围猜测，需要真实重建来源。train中三角色齐全的病例缺独立输入脑mask，不能拿QSM_mask充当 `swi_mask_path`、假定全脑mask或复制一个文件凑四列，继而写成可训练manifest。validate样例不作为配对QSM targets，也不因目录名而成为选模val。

因此本次真实样例适合有限的**raw三文件读取/解码/访问顺序探测**；完整四角色预处理、tensor数值一致性先用已知mask的合成数据检查。raw路径列表是本地私有探测输入，不在本文公开参与者ID，也不作为正式训练manifest。

## 有界benchmark与结果记录

`benchmark_io`分别比较slice_random和volume_grouped，不运行模型forward/优化器，不据速度结果修改固定权重或test定义。`--max-volumes`默认3，manifest只访问所选split前若干卷的影像；raw始终另有最多3卷的硬上限。每卷 `--slices-per-volume`默认8，从有效第三轴slice均匀取至多这一数量，不代表随机选取全部解剖层。完整manifest仍做原有元数据/路径存在检查。报告记录cache、seed、请求/读取计数、时间及数值校验范围；输出JSON不得覆盖既有报告。

两种输入范围需清楚区分：

- `--manifest`：需要真实完整的四角色输入以及显式 `--qsm-scale`、`--qsm-offset`、`--units`，比较实际PairedSlices准备与取样；每行真实输入单位也须已知，不能为速度探测补造缺失mask或猜测ppm/ppb。
- `--raw-probe-json`：本地三文件tuple读取探测，不执行SWI脑mask归一化或QSM物理解释，不能宣称验证了完整训练输入数值协议。

缓存cold只指每次比较重新建立/清理应用内cache，不代表操作系统磁盘cache已清空。小样本、相邻运行、外部设备及gzip条件都会影响wall time；两种模式耗时比不能直接外推成完整训练提速，load调用数也不等于真实磁盘物理读次数。

### 本次已完成的有界结果

原始记录为 [真实raw报告](../outputs/SWIXQSM-GAN_priority5_real_raw_io_20261017.json)与 [合成四角色报告](../outputs/SWIXQSM-GAN_priority5_synthetic_io_20261017.json)。两项均采用seed=42、cache最多2卷/256MiB、group最多2卷、batch_size=4。报告中的policy名 `global_random_slices`对应slice_random，`cache_aware_volume_groups`对应volume_grouped。

| 观测项：正式遍历阶段 | 全局随机slice | 分组slice |
|---|---:|---:|
| 真实raw：3卷×8slice，三角色NIfTI解码调用 | 45 | 9 |
| 真实raw：volume loads | 15 | 3 |
| 真实raw：遍历wall time（含取样与值hash，秒） | 0.9913 | 0.2106 |
| 真实raw：dataset access time（秒） | 0.9778 | 0.1986 |
| 合成四角色：10卷×8有效slice，准备期NIfTI解码调用 | 276 | 40 |
| 合成四角色：volume preparations/loads | 69 | 10 |

真实raw正式遍历解码调用减少80%；两模式各自startup另解码9个三角色文件以建立探测slice索引，不包含在45/9内。这是raw工具的索引行为，不是新PairedSlices仅mask解码的startup行为。两个模式均覆盖同一24个slice，逐index返回值hash一致；9个输入压缩NIfTI在probe前后SHA256一致。resident tuple峰值约81MiB，不能解释为进程RSS峰值。真实缺失输入脑mask及QSM单位条件未被补造，未测真实四角色预处理或物理QSM精度。

合成四角色显式设置max_volumes=10，使用10个已知volume、shape=32×40×10、每卷8个有效slice，输入QSM单位交替ppm/ppb，固定变换scale=0.1ppm、offset=0.02ppm。新dataset每个policy启动另读40次header、解码10次有效mask；正式遍历解码调用276→40，减少约85.5%，不含startup。两模式80个slice索引与返回值逐项一致。

另外以保存的**修改前minimal loader**对同一合成数据复核：旧构造阶段解码40次四角色文件，新构造只解码10次有效mask；旧全局随机遍历同为276次解码。全部80个slice的index和返回tensor字节与新loader逐bit相同，覆盖既定SWI归一化、单位换算、非零offset、mask与padding路径。这不是原论文历史训练代码/权重的复现，也不证明不同batch顺序的优化等效。

这些时间是一次小型顺序探测，OS page cache未受控；不给出全量训练加速倍数或精度收益。合成结果以数值/调用计数为主要证据，不把其wall time作为真实数据速度承诺。

### 可复用命令

在CVMed根目录、使用CVMed Python环境运行。以下是完整四角色manifest的通用语法；先按真实来源填写输入单位、mask及固定变换，不使用本次缺mask的样例凑列。scale=0.1、offset=0、units=ppm只是语法示例，不是从外部样例估计出的参数。

```bash
python -m swixqsm_minimal.benchmark_io \
  --manifest /prepared/complete_four_role_manifest.csv --split train \
  --units ppm --qsm-scale 0.1 --qsm-offset 0 --size 416 \
  --max-volumes 3 --slices-per-volume 8 --batch-size 4 \
  --cache-volumes 2 --cache-mib 256 --group-volumes 2 --seed 42 \
  --output /experiments/io/four_role_probe_new.json
```

benchmark参数名为 `--group-volumes`，训练参数名为 `--volume-group-size`。一条benchmark命令自动执行两种policy，不需先后手动修改模式；输出含各自startup、traversal、逐index值hash及比较。manifest输入的status仍为data_qc not_assessed，benchmark通过不代替正式QC/独立性门。

本次raw probe使用本地临时候选JSON，不在文档公开参与者路径。raw模式拒绝QSM scale/offset/units声明，不做物理解释；正式报告见上面的真实raw链接。重跑必须使用新输出文件名，不能覆盖上述记录；临时合成fixture不作为真实队列数据发布。

本次不建立persistent cache，不改外部影像/头信息，不把有界probe当作训练或QC通过，不以合成数值相同声称模型精度、病灶表现或历史结果已经验证。
