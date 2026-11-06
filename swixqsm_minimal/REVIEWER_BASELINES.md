# 评审要求对应的基线与消融实验

本清单依据原评审的 Reviewer #3 Major 1、Major 8，以及 Reviewer #2 Major 4，说明新训练入口覆盖的实验及仍需数据条件的比较。**代码与实验方案已补充，不代表真实训练、测试性能或论文结论已经得到验证。**

## 1. 实验清单与当前实现

| 评审要求 | 实验及入口 | 当前状态与解释 |
|---|---|---|
| #3 Major 1：零图 | `evaluate.py` 的 `zero` 方法 | 已有实现；取共同物理单位和已声明 QSM 零参考下的数值 0，不是网络归一化空间的 0。应与其他方法使用同一评价支持域。 |
| #3 Major 1：平均 QSM 图 | `fit_baselines.py` → `train_mean` | 已有实现；仅由**新 train** 拟合逐体素平均图。不能使用 val、N_test 人 test 或外部队列的 QSM 计算“队列平均图”。 |
| #3 Major 1：逐体素 SWI→QSM 线性回归 | `fit_baselines.py` → `linear` | 已有实现；在共同解剖空间内，逐体素拟合带截距的加权 OLS，输入采用既定 SWI 归一化。每体素 SWI 无变化时退化为训练均值；不在 test 上重新拟合斜率、截距或校正系数。 |
| #3 Major 8：plain L1-only U-Net | `--model plain_unet` | 新训练配方；与 SWIX 相同的 UNetSmall 生成器、线性输出，`lambda_l1=1`、`lambda_adv=0`，不构造判别器。 |
| #3 Major 8：pix2pix 标准权重 | `--model pix2pix` | 新训练配方；八级 U-Net、条件 70×70 PatchGAN，`lambda_l1=100`、`lambda_adv=1`。采用适合 QSM 的明确修改，见下文，不称原论文逐项精确复现。 |
| #3 Major 8：移除对抗项 | `--model swixqsm_l1_ablation` | 新增严格配方；保留 SWIX 模型工厂及初始化流程，固定 `lambda_l1=0.2`、`lambda_adv=0`。判别器不前向、不反向、不更新，用于与默认 SWIX 的因果消融；不能用推理时关闭 D 代替重新训练。 |
| #3 Major 8：解释 L1 的 0.2 与 100 | 默认 SWIX + `swixqsm_lambda1`、`swixqsm_lambda10`、`swixqsm_lambda100` | 新增固定敏感性配方；相同 SWIX 架构，`lambda_adv=1`，`lambda_l1` 分别为 0.2、1、10、100。比较实际损失、输出分布、验证和测试误差；本清单未预判哪个系数更好。 |
| 用户确认的第三个学习基线 | `--model fcn_gan_l1` | 新训练配方；FCN 明确定义为移除 encoder→decoder 长跳连的同宽 32/64/128/256 encoder–decoder，保留相同卷积块、全局判别器和 `0.2 L1 + 1 GAN`。这是本项目操作定义，不冒称评审指定了某个唯一 FCN 文献模型。 |
| #2 Major 4：filtered-SWI phase 路线 | 需真实相位数据及可核查外部方法 | **条件未满足，未在本训练包中伪实现或宣称完成。**与 SWI 幅度输入的方法具有不同信息条件，详见第 5 节。 |

默认 SWIX 配方为 `--model swixqsm_gan`：UNetSmall 生成器、QSM-only 全局判别器，`lambda_l1=0.2`、`lambda_adv=1`。`plain_unet` 是常规 L1 基线；精确比较“仅移除对抗项”时，应使用单独的 `swixqsm_l1_ablation`，避免同时把 L1 系数从 0.2 改为 1。

## 2. 模型名称和损失应如何报告

新配方统一采用

```text
L_G = lambda_l1 × mean(|y_pred - y_true|, mask > 0)
    + lambda_adv × BCEWithLogits(D(fake), 1)
L_D = 0.5 × [BCEWithLogits(D(real), 1)
           + BCEWithLogits(D(detached_fake), 0)]
```

这里的 L1 在**固定归一化后的 QSM**上计算，正 mask 值表示体素成员资格，不是概率权重。若 `y=(q-offset)/scale`，则 `L1_normalized=L1_physical/scale`，因此物理单位误差的有效系数是 `lambda_l1/scale`。所有方法必须共用相同 units、scale、offset 及参考约定；只比较 0.2 和 100 的数字，不能证明两种目标在原始 ppb 数据中具有同样含义。旧 minimal 的 `L1 + adv_weight×BCE` 也不能直接等同于本入口的系数写法。

生成器均为线性输出，允许负 QSM 及超出训练范围的预测，不使用 Sigmoid、Tanh 或目标裁剪。新训练 SWIX 的全局判别器保留历史 3×3 卷积、InstanceNorm 及 Flatten 空间分类设计，将末端两输出/ReLU 改为单个无激活 logit，并按配置的图像尺寸建立分类层；它不加载历史判别器权重。InstanceNorm 使用单切片内全部空间位置的统计，不具有严格局部性，但不混用不同切片的批统计。**这是修订实现的新训练，不是加载旧权重或逐项重现旧训练轨迹。**

pix2pix 依据[原论文](https://arxiv.org/abs/1611.07004)及[官方网络](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py)、[官方目标](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py)采用八级 U-Net 和 `GAN + 100 L1`。本包将输出 Tanh 改为线性头，单通道输入输出；不足 256 倍数的图像只在右/下补零，输出裁回原网格，不插值。416×416 的输入在生成器内部补至 512×512。统一训练预算、归一化、mask 和初始化策略也须如实报告，因此名称应写为“pix2pix adapted for QSM”。

pix2pix 的 70×70 是**卷积拓扑的名义感受野**。训练态 BatchNorm 仍耦合空间位置及批内样本。`valid_patch` 仅排除直接卷积 mask 足迹完全为空的损失位置，不证明背景或其他样本没有梯度影响。SWIX/FCN 的全局判别器每个非空切片只有一个 logit，不能把它的全局损失称为局部 PatchGAN。两类判别器均对真图、假图施加同一有效 mask。

## 3. 共用数据划分和选模协议

1. 原始 N_test 人在初始训练前已分离，用户确认未用于旧模型权重或超参数选择。新训练继续保持这批人的 test membership，不把目录名 `validate` 自动解释为新 val。
2. 使用 `split_training.py` **仅从原 train** 生成新 train/val，以 `global_subject_id` 将同人全部访视和跨队列别名放入同组，固定 seed 和分割清单。该步骤只处理 CSV 元数据，不读取影像，也不能替代身份内容重复、配准、单位和参考审计。
3. 所有学习模型、消融和敏感性实验共用这份分割、有效切片、输入归一化及预先固定的训练设置。QSM 固定 scale/offset 若由数据估计，只能依据新 train；不能使用 val/test 的标签估计它们。单例 SWI 的既定脑内百分位归一化仅使用该例部署时可获取的输入。
4. 只在新 val 上依据物理单位的受试者等权 MAE 选 epoch 或预先声明的超参数。保持共同采样、批次预算、学习率策略和增强政策，并记录 G/D 的实际更新次数、初始 G 状态哈希、采样顺序、参数量及随机种子。FCN 去掉跳连会改变参数量，不能宣称完全容量匹配。
5. 默认不额外旋转、改变 TE 或生成滤波参数增强；如将来增加，应事前定义并对所有方法一致使用。普通图像增强不能证明具备 B0/TE/协议不变性。
6. **先冻结完整方案及所有 val 选模结果，再对预声明方法统一开展一次 test 评价**。必须保留全部预声明 λ 的结果，不能依据 test 选择“最佳 λ”再把同一 test 称为未参与开发。外部顺义、CSVD 仅用于外部评价，不用于新训练、归一化拟合或选模。

N_test 人既有结果已经进入返修分析并影响问题识别。保持其成员及训练隔离，并不使新比较自动成为完全没有历史反馈的首次盲测；论文应说明这是固定测试集上的预定返修再评价。若需要完全未受开发反馈影响的最终结论，另用未参与上述过程的数据。

## 4. 简单基线必须重新拟合并共用评价口径

已有 `train_mean`、`linear` 的任何旧拟合文件不能直接假定仍满足新 train/val 的边界。应使用新 manifest 重新拟合，只读取新 train，并保留拟合清单、输入 SHA256、单位、参考及共同空间标识。

逐体素平均和逐体素线性回归要求**已经完成并质量检查的共同解剖空间**；形状和 affine 一样不代表不同人的组织已对应。当前拟合入口不执行配准或重采样，不能把各人的原生空间直接按数组下标平均。若神经网络在原生空间训练，而空间基线需要模板空间，应事前固定映射和共同评价流程，使所有方法的结果在同一空间、同一支持域评价，并记录额外插值。不能在看到误差后专门改变某一种方法的网格。

评价支持域应事前固定：同一病例的所有方法使用同一个有效 mask、物理单位、QSM 参考以及 PSNR/SSIM range、HFEN 和 NMSE 定义。不允许针对某方法悄悄丢掉误差大的体素。若 test mask 超出基线训练覆盖，现有实现会拒绝运行，不会自动缩小该 mask 后继续评分；应报告未覆盖及缺失原因，再依预定协议处理，而不是把缺失当成零误差。

使用已有 `evaluate.py` 同口径生成 zero、train_mean、linear 和模型指标，并按受试者汇总。零图的 NMSE 在参考能量大于零时应为 1，这是核查 NMSE 的重要参照。保留负 ICC、未定义值、失败/缺失病例及实际 n，补充误差图、体素散点、ROI bias/LoA/ICC 与置信区间；相关性不能代替绝对一致性。单位、反归一化或参考零点未核实时，不能将原存储尺度诊断表标成正式 ppb 一致性结果。

## 5. filtered-SWI phase 对照的条件实验

Reviewer #2 Major 4 确实建议比较两类路线。所引 [Kames 等，Recovering SWI-filtered phase data using deep learning，MRM 2022，PMID 34611931，DOI 10.1002/mrm.29013](https://pubmed.ncbi.nlm.nih.gov/34611931/)使用实际滤波相位相关数据恢复局部场，再计算 QSM；它不是给 SWI 幅度网络换个名称。该路线具有相位信息，而当前 SWI 幅度/处理后 SWI 输入的信息条件不同。

开展数值对照前，需要同例真实导出的 filtered phase，确认其不是 phase mask 或 mIP；核实相位整数到弧度的缩放、符号、TE/echo、场强与 B0 方向、图像方向和滤波流程，并获得可核查的方法实现、权重和适用输入约定。随后才能预定其处理步骤，在共同受试者、配准、有效 mask、物理单位和 QSM 零参考下评价；不能用 test 调整相位缩放或恢复参数。

当前未把该外部方法作为可运行的训练配方，未从幅度图造相位、下载执行未经核实的相位重建流程或宣称该项完成。相位数据缺失时，应将数值对照登记为“数据条件未满足”，在回复中准确比较输入可用性、物理信息保留、适用场景及局限；不能以虚构 phase baseline 替代。仅有常规 SWI 而真实相位不可恢复，是说明本方法适用范围的关键条件。

## 6. 运行与交付检查

```bash
# 仅列出配方，不启动训练。
python -m swixqsm_minimal.train --list-models

# 从原 train/test 清单生成新 train/val；保护原 test。
python -m swixqsm_minimal.split_training \
  --manifest /prepared/original_train_test.csv \
  --output /prepared/new_split --val-fraction 0.1 --seed 42
```

正式训练在完成新分割对应的 QC 与独立性审计后执行。新入口 `--model all` 顺序运行四个主模型；`--model reviewer_suite` 顺序运行四个主模型、严格对抗消融和三个附加 λ 配方。全部使用同一新 manifest、验证政策和数据准备参数；具体必填参数及单模型、续训命令以新训练说明和 `--help` 为准。`--allow-unverified-*` 只记录未核实原因，不把数据变成审计通过，也不能据此补造论文证据。

交付论文结果时应逐项登记：是否实际训练、所用 split/seed/epoch、选模规则和 checkpoint、输入归一化及 QSM 参考、实际参数及更新预算、每队列病例数/缺失原因、统一指标和不确定性。代码通过合成测试，只证明实现的受控行为；真实性能、外部泛化和定量有效性必须由随后实际实验回答。
