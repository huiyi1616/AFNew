🛠 实施计划文档（Implementation Plan）
项目名称：AFNet
文档类型：分阶段任务清单（50 步以上）
作者：Huiyi

📌 一、准备阶段（基础环境 & 项目结构）
✅ 创建项目根目录 AFNew/

✅ 创建子目录：config/, data/, models/, loss/, scripts/, results/, logs/, docs/

✅ 初始化 Git 仓库，配置 .gitignore

✅ 在 config/constants.py 中配置路径、常量、坐标等全局参数

✅ 在 config/train_config.py 中定义 batch_size、epochs、learning_rate、device 等训练参数

✅ 创建 triElectrodeFlat_Points.csv，准备好电极坐标数据

✅ 安装依赖：PyTorch、numpy、scipy、h5py、pandas、matplotlib、tqdm、tensorboard

📌 二、数据合成与增强阶段（真实波形 → 合成数据）
✅ 编写 generate_electrogram() 函数（支持 focal, rotor, planar 模式）

✅ 加入真实心房电图窗口数据 .mat 文件读取与 peak phase 对齐逻辑

✅ 实现 Hilbert + bandpass 滤波处理函数（在 utils/signal_processing.py）

✅ 加入 activation time 插值采样逻辑（从 activation map → electrode time）

✅ 编写 real_data_aug_generate.py，支持批量合成样本并保存 .h5

✅ 每个 .h5 包含：electrograms, activation_map, activation_times, source_position, activation_type

✅ 在合成数据中添加随机噪声增强（Gaussian 低频噪声）

✅ 测试合成数据生成流程，确保 waveform 与 activation map 同步

📌 三、数据加载阶段（构建 PyTorch Dataset）
✅ 编写 ElectrogramDataset(torch.utils.data.Dataset)

✅ 支持 train/validation 分别加载

✅ 加载 .h5 文件并统一 tensor 类型、维度

✅ 打印一个 batch 样本，验证数据管线通畅

📌 四、模型搭建阶段（AFNet 架构）
✅ 编写 AFNetResNet(nn.Module)

✅ 将 ResNet18 替换为 19 通道输入（conv1 重定义）

✅ 添加共享特征层（512 → 1024）

✅ 输出两个分支：activation_map（100x100）、activation_times（19）

✅ 封装 forward(x) 返回两个输出

✅ 编写模型初始化与 to(device) 测试代码

📌 五、方向向量模块开发（fast compute）
✅ 编写 fast_compute_direction_vectors(activation_times, coords)

✅ 使用邻接半径（欧氏距离 < 5mm）筛选邻居

✅ 使用传播时间差 × 单位方向向量 → 得到方向向量

✅ 对每个点方向向量求平均作为 supervision target

📌 六、损失函数模块开发
✅ 编写 directional_vector_loss(pred, true)

✅ 使用余弦相似度 loss：1 - mean(cosine_similarity)

✅ 与 map MSE / time MSE 一起组成多目标 loss

✅ 在 train/evaluate 函数中整合 3 种 loss

📌 七、训练脚本主逻辑开发
✅ 编写 train() 函数（for loop, loss, backward）

✅ 编写 evaluate() 函数（no_grad, 同样 loss）

✅ 每个 epoch 打印训练与验证指标

✅ 保存 best 模型（基于 direction score 或 time MAE）

✅ 保存 final 模型

✅ 写入 CSV 日志（记录 epoch, loss, score）

✅ 写入 TensorBoard 可视化 log

📌 八、可视化模块开发（分析与展示）
✅ Electrogram 曲线图（19 通道叠加图）

✅ Activation Map 热图 + 源点位置

✅ Direction Vector 箭头图（使用 quiver）

🕐 保存每个验证样本的预测图（planar only / rotor only）

🕐 统计每类样本的得分（按 activation_type）

📌 九、扩展任务（计划进行中）
🕐 加入模型预测结果的物理约束 loss（Eikonal loss）

🕐 支持使用 Transformer / Diffusion 模型替代 ResNet

🕐 尝试电极图上的 GNN 推理（使用 torch_geometric）

🕐 加入真实病人数据推理模块（不需 ground-truth activation map）

🕐 集成前端 Streamlit 可视化工具

✅ 十、发布 & 实验管理
✅ 自动创建 log、result、model 保存目录（带时间戳）

✅ 创建实验目录：results/run_YYYYMMDD_HHMMSS

✅ 保存 config 文件副本到每次实验目录

✅ 使用 Git 分支 feature/... 管理新功能

✅ 每完成一阶段任务，写一次 docs/system_log.md

