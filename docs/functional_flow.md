 功能流程文档（中文版本）
AFNet – 心房电图激活图预测系统

🧩 模块总览
整个项目共包含以下 5 个主要模块：

数据增强与合成模块：从真实心房电图中提取峰值窗，生成三类合成数据。

数据加载模块（Dataset + DataLoader）：将合成数据组织为 PyTorch 可训练的数据形式。

深度神经网络模型（AFNetResNet）：基于 ResNet 架构输出两个目标。

损失函数模块：三个目标同时优化，包括方向性监督。

训练与评估脚本：运行主训练逻辑，保存日志、模型、评估指标。

🔄 功能流程图（文字版）

真实数据片段 (Peak_Window) + 电极坐标
          ↓
generate_electrogram(mode="planar")       ← 模拟 focal / rotor / planar 激活场
          ↓
real_data_aug_generate.py                 ← 循环生成多个样本，保存为 .h5
          ↓
ElectrogramDataset                        ← 从 train/validation 加载数据
          ↓
AFNetResNet                               ← 模型前向传播：
             ↳ activation_map (100x100)
             ↳ activation_times (19通道)
          ↓
fast_compute_direction_vectors            ← 推出方向向量 (19, 2)
          ↓
directional_vector_loss                   ← 使用 cosine 相似度监督方向预测
          ↓
train(), evaluate()                       ← 同时优化 map + time + direction
          ↓
TensorBoard + CSV + 模型文件             ← 日志保存、模型保存、可视化准备
🧠 页面 / 接口说明（AI友好描述）
1. real_data_augmentation_synthesis.py
功能：输入真实数据，合成电激活图与合成 electrograms。

模式：支持 focal / rotor / planar 三种模式。

输出：activation_time（激活图）、electrograms（19通道）、source（传播源点）

2. real_data_aug_generate.py
功能：批量生成训练 / 验证数据。

步骤：

循环调用 generate_electrogram(...)

将每个样本保存为 .h5 文件（包含激活图、电图信号、源点、类型标签）

输出路径：data/dataloader_data/train/ 和 .../validation/

3. train_on_synthesis.py
功能：主训练脚本，包含模型创建、数据加载、训练与验证。

模型结构：

输入维度：19 通道 electrogram，长度 400

输出：

activation_map：尺寸为 100x100

activation_times：19 个电极点的激活时间

损失函数：

MSE for map

MSE for activation time

Directional cosine loss（用来监督传播方向）

4. evaluate() 函数
功能：在验证集上评估模型性能。

输出指标：

Activation Time MAE

Activation Map MAE

Direction Vector Score（cosine 相似度）

5. 可视化部分（后续脚本）
Electrogram 曲线叠加图

激活图 + 电极编号 + 源点位置 + 方向箭头

支持后期添加方向向量 quiver 图

✅ 最后输出内容路径
内容	路径
模型权重	models/best_model.pth
日志记录	results/training_log.csv
TensorBoard	logs/AFNetResNet_xxxx/
合成数据	data/dataloader_data/train/