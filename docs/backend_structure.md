📂 后端结构文档（Backend Structure）
项目名称：AFNet — 心房电图的激活图与传播方向预测系统
文档类型：后端结构说明文档
维护人：Huiyi

🧱 一、核心数据格式
本系统的所有训练和验证样本，均保存在 .h5 文件中，每个文件对应一个样本，字段如下：

字段名	形状	类型	说明
electrograms	(19, 250)	float32	19 通道的合成电图，每通道一个波形
activation_map	(100, 100)	float32	合成的 2D 激活时间场，用于监督模型预测
activation_times	(19,)	float32	每个电极点的激活时间
activation_type	标量字符串	bytes	'focal', 'rotor', 'planar'
source_position	(2,)	float32	合成数据中传播源的坐标 (x, y)
📁 二、数据结构与目录布局
数据集按任务划分为 train/ 与 validation/ 两部分：

data/
├── dataloader_data/
│   └── synthesis_data/
│       ├── train/
│       │   ├── sample_00001.h5
│       │   ├── sample_00002.h5
│       │   └── ...
│       └── validation/
│           ├── sample_00001.h5
│           └── ...
你也可以使用 glob 加载所有文件：


import glob
train_files = glob.glob("data/dataloader_data/synthesis_data/train/*.h5")
🗂 三、电极位置信息（电极拓扑结构）
电极布局数据以 .csv 存储，加载后用于：

定位每个电极在 2D 空间中的位置（用于激活时间采样）

计算传播方向向量

文件	路径	字段
电极坐标	data/triElectrodeFlat_Points.csv	每行两个值：x, y
电极邻接图（可选）	triElectrodeFlat_ConnectivityList.csv	电极对列表，后期用于 GNN
🧠 四、方向向量计算逻辑
传播方向 supervision 使用如下步骤：


每个样本：
  1. 输入电极坐标 (19, 2)
  2. 输入激活时间 (19,)
  3. 对每个点，寻找邻居（欧氏距离 < 5mm）
  4. 计算方向差值向量（时间差 × 方向单位向量）
  5. 汇总得到每个电极点的“传播方向”
方向向量作为监督目标，与模型输出做余弦相似度损失（directional_vector_loss）

🛠 五、模型输出结构
模型 AFNetResNet 输出两个预测量：

输出名称	形状	含义
activation_map_pred	(B, 100, 100)	波前传播的激活图
activation_times_pred	(B, 19)	每个电极点的激活时间预测值
这些结果将参与 3 种损失函数：

MSE(activation_map_pred, activation_map_gt)

MSE(activation_times_pred, activation_times_gt)

Directional Cosine Loss(pred_dirs, true_dirs)

🔐 六、边界条件处理（Edge Cases）
情况	当前处理方式
某个电极无信号 / 空值	不出现，合成数据中强制保证 19 通道完整
相邻电极太少（<2）	对方向向量归一化时用 clamp(min=1.0) 避免除零
波形长度小于窗口	使用 zero-padding + window shifting 保证统一长度
📌 七、未来拓展结构预留
项目	描述
GNN 支持	可引入电极连接图 .csv 用作邻接矩阵
多导图预测	future: 每个 sample 支持多个方向场预测
多病人适配	future: 将数据按 patient_id 分类存储
真实 EGM 映射	future: 添加真实 mapping data loader + 对应采样逻辑