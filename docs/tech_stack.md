⚙️ 技术栈文档（Tech Stack）
项目名称：AFNet — 心房电图的激活图与传播方向预测系统
文档类型：系统技术栈说明文档
维护人：Huiyi

🧠 一、开发语言与主框架
类别	使用
开发语言	Python 3.9+
深度学习框架	PyTorch
IDE / 编辑器	Spyder（主要调试） / Git + VSCode（版本控制与规范）
可视化	Matplotlib、TensorBoard
📦 二、关键依赖库与用途
库名	作用
torch / torchvision	模型构建、训练流程、ResNet backbone
torch.utils.data	自定义数据加载类 ElectrogramDataset
h5py	存储合成数据（.h5 格式）
numpy / scipy	信号处理（滤波、Hilbert 变换）、激活计算
pandas	读取电极坐标点（.csv）
matplotlib.pyplot	可视化波形、电激活图等
glob / os	文件读写、路径拼接
tqdm	显示训练进度条
csv	写入训练日志文件
tensorboardX or torch.utils.tensorboard	可视化训练过程
✅ 三、首选实现策略
类型	指导原则
🔍 网络结构	首选 torchvision.models.resnet18 为主干网络，不使用 ViT/Transformer/GNN 结构（未来可扩展）
⚙️ 激活图模拟	合成数据基于 Gaussian filter + virtual source (focal/rotor/planar)
📊 损失函数	优先使用自定义 directional cosine loss + 多目标 loss (activation map + time)
📁 路径管理	所有路径统一以 BASE_DIR 开始，兼容 CLI + IDE 调试
📂 数据格式	所有样本为 .h5 格式，字段名固定为：electrograms, activation_map, activation_times
🔗 四、未来可能接入的技术栈（可选项）
类型	工具
🚀 GNN 支持	PyTorch Geometric（计划用于电极连接图上的 propagation learning）
🧪 自动超参数搜索	Optuna / Ray Tune（后续扩展）
🧠 模型替代方案	AFNetTransformer / Diffusion-based 模型架构
📊 可视化增强	Plotly / Streamlit 前端交互（用于模型解释）
📡 临床集成	接入实际临床 electrogram 数据接口（ECGi 或 Mapping System 数据）
💡 五、配置管理方式
内容	位置
所有静态路径	config/constants.py
所有训练参数	config/train_config.py
实验目录统一管理	自动使用时间戳生成 results/run_YYYYMMDD_HHMMSS 目录
❗ 六、禁止使用 / 避免混用的技术
禁止/不推荐使用项	理由
TensorFlow / Keras	本项目为纯 PyTorch 实现
sklearn.model_selection	不涉及分类任务，不使用交叉验证
matplotlib.animation	会造成训练过程变慢，不适合大规模可视化