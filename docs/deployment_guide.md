📦 AFNet 部署说明文档（Deployment Guide）
项目名称：AFNet — Electrogram-based Wavefront Prediction System
文档类型：部署说明书
维护人：Huiyi

🎯 1. 部署目标与形式
本项目的目标是将训练好的 AFNet 模型部署为以下两种形式：

类型	描述
✅ CLI 调用脚本	使用 Python 脚本从命令行输入 .h5 电图数据，输出预测图
✅ FastAPI 服务接口	启动本地 HTTP 接口，接收 JSON 输入电图，返回预测结果
⬜ Docker 封装（可选）	后续支持打包成可部署容器，方便迁移部署或云服务调用
📁 2. 文件目录结构（部署相关）
AFNew/
├── scripts/
│   ├── predict_cli.py            ← CLI 推理脚本（.h5 → 输出图）
│   └── deploy_api.py             ← FastAPI 接口服务
├── models/
│   └── best_model.pth            ← 已训练完成的模型权重
├── data/
│   └── test_h5/                  ← 待推理的测试电图样本
├── results/
│   └── predictions/              ← 存放预测输出图像或结果
├── config/
│   └── constants.py              ← 路径 & 电极坐标定义
├── requirements.txt              ← 环境依赖
└── Dockerfile (可选)            ← 容器封装说明
🧠 3. 输入输出规范
输入格式（统一为 .h5 或 JSON）：
.h5 文件（离线推理）

electrograms : shape = (19, 250)

JSON（API 推理）

{
  "electrograms": [[...], [...], ..., [...]]  // 19 通道电图，每通道长度 250
}
输出格式：
.png 激活图（保存到 results/predictions/）

JSON（API 输出）

{
  "activation_map": [[...], [...], ...],  // 100×100 激活图
  "activation_times": [...],              // 每个电极点预测时间（19,）
  "direction_vectors": [[...], ..., [...]] // 每点方向向量（19,2）
}
⚙️ 4. 推理脚本使用（predict_cli.py）
# CLI 用法：
python scripts/predict_cli.py --input data/test_h5/sample_00001.h5 --output_dir results/predictions/
功能说明：

加载模型 best_model.pth

加载 .h5 电图输入

输出：

可视化激活图 sample_00001_map.png

可视化方向向量图 sample_00001_vector.png

🌐 5. 启动 FastAPI 接口服务（deploy_api.py）
# 启动服务（默认本地 127.0.0.1:8000）：
uvicorn scripts.deploy_api:app --reload
访问接口
POST /predict

请求内容：

{
  "electrograms": [[...], ..., [...]]  // shape = (19, 250)
}
响应内容：

{
  "activation_map": [[...]],
  "activation_times": [...],
  "direction_vectors": [[...], ...]
}
你可以使用 Postman 或 Python requests 库发送请求。

🐳 6. Docker 封装（可选）
Dockerfile 示例：

FROM python:3.9

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["uvicorn", "scripts.deploy_api:app", "--host", "0.0.0.0", "--port", "8000"]
构建与运行：
docker build -t afnet_api .
docker run -p 8000:8000 afnet_api
✅ 7. 环境依赖（requirements.txt）
torch>=1.12
torchvision
numpy
scipy
pandas
h5py
matplotlib
fastapi
uvicorn
tqdm
🚧 8. 限制与未来部署拓展
限制	描述
当前仅支持 CPU 推理	GPU 加速支持后续可添加
推理接口为同步阻塞	可用 async 改造为异步处理
不支持上传 .h5 文件推理	仅 JSON 格式推理
缺乏身份认证	后续支持 OAuth2 / Token 验证接口

