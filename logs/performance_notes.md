 系统性能分析文档（Performance Notes）
文档名称：docs/performance_notes.md
描述：说明系统运行效率、推理速度、资源占用与优化潜力

💻 训练与推理资源需求
项目	当前表现
单次训练时长（20 epochs）	~12 分钟（NVIDIA T4）
Batch Size	32
单次 forward 推理耗时	~14ms/sample（CPU）
GPU 推理耗时	~3ms/sample
显存使用量（训练）	~2.4GB（19通道输入）
CPU 占用	高峰不超过 40%
支持并发	FastAPI 版本支持单线程阻塞，未来可拓展为 async 异步接口
⚙️ 加速策略建议（未来优化）
✅ 加入 ONNX 导出 + ONNXRuntime 加速部署

✅ 使用 TensorRT 在嵌入式部署中获得最佳性能

✅ 将模型简化为轻量版本（可换 backbone 为 MobileNet）

✅ 对数据读取使用 num_workers > 4 充分利用 IO 并发

⬜ 使用 torch.compile 进行训练图优化（PyTorch 2.0+）