评估指标说明文档（Evaluation Metrics）
文档名称：docs/evaluation_metrics.md
说明：用于量化模型训练与推理质量的核心指标

✅ 指标清单
指标名称	含义	公式	说明
Activation Time MAE	平均预测激活时间误差	mean(abs(pred - true))	单位：归一化 scale (0~1)
Activation Map MAE	2D map 的平均误差	mean(abs(pred_map - true_map))	反映时空传播图一致性
Direction Vector Score	向量方向一致性分数	mean(cos(pred_dir, true_dir))	值域：0~1，越大越好
Total Loss	总损失	w1*MSE(map) + w2*MSE(time) + w3*dir_loss	当前设置为 0.1+0.1+0.8 权重
Direction Vector Loss	1 - cosine_similarity	1 - mean(cos(pred, true))	越小越好，越接近0越一致
🧠 使用建议
当 Direction Score ≥ 0.90 可认为模型对传播趋势掌握良好

评估指标建议绘图呈现每个 epoch 的走势（已接入 TensorBoard）