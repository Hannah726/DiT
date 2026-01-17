# EHR Diffusion 架构 Pipeline 总结

## 整体架构

本系统采用基于RQ-VAE代码的扩散模型生成EHR数据，核心流程分为编码-扩散-解码三个阶段。

## 数据流程

### 1. 预处理阶段
- **输入**：原始EHR tokens（input_ids, type_ids, dpe_ids）
- **编码**：通过预训练RQ-VAE将tokens编码为离散codes（每个事件8个code，codebook_size=1024）
- **输出**：离散codes矩阵 (N, max_events, 8)

### 2. 训练阶段
**编码层（CodeEmbedder）**：
- 将离散codes (B, N, 8) 通过embedding查找和聚合（mean/sum/max）转换为连续latent (B, N, latent_dim)
- 可加载预训练RQ-VAE codebook权重

**扩散去噪层（DiT）**：
- 在latent空间进行扩散：随机采样timestep t，添加高斯噪声得到noisy_latent
- DiT Transformer使用时间条件（time_ids作为离散多位数token）和扩散timestep进行去噪
- 通过自注意力+交叉注意力机制融合时间信息，预测噪声

**解码层（CodeDecoder）**：
- 将去噪后的latent解码为离散codes预测
- 使用多个MLP头分别预测每个code位置

**损失计算**：
- 扩散损失：预测噪声与真实噪声的MSE（支持mask）
- 可选辅助损失：code重建的交叉熵损失

### 3. 生成阶段（DDIM采样）
- 从随机噪声开始，逐步去噪
- 使用DDIM scheduler进行确定性/随机性采样
- 最终解码为codes，可选通过RQ-VAE解码器还原为tokens

## 核心组件

- **EHRDiffusion**：主模型，整合三个组件
- **CodeEmbedder**：codes → latent编码器
- **DiT**：带时间条件的扩散Transformer
- **CodeDecoder**：latent → codes解码器
- **EHRTrainer**：训练循环、验证、检查点管理
- **DDIMCodesSampler**：生成采样器

## 关键特性

- 离散-连续混合：在离散codes和连续latent间转换，结合两种表示优势
- 时间条件：使用离散时间token（如720min→[7,2]）进行条件生成
- 可扩展：支持梯度累积、混合精度、分布式训练
