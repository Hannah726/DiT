# RawDiff 环境配置指南

## 环境要求

- **Python**: 3.9
- **CUDA**: 11.8
- **PyTorch**: 2.0.1

## 环境适配性分析

✅ **完全兼容**！您的环境配置是适配的：

1. **Python 3.9** ✅
   - PyTorch 2.0.1 完全支持 Python 3.9

2. **CUDA 11.8** ✅
   - PyTorch 2.0.1 原生支持 CUDA 11.8

3. **依赖包版本** ✅
   - 所有列出的依赖包版本要求都与 Python 3.9 和 PyTorch 2.0.1 兼容

## 安装步骤

### 方法 1: 使用 Conda（推荐）

```bash
# 1. 创建新的 Conda 环境
conda create -n rawdiff_env python=3.9 -y

# 2. 激活环境
conda activate rawdiff_env

# 3. 安装 PyTorch（支持 CUDA 11.8）
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. 安装其他依赖
pip install -r requirements.txt
```

### 方法 2: 使用 pip

```bash
# 1. 创建虚拟环境
python3.9 -m venv rawdiff_env
source rawdiff_env/bin/activate  # Linux/Mac
# 或
rawdiff_env\Scripts\activate  # Windows

# 2. 安装 PyTorch（支持 CUDA 11.8）
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 3. 安装其他依赖
pip install -r requirements.txt
```

## 验证安装

```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"GPU 数量: {torch.cuda.device_count()}")
```

应该输出：
- PyTorch 版本: 2.0.1+cu118
- CUDA 可用: True
- CUDA 版本: 11.8

## 注意事项

1. 确保系统已安装 CUDA 11.8 驱动
2. 如果使用服务器，确保 GPU 可用
3. 建议使用 Conda 管理环境，可以更好地处理 CUDA 依赖

