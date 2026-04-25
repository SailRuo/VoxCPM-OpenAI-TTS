FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# 设置工作目录
WORKDIR /app

# 安装系统音频依赖和 Git
# 使用国内镜像源加速 apt-get
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN pip install --no-cache-dir --upgrade pip

# 安装 VoxCPM 和必备的 Linux 加速库 Triton, 以及 API 服务依赖
RUN pip install --no-cache-dir voxcpm triton fastapi uvicorn

# 预创建输出目录
RUN mkdir -p out

# 暴露端口
EXPOSE 8000

# 默认启动 API 服务
CMD ["python", "api_server.py"]
