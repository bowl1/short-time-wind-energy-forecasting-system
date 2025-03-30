# 基础镜像
FROM python:3.10

# 设置工作目录
WORKDIR /app

# 复制 requirements.txt 并先单独安装
COPY requirements.txt .

# 安装依赖（这一层会被 Docker 缓存，只要 requirements.txt 不变，就不会重新执行）
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件（放在安装依赖之后）
COPY app/ ./app/
COPY .env .env

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]