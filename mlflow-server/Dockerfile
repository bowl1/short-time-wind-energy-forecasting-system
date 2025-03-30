FROM python:3.10

WORKDIR /mlflow

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制启动脚本和 mlruns
COPY start.sh .
RUN chmod +x start.sh

COPY mlruns/ ./mlruns/

# 开放端口
EXPOSE 5000

# 启动 MLflow Server
CMD ["bash", "start.sh"]