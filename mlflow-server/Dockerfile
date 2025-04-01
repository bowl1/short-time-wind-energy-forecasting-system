FROM python:3.10

WORKDIR /mlflow

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 开放端口
EXPOSE 5000

CMD ["mlflow", "server"]
