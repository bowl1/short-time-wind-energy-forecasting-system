# 最小 Kubernetes 部署

只迁移 FastAPI 推理服务：1 个副本的 Deployment + ClusterIP Service。
模型随现有 Dockerfile 打包，训练和 MLflow 无需迁入集群，也无需 PVC。
需要已有 Kubernetes 集群、已配置的 kubectl，以及 Docker。
以下命令均在仓库根目录执行，资源部署到当前 context 的 namespace。

## 本地验证（已有 kind 集群）

```bash
docker build -f fastapi-app/Dockerfile -t wind-api:local .
kind load docker-image wind-api:local # 非默认集群添加 --name <集群名>
kubectl apply -k k8s
kubectl rollout status deployment/wind-api --timeout=300s
kubectl port-forward service/wind-api 8000:80
```

打开 http://localhost:8000/docs 。另一个终端验证：

```bash
curl --fail http://localhost:8000/ready
curl --fail -X POST 'http://localhost:8000/Predict/next%201h' \
  -H 'Content-Type: application/json' \
  -d '{"wind_speed":12.3,"wind_direction":210,"timestamp":"2025-03-30T14:06:02"}'
```

如果使用 minikube，将 `kind load docker-image wind-api:local` 替换为
`minikube image load wind-api:local`，其余步骤相同。Docker daemon 必须已启动，
且镜像需要加载到 kubectl 当前指向的集群。

## 已有远程集群

将镜像发布到节点可访问的仓库，并将 `k8s/deployment.yaml` 的 `image`
改成对应的完整地址和唯一版本标签，然后执行 `kubectl apply -k k8s`。
例如（先替换仓库地址；构建平台需匹配集群节点）：

```bash
docker build -f fastapi-app/Dockerfile -t YOUR_REGISTRY/wind-api:v1 .
docker push YOUR_REGISTRY/wind-api:v1
# 修改 deployment.yaml 的 image: YOUR_REGISTRY/wind-api:v1 后：
kubectl apply -k k8s
kubectl rollout status deployment/wind-api --timeout=300s
```

私有仓库需要配置集群的镜像拉取凭据。Service 默认仅集群内部可访问，
本地用上述 port-forward 验证；公网发布需要按集群环境配置 Ingress 或 LoadBalancer。
现有 `.github/workflows/deploy.yml` 仍部署 Docker 到 EC2，尚未切换到 Kubernetes。

## 检查与更新

`/Health` 检查进程存活，`/ready` 在模型成功加载后才返回 200；加载失败返回 503，
Pod 不会接收 Service 流量。启动探针最多等待约 5 分钟。
资源配置为初始值，需要按实际预测请求的内存和 CPU 使用量调整。

```bash
kubectl kustomize k8s # 本地渲染清单，无需连接集群
kubectl get pods -l app=wind-api
kubectl logs deployment/wind-api
kubectl describe deployment wind-api
```

后续更新使用新的镜像标签，修改清单后重新 apply。若重复使用本地 `:local` 标签，
需重新 build、kind load，再执行 `kubectl rollout restart deployment/wind-api`。
minikube 的 `minikube image load` 可能不会覆盖集群里已有的同名标签（不报错但仍是旧镜像），
可改用 `docker save wind-api:local | minikube ssh --native-ssh=false "docker load"`。
回滚可执行 `kubectl rollout undo deployment/wind-api`，并同步清单中的镜像版本。
卸载本次部署：`kubectl delete -k k8s`。

## 监控（Prometheus + Grafana）

应用在 `/metrics` 暴露请求数、延迟、状态码、`wind_model_loaded`（模型是否加载），
以及最近一批 24 小时预测（见上节）。`k8s/monitoring` 是独立的轻量部署，可选：

```bash
kubectl apply -k k8s/monitoring
kubectl rollout status deployment/prometheus --timeout=300s
kubectl rollout status deployment/grafana --timeout=300s
kubectl port-forward service/prometheus 9090:9090 # Status → Targets 应显示 wind-api UP
kubectl port-forward service/grafana 3000:3000    # 登录 admin/admin，打开 Wind API 看板
```

Prometheus 通过 Kubernetes 服务发现抓取带 `app=wind-api` 标签的 Pod，仅有本 namespace
读取 Pod 的 RBAC 权限。数据存于 emptyDir，Pod 重启会丢失，保留 2 天；Grafana 使用默认
账号密码，仅适合本地。卸载：`kubectl delete -k k8s/monitoring`。

## 定时预测（真实天气数据）

`k8s/forecast` 每小时（UTC 第 5 分钟）从 [Open-Meteo](https://open-meteo.com/)（非商业免费，无需 key）
拉取 Orkney（Kirkwall）从下一个整点 UTC 起的 24 条 10 米风速/风向预报，调用一次 `POST /predict/24h`
批量预测。预报数据约每小时更新，因此不更频繁地拉取。需要集群能访问外网。

- 所有时间统一为 UTC（训练数据来自 InfluxDB，同为 UTC），避免时区或夏令时导致小时特征错位。
- `/predict/24h` 只接受从下一个整点 UTC 起、连续的 24 个整点；任何一条天气缺失则整批不发布。
- 最近一次成功的批次保存在 API 内存中：`GET /forecast/latest` 返回它及 `age_seconds`，
  `/metrics` 持续暴露 `wind_predicted_power_kw{horizon="1".."24"}`、对应目标时间和生成时间。
  更新失败时保留旧批次，Grafana 的 Forecast age 超过 65 分钟变橙、超过 125 分钟变红。
- 批次只存在内存中：wind-api 重启后需等下一次定时任务，或手动运行一次。

```bash
kubectl apply -k k8s/forecast
kubectl create job wind-forecast-now --from=cronjob/wind-forecast # 立即运行一次
kubectl logs job/wind-forecast-now
```

配置使用 kubectl 内置的 [Kustomize](https://kubernetes.io/docs/tasks/manage-kubernetes-objects/kustomization/)
和 Kubernetes [健康探针](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)。
