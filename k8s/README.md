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
回滚可执行 `kubectl rollout undo deployment/wind-api`，并同步清单中的镜像版本。
卸载本次部署：`kubectl delete -k k8s`。

配置使用 kubectl 内置的 [Kustomize](https://kubernetes.io/docs/tasks/manage-kubernetes-objects/kustomization/)
和 Kubernetes [健康探针](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)。
