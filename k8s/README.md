# Kubernetes Runbook

This folder contains a `kubectl` + `kustomize` deployment for the Cancer Hospital stack.

## Prerequisites

- A working Kubernetes cluster
- Best local experience: **Docker Desktop Kubernetes**
- The host-side model API running on port `8001`
- A local Docker image tagged as `cancer-hospital-app:latest`

Before running any `kubectl` deployment commands, enable Kubernetes in **Docker Desktop -> Settings -> Kubernetes**.

If you are using Docker Desktop Kubernetes:

```bash
kubectl config use-context docker-desktop
kubectl get nodes
```

For local image reuse, Docker Desktop's `kubeadm` provisioner works with the Docker image store, while `kind` does not. If Docker Desktop is currently set to `kind`, switch the provisioner to `kubeadm` before deploying this stack.

## Deploy

```bash
kubectl apply -k k8s/
```

## Watch resources

```bash
kubectl get all -n cancer-hospital
kubectl get pods -n cancer-hospital -w
```

## Access services

```bash
kubectl port-forward -n cancer-hospital svc/app 7860:7860
kubectl port-forward -n cancer-hospital svc/prometheus 9090:9090
kubectl port-forward -n cancer-hospital svc/grafana 3000:3000
kubectl port-forward -n cancer-hospital svc/loki 3100:3100
```

## Scale the UI

```bash
kubectl scale deployment app -n cancer-hospital --replicas=3
kubectl rollout status deployment/app -n cancer-hospital
```

## Restart a workload

```bash
kubectl rollout restart deployment/app -n cancer-hospital
kubectl rollout restart deployment/prometheus -n cancer-hospital
kubectl rollout restart deployment/grafana -n cancer-hospital
```

## Logs

```bash
kubectl logs -n cancer-hospital deployment/app -f
kubectl logs -n cancer-hospital daemonset/promtail -f
```

## Cleanup

```bash
kubectl delete -k k8s/
```
