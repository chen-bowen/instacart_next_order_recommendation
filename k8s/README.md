# Kubernetes Deployment

Deploy the Instacart Recommendation API to Kubernetes.

## Prerequisites

- `kubectl` configured for your cluster
- Docker image built and pushed to a registry
- Models and processed data (from data prep + training)

## 1. Build and push the image

```bash
# Build
docker build -t <your-registry>/instacart-rec-api:latest .

# Push (login to registry first if needed)
docker push <your-registry>/instacart-rec-api:latest
```

## 2. Update the deployment image

Edit `deployment.yaml` and set the container image:

```yaml
containers:
  - name: api
    image: <your-registry>/instacart-rec-api:latest
```

For a private registry, create an `imagePullSecret` and add it to the pod spec:

```yaml
spec:
  template:
    spec:
      imagePullSecrets:
        - name: regcred
```

## 3. Create namespace and PVCs

```bash
kubectl create namespace instacart-rec
kubectl apply -f pvc.yaml -n instacart-rec
```

## 4. Populate the PVCs with data

You must populate `instacart-processed-pvc` with at least `eval_corpus.json` (from data prep). Optionally populate `instacart-models-pvc` with your trained model.

### Option A: Copy from local via temporary pod

Apply the data-loader pod (see `data-loader-pod.yaml`), wait for it to be Running, then copy:

```bash
kubectl apply -f data-loader-pod.yaml -n instacart-rec
kubectl wait --for=condition=Ready pod/data-loader -n instacart-rec --timeout=60s

# Copy processed data (required)
kubectl cp ./processed/p5_mp20_ef0.1 instacart-rec/data-loader:/mnt/processed/p5_mp20_ef0.1 -n instacart-rec

# Copy models (optional; skip if using HuggingFace model)
kubectl cp ./models/two_tower_sbert instacart-rec/data-loader:/mnt/models/two_tower_sbert -n instacart-rec

kubectl delete pod data-loader -n instacart-rec
```

### Option B: Use the Hugging Face model (no local model needed)

If you use the pre-trained model from Hugging Face, you only need the processed corpus. Update the ConfigMap in `deployment.yaml`:

```yaml
data:
  MODEL_DIR: "chenbowen184/instacart-two-tower-sbert"
  CORPUS_PATH: "/app/processed/p5_mp20_ef0.1/eval_corpus.json"
```

Then you can skip populating `instacart-models-pvc` (it can stay empty). You still need `eval_corpus.json` and `eval_queries.json` in the processed volume.

### Option C: Init container or Job

For automation, use an init container or a one-off Job that pulls from object storage (S3, GCS) or a shared NFS, then copies into the PVC mount.

## 5. Apply the deployment

```bash
kubectl apply -f deployment.yaml -n instacart-rec
```

## 6. Verify and access

```bash
# Check pods
kubectl get pods -n instacart-rec

# Check logs (model loading may take 1–2 minutes)
kubectl logs -f deployment/instacart-rec-api -n instacart-rec

# Port-forward to test locally
kubectl port-forward svc/instacart-rec-api 8000:8000 -n instacart-rec
```

Then open http://localhost:8000/docs.

## 7. Optional: Ingress

To expose the API externally, add an Ingress:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: instacart-rec-api
  annotations:
    # Add your ingress controller annotations (e.g. nginx, traefik)
spec:
  rules:
    - host: rec-api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: instacart-rec-api
                port:
                  number: 8000
```

## Configuration

| ConfigMap key    | Description                          | Default                                  |
| -----------------| ------------------------------------ | ---------------------------------------- |
| `MODEL_DIR`      | Model path or HuggingFace model ID   | `/app/models/two_tower_sbert/final`       |
| `CORPUS_PATH`    | Path to eval_corpus.json             | `/app/processed/p5_mp20_ef0.1/eval_corpus.json` |
| `FEEDBACK_DB_PATH` | SQLite path for feedback           | `/app/data/feedback.db`                  |
| `INFERENCE_DEVICE` | `cuda`, `mps`, or `cpu`            | `cpu`                                    |

For GPU inference, use a CUDA base image, set `INFERENCE_DEVICE=cuda`, and add GPU resource requests to the deployment.
