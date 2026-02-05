# gpu-embedding-platform

## What is this
Single-node RTX 4090 multi-model embedding serving stack
Models: Qwen3-Embedding-8B (vLLM), PubMedBERT (vLLM/ORT), ColQwen2 (ORT+TRT)

## Architecture diagram（放一張圖）
```text
gpu-embedding-platform/
  README.md
  LICENSE
  .gitignore
  .editorconfig

  docs/
    architecture.md
    api.md
    models.md
    runbook.md              # oncall/故障排查
    benchmarks.md

  services/
    router/                 # FastAPI gateway/router
      app/
        main.py
        routing.py
        batching.py         # micro-batch（可選）
        limits.py           # concurrency / queue
        auth.py             # API key（可選）
        metrics.py          # Prometheus metrics
      tests/
      Dockerfile
      pyproject.toml

    vllm-qwen3-embed/        # vLLM service: Qwen3-Embedding-8B
      Dockerfile
      start.sh              # vllm serve ... --runner pooling
      config/
        vllm.yaml           # 你可把參數集中
      README.md

    vllm-pubmedbert-embed/         # vLLM service: pubmedbert-base-embeddings (pooling)
      Dockerfile
      start.sh
      config/
      README.md

    vllm-biomednlp-biomedbert/         # vLLM service: biomednlp-biomedbert (pooling)
      Dockerfile
      start.sh
      config/
      README.md

    orttrt-colqwen2/         # ORT + TensorRT EP service
      app/
        main.py             # FastAPI/gRPC server
        preprocess.py       # image/pdf page -> tensor
        model.py            # ORT session, TRT EP options
        metrics.py
      export/
        export_to_onnx.py   # 轉 ONNX（或你也可用 notebook）
        calibrate_int8.py   # 可選：INT8 校準
      Dockerfile
      README.md

  infra/
    compose/
      docker-compose.yml
      prometheus.yml
      grafana/
        dashboards/
        provisioning/
    k8s/                    # 你要升級到 k3s 再用
      router-deploy.yaml
      vllm-qwen3.yaml
      vllm-pubmedbert.yaml
      orttrt-colqwen2.yaml

  bench/
    load/
      locustfile.py         # 或 k6 script
      k6.js
    datasets/
      sample_queries.jsonl  # 你自製測試集（別放敏感資料）
    scripts/
      benchmark_matrix.py   # 跑 32/128/512 tokens + batch sweep
      report.py             # 輸出 markdown 報告

  ops/
    scripts/
      build_all.sh
      start_stack.sh
      stop_stack.sh
      smoke_test.sh
    ci/
      github-actions.yml    # 或放 .github/workflows

  .github/
    workflows/
      ci.yml                # lint/test/build
      docker.yml            # build/push images（可選）
```

## Quickstart（docker compose 一鍵起）

## API（OpenAI-compatible /v1/embeddings + router 的 model= 路由）

## Observability（Grafana dashboard 截圖 + 指標）

## Benchmarks（你跑出來的吞吐/延遲表）

## Roadmap（k3s、A/B、autoscaling、model registry）