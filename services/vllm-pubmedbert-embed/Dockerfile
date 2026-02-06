FROM vllm/vllm-openai:v0.10.2-x86_64

RUN python3 -m pip install --no-cache-dir pyyaml

WORKDIR /app

COPY start.sh /app/start.sh
COPY config/vllm.yaml /app/config/vllm.yaml

RUN chmod +x /app/start.sh

EXPOSE 8000

ENTRYPOINT ["/app/start.sh"]