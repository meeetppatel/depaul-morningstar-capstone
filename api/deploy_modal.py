"""
Deploy GECS API to Modal.com (GPU serverless)
Install: pip install modal
Setup  : modal token new
Deploy : python deploy_modal.py
"""
import modal
from pathlib import Path

# ── Modal app ──────────────────────────────────────────────
app = modal.App("gecs-classification-api")

# GPU image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.111.0",
        "uvicorn[standard]==0.29.0",
        "torch==2.3.0",
        "transformers==4.40.0",
        "anthropic>=0.25.0",
        "scikit-learn==1.4.2",
        "numpy==1.26.4",
        "pydantic==2.7.1",
    )
)

# Mount artifacts
artifacts_mount = modal.Mount.from_local_dir(
    "artifacts",
    remote_path="/app/artifacts",
)

@app.function(
    image       = image,
    gpu         = "A10G",           # cheapest GPU on Modal
    memory      = 16384,
    mounts      = [artifacts_mount],
    secrets     = [modal.Secret.from_name("anthropic-secret")],
    container_idle_timeout = 300,   # keep warm 5 min
)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/app")
    from main import app
    return app
