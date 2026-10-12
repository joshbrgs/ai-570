# syntax=docker/dockerfile:1.7

# ---- uv binary (pinned) ----
FROM ghcr.io/astral-sh/uv:0.9.9 AS uv

# ---- Runtime: JupyterLab + TensorFlow/Keras ----
FROM python:3.12-slim-bookworm

# graphviz is required by keras.utils.plot_model (pydot shells out to `dot`).
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

# The venv lives OUTSIDE the bind-mounted repo so the host's .venv (macOS/Windows
# binaries) never shadows or collides with the Linux one inside the container.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/workspace \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Non-root user. On Linux hosts, pass your UID/GID so files written to the
# bind mount are owned by you (see README). Docker Desktop maps this automatically.
ARG UID=1000
ARG GID=1000
RUN groupadd --gid "${GID}" app \
    && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash app \
    && mkdir -p /opt/venv /workspace /home/app/.keras \
    && chown -R app:app /opt/venv /workspace /home/app

USER app
WORKDIR /workspace

# Install deps from the lockfile only (layer is cached until pyproject/uv.lock change).
# --frozen: fail rather than silently re-resolve if the lock is out of date.
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=${UID},gid=${GID} \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --group notebook

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--ServerApp.root_dir=/workspace"]
