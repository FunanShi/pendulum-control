FROM python:3.12-slim
# python3-tk: matplotlib TkAgg (interactive X11 windows). ffmpeg: FuncAnimation MP4 export.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# INVARIANT: WORKDIR must equal the compose.yaml bind-mount target (.:/work).
# `pip install -e` below records dpend as a .pth pointer to /work; if these two
# paths ever diverge, imports silently resolve to the stale copy baked here.
WORKDIR /work
COPY pyproject.toml README.md ./
COPY dpend/ dpend/
# Deps are physically installed into the image (changing them = rebuild).
# dpend itself is editable: the runtime bind mount supplies the live source.
RUN pip install --no-cache-dir -e ".[dev,ui,mpc]"
CMD ["bash"]
