# APEX-ORACLE full-app container: real backend (engine + AI model + real TESS) + UI
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY app/inspector ./app/inspector

# Package + real-data deps (lightkurve for real TESS, wotan for detrending).
# TLS is optional and guarded in code, so we skip it to keep the build light.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . lightkurve wotan

# Smoke-test the install at build time (synthetic, no network)
RUN apex-oracle demo --kind planet

ENV APEX_UI_DIR=/app/app/inspector
EXPOSE 8800

# Serve the UI at / and the JSON API at /analyze on one origin.
# Binds $PORT if the host sets it (Render/Railway/Fly), else 8800.
ENTRYPOINT ["apex-oracle"]
CMD ["serve", "--host", "0.0.0.0"]
