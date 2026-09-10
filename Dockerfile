# SwarmSim -- portable container build.
#
# WHY this exists alongside render.yaml: render.yaml deploys SwarmSim on
# Render's native Python runtime (no container build needed there). This
# Dockerfile is for everywhere else that wants a container image directly
# -- Google Cloud Run (`gcloud run deploy --source .` builds this exact
# file), Railway, Fly.io, a plain VPS, or a local `docker run`. It needs
# zero changes for Cloud Run specifically: the CMD already binds
# 0.0.0.0 and reads $PORT, which Cloud Run injects into the container
# at runtime (it does not use the ENV PORT=8000 default below except
# for a plain local `docker run` with no -e PORT set).
FROM python:3.11-slim

WORKDIR /app

# libgomp1 is scikit-learn's OpenMP runtime dependency; python:slim doesn't
# ship it, so imports fail without this on a truly minimal base image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

# Shell form so ${PORT} is expanded -- most hosts (Render, Railway, Fly)
# inject PORT at runtime rather than at build time.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
