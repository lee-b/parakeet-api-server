FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y ffmpeg

RUN pip install --upgrade pip && pip install uv

WORKDIR /app



FROM base AS build

COPY README.md pyproject.toml src/ ./

RUN uv build



FROM base AS runtime

COPY --from=build /app/dist/*.whl /app/dist/

RUN uv tool install /app/dist/*.whl

RUN rm -rf /app/dist

EXPOSE 8000

ENV PATH=/root/.local/bin:$PATH

# Run the server directly
CMD ["parakeet-api-server", "--cpu"]
