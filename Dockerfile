# Minimal, offline, non-root image for slotfit.
FROM python:3.12-slim

# Create an unprivileged user to run as.
RUN useradd --create-home --uid 10001 runner

WORKDIR /app
COPY --chown=runner:runner . /app

# Install the package (numpy + pydantic only; no network needed at runtime).
RUN pip install --no-cache-dir . \
    && rm -rf /root/.cache

USER runner

# Default: print the benchmark without writing files.
ENTRYPOINT ["slotfit"]
CMD ["bench", "--no-write", "--instances", "50"]
