FROM spark:4.0.0-scala2.13-java21-python3-ubuntu@sha256:23553639f445ec9983ae5f172fde61adac53269d5f80842055e5f2e2043fdb0b
USER root
COPY deps/ ./deps/
COPY mymodule.py .
COPY hello.py .
# USER spark
