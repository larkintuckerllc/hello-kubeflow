FROM spark:4.0.0-scala2.13-java21-python3-ubuntu@sha256:23553639f445ec9983ae5f172fde61adac53269d5f80842055e5f2e2043fdb0b
USER root
RUN pip install pipenv==2025.0.4
COPY Pipfile .
COPY Pipfile.lock .
RUN pipenv install
COPY hello.py .
USER spark
