FROM spark:4.0.0-scala2.13-java21-python3-ubuntu@sha256:23553639f445ec9983ae5f172fde61adac53269d5f80842055e5f2e2043fdb0b
USER root
RUN mkdir -p /home/spark && chown spark:spark /home/spark
RUN mkdir -p target/scala-2.13
COPY target/scala-2.13/fare-prediction_2.13-0.1.0-SNAPSHOT.jar target/scala-2.13/
USER spark
