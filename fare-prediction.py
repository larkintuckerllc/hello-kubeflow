import boto3
import os
from pyspark.sql import SparkSession


S3_CLIENT = boto3.client("s3")
BUCKET_NAME = "hello-kubeflow"
S3_KEY = "fare-prediction/chicago_taxi_train.csv"
LOCAL_FILE_PATH = "/tmp/data/chicago_taxi_train.csv"

os.makedirs("/tmp/data", exist_ok=True)
S3_CLIENT.download_file(BUCKET_NAME, S3_KEY, LOCAL_FILE_PATH)

spark = SparkSession.builder.appName("Fare Prediction").getOrCreate()
df = spark.read.csv("data/chicago_taxi_train.csv", header=True, inferSchema=True)
df.printSchema()
df.show(5)
