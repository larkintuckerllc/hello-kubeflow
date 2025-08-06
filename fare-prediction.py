import os
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Fare Prediction").getOrCreate()
# df = spark.read.csv("data/chicago_taxi_train.csv", header=True, inferSchema=True)
# df.printSchema()
# df.show(5)
