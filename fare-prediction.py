from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Fare Prediction").getOrCreate()
