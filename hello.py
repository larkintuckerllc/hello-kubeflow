from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("Hello World").getOrCreate()
rdd = spark.sparkContext.parallelize([1, 2, 3, 4, 5])
print("RDD count: ", rdd.count())
