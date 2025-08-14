import org.apache.spark.sql.SparkSession

object FarePrediction {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("Fare Prediction").getOrCreate()
    val rdd = spark.sparkContext.parallelize(Array(1, 2, 3, 4, 5))
    println(s"RDD count: ${rdd.count()}")
    spark.stop()
  }
}
