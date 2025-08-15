import org.apache.spark.sql.SparkSession

object FarePrediction {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Fare Prediction")
      .config("spark.hadoop.fs.s3a.access.key", "REPLACE") // Replace with your key
      .config("spark.hadoop.fs.s3a.secret.key", "REPLACE") // Replace with your key
      .getOrCreate()
    val df = spark.read
      .option("header", "true")
      .csv("s3a://hello-kubeflow/fare-prediction/chicago_taxi_train.csv")
    df.show()
    spark.stop()
  }
}
