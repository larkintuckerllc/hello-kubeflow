import org.apache.spark.sql.SparkSession

object FarePrediction {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Fare Prediction")
      .config("spark.hadoop.fs.s3a.access.key", "REPLACE") // Replace with your key
      .config("spark.hadoop.fs.s3a.secret.key", "REPLACE") // Replace with your key
      .config("spark.hadoop.fs.s3a.session.token", "REPLACE") // TODO
      .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
      .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
      .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider")
      .config("spark.hadoop.fs.s3a.connection.maximum", "1000")
      .config("spark.hadoop.fs.s3a.connection.timeout", "200000")
      .config("spark.hadoop.fs.s3a.connection.ttl", "200000")
      .getOrCreate()
    val df = spark.read
      .option("header", "true")
      .csv("s3a://hello-kubeflow/fare-prediction/chicago_taxi_train.csv")
    df.show()
    spark.stop()
  }
}
