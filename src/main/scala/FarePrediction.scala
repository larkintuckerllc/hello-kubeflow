import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.functions.when
import org.apache.spark.sql.functions.rand

object FarePrediction {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Fare Prediction")
      .config("spark.hadoop.fs.s3a.access.key", sys.env.getOrElse("AWS_ACCESS_KEY_ID", sys.error("AWS_ACCESS_KEY_ID environment variable is required")))
      .config("spark.hadoop.fs.s3a.secret.key", sys.env.getOrElse("AWS_SECRET_ACCESS_KEY", sys.error("AWS_SECRET_ACCESS_KEY environment variable is required")))
      .config("spark.hadoop.fs.s3a.session.token", sys.env.getOrElse("AWS_SESSION_TOKEN", sys.error("AWS_SESSION_TOKEN environment variable is required")))
      .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider") // USING TEMPORARY CREDENTIALS
      .getOrCreate()
    var ds = spark.read
      .option("header", "true")
      .csv("s3a://hello-kubeflow/fare-prediction/chicago_taxi_train.csv")
      .limit(1000) // TODO: REMOVE THIS WHEN RUNNING AS A SPARKAPPLICATION
    ds = ds.withColumn("TRIP_MINUTES", ds("TRIP_SECONDS") / 60)
    ds = ds.withColumn("PARTITION", when(rand() < 0.6, lit("train")).otherwise(when(rand() < 0.5, lit("validation")).otherwise(lit("test"))))
    ds = ds.select("FARE", "TRIP_MILES", "TRIP_MINUTES", "PARTITION")
    ds.show()
    spark.stop()
  }
}
