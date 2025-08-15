ThisBuild / scalaVersion := "2.13.16"

lazy val farePrediction = (project in file("."))
  .settings(
    name := "Fare Prediction",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-sql" % "4.0.0",
      "org.apache.hadoop" % "hadoop-aws" % "3.4.0",
      "org.apache.hadoop" % "hadoop-client" % "3.4.0"
    )
  )
