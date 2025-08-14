ThisBuild / scalaVersion := "2.13.16"

lazy val farePrediction = (project in file("."))
  .settings(
    name := "Fare Prediction",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-sql" % "4.0.0"
    )
  )
