package org.example;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import java.io.IOException;

public class WineQualityTraining {

    public static void main(String[] args) throws IOException {

        SparkSession spark = SparkSession.builder()
                .appName("WineQualityLocalTraining")
              .master("local[*]")
                // .master("spark://ip-172-31-29-251:7077")
              //  .master("spark://ip-172-31-29-251.ec2.internal:7077")
                .getOrCreate();

        Dataset<Row> rawData = spark.read()
                .option("header", "true")
                .option("delimiter", ";")
                .option("inferSchema", "true")
                  .csv("src/main/resources/TrainingDataset.csv");
                //.csv("/home/ubuntu/TrainingDataset.csv");

        String[] originalCols = rawData.columns();
        for (String col : originalCols) {
            String newCol = col.replaceAll("\"", "");
            rawData = rawData.withColumnRenamed(col, newCol);
        }

        String[] inputCols = {
                "fixed acidity", "volatile acidity", "citric acid",
                "residual sugar", "chlorides", "free sulfur dioxide",
                "total sulfur dioxide", "density", "pH",
                "sulphates", "alcohol"
        };

        for (String col : inputCols) {
            rawData = rawData.withColumn(col, rawData.col(col).cast(DataTypes.DoubleType));
        }

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(inputCols)
                .setOutputCol("features");

        Dataset<Row> transformedData = assembler.transform(rawData);

        LogisticRegression logisticRegression = new LogisticRegression()
                .setLabelCol("quality")
                .setFeaturesCol("features");

        LogisticRegressionModel rfModel = logisticRegression.fit(transformedData);

          rfModel.save("src/main/resources/TrainedLogisticRegressionFinal");
      //  rfModel.save("/home/ubuntu/TrainedLogisticRegressionFinal");

        spark.stop();
    }
}