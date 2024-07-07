from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

def main():
    spark = SparkSession.builder.appName("DataIngestionPreprocessing").getOrCreate()

    schema = StructType([
        StructField("id", StringType(), True),
        StructField("text", StringType(), True),
        StructField("label", IntegerType(), True)
    ])

    data = spark.read.csv("hdfs://path_to_large_dataset.csv", schema=schema, header=True)

    # Preprocess the data
    processed_data = data.withColumn("label", when(col("label") == "positive", 1).otherwise(0))

    processed_data.show(5)

    processed_data.write.parquet("hdfs://path_to_processed_data.parquet")

    spark.stop()

if __name__ == "__main__":
    main()
