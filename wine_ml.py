from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import happybase

# Create Spark Session with Hive Support
spark = SparkSession.builder \
    .appName("WineClassificationML") \
    .enableHiveSupport() \
    .getOrCreate()

# Load data from Hive table
wine_df = spark.sql("""
SELECT
    class,
    alcohol,
    malic_acid,
    ash,
    alcalinity_of_ash,
    magnesium,
    total_phenols,
    flavanoids,
    nonflavanoid_phenols,
    proanthocyanins,
    color_intensity,
    hue,
    od280_od315_of_diluted_wines,
    proline
FROM wine_classification
""")

# Remove null rows
wine_df = wine_df.na.drop()

# Convert class labels from 1,2,3 to 0,1,2
wine_df = wine_df.withColumn("label", wine_df["class"] - 1)

# Assemble features
assembler = VectorAssembler(
    inputCols=[
        "alcohol",
        "malic_acid",
        "ash",
        "alcalinity_of_ash",
        "magnesium",
        "total_phenols",
        "flavanoids",
        "nonflavanoid_phenols",
        "proanthocyanins",
        "color_intensity",
        "hue",
        "od280_od315_of_diluted_wines",
        "proline"
    ],
    outputCol="features",
    handleInvalid="skip"
)

assembled_df = assembler.transform(wine_df).select("features", "label")

# Split into train and test data
train_data, test_data = assembled_df.randomSplit([0.7, 0.3], seed=42)

# Logistic Regression Model
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=20
)

lr_model = lr.fit(train_data)

# Generate predictions
predictions = lr_model.transform(test_data)

# Accuracy
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

# F1 Score
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

accuracy = accuracy_evaluator.evaluate(predictions)
f1_score = f1_evaluator.evaluate(predictions)

print("===================================")
print("WINE MODEL RESULTS")
print("===================================")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1_score}")
print("===================================")

# Write metrics to HBase
metrics = [
    ("wine_run", "cf:accuracy", str(accuracy)),
    ("wine_run", "cf:f1_score", str(f1_score))
]

def write_to_hbase_partition(partition):
    connection = happybase.Connection("master")
    connection.open()

    table = connection.table("wine_ml_metrics")

    for row in partition:
        row_key, column, value = row
        table.put(row_key.encode(), {column.encode(): value.encode()})

    connection.close()

rdd = spark.sparkContext.parallelize(metrics)
rdd.foreachPartition(write_to_hbase_partition)

spark.stop()