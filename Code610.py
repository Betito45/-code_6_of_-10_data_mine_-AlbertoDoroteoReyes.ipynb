#Ai Transcript:https://share.gemini.google/NoqB4MZiWXv0

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("SpookyAuthorship").getOrCreate()
df = spark.read.csv("train.csv", header=True, inferSchema=True)
print("Data structure and row count:")
df.printSchema()
print(f"Total Rows: {df.count()}")
df.groupBy("author").count().show()

from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, Normalizer, StringIndexer
from pyspark.ml import Pipeline

tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
label_indexer = StringIndexer(inputCol="author", outputCol="label")
vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="vectorized_tokens")
idf = IDF(inputCol="vectorized_tokens", outputCol="tfidf")
normalizer = Normalizer(inputCol="tfidf", outputCol="normalized_features")

pipeline = Pipeline(stages=[label_indexer, tokenizer, stopwords_remover, vectorizer, idf, normalizer])
processed_data = pipeline.fit(df).transform(df)
print("Processed features preview:")
processed_data.select("author", "label", "normalized_features").show(5)

from pyspark.ml.classification import NaiveBayes
train_data, test_data = processed_data.randomSplit([0.8, 0.2], seed=42)
nb = NaiveBayes(featuresCol="normalized_features", labelCol="label", predictionCol="prediction")
nb_model = nb.fit(train_data)

predictions = nb_model.transform(test_data)
print("Prediction preview:")
predictions.select("id", "label", "prediction").show(5)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Accuracy: {round(accuracy, 4)}")
print("Confusion Matrix:")
predictions.groupBy("label", "prediction").count().show()

