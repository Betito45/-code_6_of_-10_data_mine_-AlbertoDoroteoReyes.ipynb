#Ai Transcript:https://share.gemini.google/NoqB4MZiWXv0

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SpookyAuthorship").getOrCreate()

df = spark.read.csv("train.csv", header=True, inferSchema=True)

print("row count")
df.printSchema()
print(f"Total Rows: {df.count()}")


df.groupBy("author").count().show()

from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, Normalizer, StringIndexer
from pyspark.ml import Pipeline

#text preprocessing stages
tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
# Index the target labels
label_indexer = StringIndexer(inputCol="author", outputCol="label")
# Generate TF and IDF matrices
vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="vectorized_tokens")
idf = IDF(inputCol="vectorized_tokens", outputCol="tfidf")
#give all text sequences an equal weight/scale
normalizer = Normalizer(inputCol="tfidf", outputCol="normalized_features")

pipeline = Pipeline(stages=[label_indexer, tokenizer, stopwords_remover, vectorizer, idf, normalizer])

processed_data = pipeline.fit(df).transform(df)

print("Processed features")
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


print("Confusion Matrix")
predictions.groupBy("label", "prediction").count().show()


import matplotlib.pyplot as plt
import seaborn as sns


cm_pyspark = predictions.groupBy("label", "prediction").count().toPandas()

cm_pivot = cm_pyspark.pivot(index='label', columns='prediction', values='count').fillna(0)


plt.figure(figsize=(6, 4))
sns.heatmap(cm_pivot, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.show()






#Setting up the pipeline using PySpark's ML library was different from Scikit-learn, but chaining the Tokenizer, StopWordsRemover, and TF-IDF tools into one Pipeline made the data preprocessing very efficient. The Bayes model got a good accuracy for classifying the three authors, though the  matrix shows there is still some overlap. If I were to improve this next, I would test a Logistic Regression model or implement a UDF to see if it reduces in the text features.


