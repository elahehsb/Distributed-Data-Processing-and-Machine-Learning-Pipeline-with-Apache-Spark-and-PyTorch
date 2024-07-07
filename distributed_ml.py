import torch
import torch.nn as nn
import torch.optim as optim
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression

# Define the PyTorch model
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(data):
    # Convert Spark DataFrame to PyTorch Tensor
    features = torch.tensor(data.select("features").rdd.map(lambda row: row[0].toArray()).collect())
    labels = torch.tensor(data.select("label").rdd.map(lambda row: row[0]).collect())

    model = SimpleNN(features.size(1))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

def main():
    spark = SparkSession.builder.appName("DistributedML").getOrCreate()

    data = spark.read.parquet("hdfs://path_to_processed_data.parquet")

    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf])

    model = pipeline.fit(data)
    transformed_data = model.transform(data)

    train_model(transformed_data.select("features", "label"))

    spark.stop()

if __name__ == "__main__":
    main()
