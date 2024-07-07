#!/bin/bash

# Update package list and install Java (required for Spark)
sudo apt-get update
sudo apt-get install -y default-jdk

# Download and install Apache Spark
SPARK_VERSION="3.1.2"
HADOOP_VERSION="3.2"
wget https://archive.apache.org/dist/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz
tar -xvf spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz
sudo mv spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION /opt/spark

# Install Miniconda for managing Python environments
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
echo "export PATH=\$HOME/miniconda/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc

# Create a conda environment and install required Python packages
conda create -n bigdata python=3.8 -y
conda activate bigdata
conda install -y pyspark torch torchvision

# Set environment variables
echo "export SPARK_HOME=/opt/spark" >> ~/.bashrc
echo "export PATH=\$PATH:/opt/spark/bin" >> ~/.bashrc
source ~/.bashrc

# Verify installation
spark-shell --version
