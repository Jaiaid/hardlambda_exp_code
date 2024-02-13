import time
import subprocess
import os
from pyspark.sql import SparkSession

from dataset import SharedDistRedisPool, DatasetPipeline
from DistribSampler import DefaultDistributedSampler, GradualDistAwareDistributedSamplerBG
from shade_modified import ShadeDataset, ShadeSampler
from DataMovementService import DataMoverServiceInterfaceClient

def create_spark_session():
    return SparkSession.builder.appName("MemoryPressureJob").getOrCreate()

def generate_large_dataset(spark, num_partitions=4, partition_size=10**6):
    # Generate a large dataset with specified number of partitions and partition size
    data = [i for i in range(num_partitions * partition_size)]
    rdd = spark.sparkContext.parallelize(data, numSlices=num_partitions)
    return rdd.map(lambda x: (x % num_partitions, x))

def main():
    # Create a Spark session
    spark = create_spark_session()

    rank = 0
    dataset = SharedDistRedisPool()
    data_sampler = DefaultDistributedSampler(
        dataset=dataset, num_replicas=2, batch_size=16)

    # create the pipeline from sampler
    dataset_pipeline = DatasetPipeline(dataset=dataset, batch_size=16,
                                       sampler=data_sampler, num_replicas=2)

    try:
        # Generate a large dataset
        # number of slices increased to avoid data serilization size <2G constraint for 30G data
        large_dataset = spark.sparkContext.parallelize(dataset_pipeline, numSlices=16)#generate_large_dataset(spark)

        # Perform some transformations to put pressure on memory
        processed_data = (
            large_dataset
            .map(lambda x: x)  # Transform data
            .groupByKey()  # Introduce shuffling operation
            .mapValues(lambda values: sum(values))  # Another transformation
        )

        # Trigger an action to materialize the results and put pressure on memory
        result = processed_data.collect()

        # Print the first few results
        # print(result[:10])

    finally:
        # Stop the Spark session
        spark.stop()

if __name__ == "__main__":
    main()

