import time
import subprocess
import os
import numpy as np
import torch

from pyspark.sql import SparkSession

from dataset import SharedDistRedisPool, GraddistBGPipeline
from DistribSampler import GradualDistAwareDistributedSamplerBG
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

    torch.distributed.init_process_group(backend="gloo", init_method="tcp://10.21.12.222:44144", world_size=1, rank=0)

    dataset = SharedDistRedisPool()
    data_sampler = GradualDistAwareDistributedSamplerBG(
        dataset=dataset, num_replicas=1, batch_size=2, rank=2, ip_mover="127.0.0.1", port_mover=50524)
    data_sampler.set_rank(rank=2)
    
    # create the pipeline from sampler
    dataset_pipeline = GraddistBGPipeline(dataset=dataset, batch_size=2,
                                       sampler=data_sampler, num_replicas=1)

    rdd_creation_timelist = []
    start_time = time.time()
    try:
        # Generate a large dataset
        # number of slices increased to avoid data serilization size <2G constraint for 30G data
        for i in range(20):
            rdd_start_time = time.time()
            large_dataset = spark.sparkContext.parallelize(dataset_pipeline, numSlices=32) #generate_large_dataset(spark)
            large_dataset.persist()
            rdd_creation_timelist.append(time.time() - rdd_start_time)
        print("RDD size: {0}".format(large_dataset.count()))

        # Perform some transformations to put pressure on memory
            # processed_data = (
            #     large_dataset
#            .map(lambda x: x[0]/255.0)  # Transform data
#            .groupByKey()  # Introduce shuffling operation
#            .mapValues(lambda values: sum(values))  # Another transformation
            # )
            # large_dataset.unpersist()

        print("rdd creation timelist {0}".format(rdd_creation_timelist))
    finally:
        # Stop the Spark session
        spark.stop()

if __name__ == "__main__":
    main()
