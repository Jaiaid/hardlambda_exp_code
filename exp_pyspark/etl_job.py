import time
import subprocess
import os
from pyspark.sql import SparkSession

from dataset import SharedRedisPool, SharedDistRedisPool, DatasetPipeline
from DistribSampler import DistAwareDistributedSampler, DefaultDistributedSampler, GradualDistAwareDistributedSampler, GradualDistAwareDistributedSamplerBG
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
    data_sampler = GradualDistAwareDistributedSamplerBG(
        dataset=dataset, num_replicas=2, batch_size=16)
    data_sampler.set_rank(rank=0)
    # starting the background data mover service
    data_mover_service = subprocess.Popen(
        """python3 {2}/DataMovementService.py --seqno {0}
        -bs 16 -cn 10.21.12.239 26379 10.21.12.239 26380 10.21.12.222 26379 -pn 10.21.12.239 10.21.12.222 -p {1}""".format(
            rank if rank < 3 else 2, 50524, os.path.dirname(os.path.abspath(__file__))).split()
    )
    # check if running
    if data_mover_service.poll() is None:
        print("data mover service is running")
    # try 10 times to connect
    connection_refused_count = 0
    while connection_refused_count < 10: 
        try:
            data_mover = DataMoverServiceInterfaceClient("127.0.0.1", 50524)
            break
        except ConnectionError as e:
            connection_refused_count += 1
            print("connection establish attempt {0} failed".format(connection_refused_count))
            # sleep for a second
            time.sleep(1)

    if connection_refused_count == 10:
        print("connection failed, exiting...")
        data_mover_service.kill()
        exit(1)
    else:
        print("connection successful after {0} attempt".format(connection_refused_count))
        print("data movement service client interface is opened")

    # create the pipeline from sampler
    dataset_pipeline = DatasetPipeline(dataset=dataset, batch_size=16,
                                       sampler=data_sampler, num_replicas=2)

    try:
        # Generate a large dataset
        large_dataset = spark.sparkContext.parallelize(dataset_pipeline, numSlices=2)#generate_large_dataset(spark)

        # Perform some transformations to put pressure on memory
        processed_data = (
            large_dataset
            .map(lambda x: x/255.0)  # Transform data
            .groupByKey()  # Introduce shuffling operation
            .mapValues(lambda values: sum(values))  # Another transformation
        )

        # Trigger an action to materialize the results and put pressure on memory
        result = processed_data.collect()

        # Print the first few results
        print(result[:10])

    finally:
        # Stop the Spark session
        spark.stop()

if __name__ == "__main__":
    main()

