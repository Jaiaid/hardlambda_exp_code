
# Import SparkSession
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext

import os
os.environ["SPARK_HOME"] = r"/opt/spark/spark-3.5.0-bin-hadoop3"
#os.environ["PYSPARK_SUBMIT_ARGS"] = "--master spark://10.21.12.222:19489 --total-executor 2 pyspark-shell"
#os.environ["JAVA_HOME"] = r"/usr/bin"

# Create SparkSession
#spark = SparkSession.builder \
#      .master("spark://10.21.12.222:19489") \
#      .appName("SparkByExamples.com") \
#      .getOrCreate()
master_url = "spark://10.21.22.222:19489"  # Replace with the actual master IP and port

# Create a Spark configuration and context
conf = SparkConf().setAppName("MySparkApplication").setMaster(master_url)
spark = SparkContext(conf=conf)


# Create RDD from parallelize
dataList = [("Java", 20000), ("Python", 100000), ("Scala", 3000)]
rdd=spark.sparkContext.parallelize(dataList)

data = [('James','','Smith','1991-04-01','M',3000),
  ('Michael','Rose','','2000-05-19','M',4000),
  ('Robert','','Williams','1978-09-05','M',4000),
  ('Maria','Anne','Jones','1967-12-01','F',4000),
  ('Jen','Mary','Brown','1980-02-17','F',-1)
]

columns = ["firstname","middlename","lastname","dob","gender","salary"]
df = spark.createDataFrame(data=data, schema = columns)

df.createOrReplaceTempView("PERSON_DATA")
for i in range(1000000):
    df2 = spark.sql("SELECT * from PERSON_DATA")
    df2.collect()
df2.printSchema()
df2.show()
