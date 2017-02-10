export HADOOP_CONF_DIR=/opt/hadoop-2.6.5/etc/hadoop/conf/
/opt/spark-1.6.0-bin-hadoop2.6/bin/spark-submit \
    --class FMExample \
    --master yarn-cluster \
    target/scala-2.11/imllib_2.11-0.0.1.jar \
    hdfs://sr443/data/a9a \
    hdfs://sr443/data/a9a.t \
    4 \
    30 \
    0.01 \
    4
