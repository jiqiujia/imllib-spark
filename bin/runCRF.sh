$SPARK_HOME/bin/spark-submit \
    --class CRFFromParsedFileExample \
    --master spark://sr443:7077 \
    --total-executor-cores 144 \
    --executor-memory 180g \
    --driver-memory 180g \
    target/scala-2.11/imllib_2.11-0.0.1.jar \
    data/crf/template \
    hdfs://sr443/data/serialized/train.data \
    hdfs://sr443/data/serialized/test.data
