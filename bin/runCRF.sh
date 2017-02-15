$SPARK_HOME/bin/spark-submit \
    --class CRFFromParsedFileExample \
    --master local \
    target/scala-2.11/imllib_2.11-0.0.1.jar \
    data/crf/template \
    hdfs://localhost:9000/data/serialized/train.data \
    hdfs://localhost:9000/data/serialized/test.data
