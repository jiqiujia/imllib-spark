$SPARK_HOME/bin/spark-submit \
    --class FFMExample \
    --master spark://sr443:7077 \
    --total-executor-cores 144 \
    --executor-memory 180g \
    --driver-memory 180g \
    target/scala-2.11/imllib_2.11-0.0.1.jar \
    hdfs://sr443:8020/data/a9a_ffm \
    8 \
    10 \
    0.01 \
    0.00002 \
    false \
    false \
    144
