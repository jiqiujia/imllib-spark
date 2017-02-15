/opt/spark-1.6.0-bin-hadoop2.6/bin/spark-submit \
    --class FMExample \
    --master local \
    target/scala-2.11/imllib_2.11-0.0.1.jar \
    data/fm/a9a \
    data/fm/a9a.t \
    144 \
    30 \
    0.01 \
    4
