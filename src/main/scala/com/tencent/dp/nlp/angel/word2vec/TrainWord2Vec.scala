package com.tencent.dp.nlp.angel.word2vec

import algorithm.utils.IOHandler
import algorithm.utils.io.DataLoader
import com.tencent.angel.conf.AngelConf
import com.tencent.angel.ps.storage.matrix.PartitionSourceArray
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.spark.ml.feature.{Features, SubSampling}
import com.tencent.angel.spark.ml.util.SparkUtils
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.JavaConversions._

object TrainWord2Vec {
  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val input = params.getOrElse("input", "")
    val output = params.getOrElse("output", "")
    val loadPath = params.getOrElse("loadPath", "")
    val embeddingDim = params.getOrElse("embedding", "32").toInt
    val windowSize = params.getOrElse("window", "10").toInt
    val sampleRate = params.getOrElse("sampleRate", "0.001").toFloat
    val numNegSamples = params.getOrElse("negative", "5").toInt
    val numEpoch = params.getOrElse("epoch", "5").toInt
    val stepSize = params.getOrElse("stepSize", "0.1").toFloat
    val decayRate = params.getOrElse("decayRate", "0.5").toFloat
    val batchSize = params.getOrElse("batchSize", "50").toInt
    val psPartitionNum = params.getOrElse("psPartitionNum", "10").toInt
    val dataPartitionNum = params.getOrElse("dataPartitionNum", "100").toInt
    val withSubSample = params.getOrElse("subSample", "false").toBoolean
    val withRemapping = params.getOrElse("remapping", "false").toBoolean
    val storageLevel = params.getOrElse("storageLevel", "MEMORY_ONLY")
    val saveModelInterval = params.getOrElse("saveModelInterval", "2").toInt
    val checkpointInterval = params.getOrElse("checkpointInterval", "5").toInt
    val nsExponent = params.getOrElse("nsExponent", "0.75").toDouble
    val minCount = params.getOrElse("minCount", "5").toInt
    val psTimeout = params.getOrElse("psTimeout", "60000").toLong
    val seed = params.getOrElse("seed", "79").toInt

    val io = start()
    val sc = io.getSparkContext()
    val numCores = SparkUtils.getNumCores(sc.getConf)
    // The number of partition is more than the cores. We do this to achieve dynamic load balance.
    var numDataPartitions = (numCores * 3.0).toInt
    if (dataPartitionNum > numDataPartitions) {
      numDataPartitions = dataPartitionNum
    }
    println(s"dataPartitionNum=$numDataPartitions")

    val data = sc.textFile(input).map(x => x.split(" ").tail.map(_.toInt))
      .filter(x => x.length > 1)
      .persist()
    data.take(1).foreach(println)
    val wordIdx = data.flatMap(x => x)
      .map(w => (w, 1))
      .reduceByKey(_ + _)
      .filter(_._2 >= minCount)
      .map(_._1)
      .collect()
      .zipWithIndex
      .toMap
    val corpus = data.map(words => words.filter(w => wordIdx.contains(w)).map(w => wordIdx(w)))
        .filter(_.nonEmpty)
    data.unpersist(false)
    DataLoader.writeLines(wordIdx.map(x => x._1 + "\t" + x._2).toSeq,
      io.getOuputStream(output + ".mapping").get)

    //    var corpus: RDD[Array[Int]] = null
    //    var denseToString: Option[RDD[(Int, String)]] = None
    //    if (withRemapping) {
    //      val temp = Features.corpusStringToInt(data)
    //      corpus = temp._1
    //      denseToString = Some(temp._2)
    //    } else {
    //      corpus = Features.corpusStringToIntWithoutRemapping(data)
    //    }
    //Subsample will use ps, so start ps before subsample
    PSContext.getOrCreate(sc)

    val (maxWordId, docs) = if (withSubSample) {
      corpus.persist(StorageLevel.DISK_ONLY)
      val subsampleTmp = SubSampling.sampling(corpus)
      (subsampleTmp._1, subsampleTmp._2.repartition(numDataPartitions))
    } else {
      val tmp = corpus.repartition(numDataPartitions)
      (tmp.map(_.max).max().toLong + 1, tmp)
    }
    docs.persist(StorageLevel.fromString(storageLevel))
    val numDocs = docs.count()
    val numTokens = docs.map(_.length).sum().toLong
    val maxLength = docs.map(_.length).max()
    println(s"numDocs=$numDocs maxWordId=$maxWordId numTokens=$numTokens maxLength=$maxLength")

    val param = new Word2VecParam()
      .setLearningRate(stepSize)
      .setDecayRate(decayRate)
      .setEmbeddingDim(embeddingDim)
      .setBatchSize(batchSize)
      .setWindowSize(windowSize)
      .setNumPSPart(Some(psPartitionNum))
      .setSeed(seed)
      .setSampleRate(sampleRate)
      .setNumEpoch(numEpoch)
      .setNegSample(numNegSamples)
      .setMaxIndex(maxWordId)
      .setNumRowDataSet(numDocs)
      .setMaxLength(maxLength)
      .setModelCPInterval(checkpointInterval)
      .setModelSaveInterval(saveModelInterval)
      .setNSExponent(nsExponent)
    val model = new Word2VecModel(param, psTimeout)
    if (loadPath.length > 0) {
      model.load(loadPath)
    } else {
      model.randomInitialize(seed)
    }
    model.train(docs, param, output)
    model.save(output)
    //    denseToString.foreach(rdd => rdd.map(f => s"${f._1}:${f._2}").saveAsTextFile(output + "/mapping"))
    stop()
  }

  def start(): IOHandler = {
    val conf = new SparkConf()

    // Set specific parameters for Word2Vec
    conf.set(AngelConf.ANGEL_PS_PARTITION_SOURCE_CLASS, classOf[PartitionSourceArray].getName)
    // Close the automatic checkpoint
    conf.set("spark.hadoop." + AngelConf.ANGEL_PS_BACKUP_AUTO_ENABLE, "false")
    conf.set("spark.hadoop." + AngelConf.ANGEL_PS_JVM_USE_PARALLEL_GC, "true")
    conf.set("spark.hadoop." + AngelConf.ANGEL_PS_JVM_PARALLEL_GC_USE_ADAPTIVE_SIZE, "false")
    conf.set(AngelConf.ANGEL_PS_BACKUP_MATRICES, "")
    conf.set("io.file.buffer.size", "16000000")
    conf.set("spark.hadoop.io.file.buffer.size", "16000000")

    // Add jvm parameters for executors
    val executorJvmOptions = " -verbose:gc -XX:-PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:<LOG_DIR>/gc.log " +
      "-XX:+UseG1GC -XX:MaxGCPauseMillis=1000 -XX:G1HeapRegionSize=32M " +
      "-XX:InitiatingHeapOccupancyPercent=50 -XX:ConcGCThreads=4 -XX:ParallelGCThreads=4 "
    println(s"executorJvmOptions = $executorJvmOptions")
    conf.set("spark.executor.extraJavaOptions", executorJvmOptions)
    conf.setAppName("Word2Vec")

    val io = new IOHandler("", conf)
    io
  }

  def stop(): Unit = {
    PSContext.stop()
    SparkContext.getOrCreate().stop()
  }
}
