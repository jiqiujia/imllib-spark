/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */


package com.tencent.dp.nlp.angel.word2vec

import java.util.concurrent.TimeUnit

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import com.tencent.angel.ml.core.optimizer.decayer.{StandardDecay, StepSizeScheduler}
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.model.{MatrixLoadContext, MatrixSaveContext, ModelLoadContext, ModelSaveContext}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.graph.algo.embedding.FastSigmoid
import com.tencent.angel.spark.ml.graph.algo.embedding.line2._
import com.tencent.angel.spark.ml.util.LogUtils
import com.tencent.angel.spark.models.PSMatrix
import it.unimi.dsi.fastutil.ints.{Int2IntOpenHashMap, Int2ObjectOpenHashMap}
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random


/**
  * 参考angel的原始实现做了以下修改：
  * 1. 修改negative sampling，使用原始论文的实现方式
  * 2. 修改学习率的下降方式
  */

/**
  * Entry in vocabulary
  */
private case class VocabWord(var word: Int, var cn: Int)

class Word2VecModel(numNode: Int,
                    embeddingDim: Int,
                    numPart: Int,
                    maxLength: Int,
                    numNodesPerRow: Int,
                    seed: Int = Random.nextInt,
                    learningRate: Float,
                    decayRate: Float,
                    params: Word2VecParam,
                    psTimeout: Long) extends Serializable {

  val matrixName = "embedding"
  // Create one ps matrix to hold the input vectors and the output vectors for all node
  val mc: MatrixContext = new MatrixContext(matrixName, 1, numNode)
  mc.setMaxRowNumInBlock(1)
  mc.setMaxColNumInBlock(numNode / numPart)
  mc.setRowType(RowType.T_ANY_INTKEY_DENSE)
  mc.setValueType(classOf[LINENode])
  mc.setInitFunc(new LINEInitFunc(2, embeddingDim))
  val psMatrix: PSMatrix = PSMatrix.matrix(mc)
  val matrixId: Int = psMatrix.id
  val ssScheduler: StepSizeScheduler = new StandardDecay(learningRate, decayRate)
  var totalPullTime: Long = 0
  var totalPushTime: Long = 0
  var totalMakeParamTime: Long = 0
  var totalCalTime: Long = 0
  var totalMakeGradTime: Long = 0
  var totalCallNum: Long = 0
  var totalWaitPullTime: Long = 0

  private var trainWordsCount = 0L
  private var vocabSize = 0
  private var vocab: Array[VocabWord] = _
  private var vocabHash = mutable.HashMap.empty[Int, Int]
  private var table: Array[Int] = _

  // initialize embeddings
  def randomInitialize(seed: Int): Unit = {
    val beforeRandomize = System.currentTimeMillis()
    psMatrix.psfUpdate(new LINEModelRandomize(new RandomizeUpdateParam(matrixId, embeddingDim, 2, seed))).get()
    LogUtils.logTime(s"Model successfully Randomized, cost ${(System.currentTimeMillis() - beforeRandomize) / 1000.0}s")
  }

  def this(params: Word2VecParam, psTimeout: Long) {
    this(params.maxIndex, params.embeddingDim, params.numPSPart, params.maxLength, params.nodesNumPerRow, params.seed,
      params.learningRate, params.decayRate, params, psTimeout)
  }

  private def learnVocab(dataset: RDD[Array[Int]], params: Word2VecParam): Unit = {
    vocab = dataset.flatMap(x => x)
      .map(w => (w, 1))
      .reduceByKey(_ + _)
//      .filter(_._2 >= params.minCount)
      .map(x => VocabWord(
        x._1,
        x._2))
      .collect()
      .sortWith((a, b) => a.cn > b.cn)

    vocabSize = vocab.length
    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check " +
      "the setting of minCount, which could be large enough to remove all your words in sentences.")

    var a = 0
    while (a < vocabSize) {
      vocabHash += vocab(a).word -> a
      trainWordsCount += vocab(a).cn
      a += 1
    }
    println(s"vocabSize = $vocabSize, trainWordsCount = $trainWordsCount")
  }

  private def initUnigramTable(params: Word2VecParam): Unit = {
    var a = 0
    val power = params.ns_exponent
    var trainWordsPow = 0.0
    table = new Array[Int](params.tableSize)

    while (a < vocabSize) {
      trainWordsPow += Math.pow(vocab(a).cn, power)
      a += 1
    }

    var i = 0
    var d1 = Math.pow(vocab(i).cn, power) / trainWordsPow
    a = 0
    while (a < params.tableSize) {
      table(a) = i
      if (a.toDouble / params.tableSize > d1) {
        i += 1
        d1 += Math.pow(vocab(i).cn, power) / trainWordsPow
      }
      if (i >= vocabSize) {
        i = vocabSize - 1
      }
      a += 1
    }
  }

  def train(corpus: RDD[Array[Int]], params: Word2VecParam, path: String): Unit = {
    learnVocab(corpus, params)
    initUnigramTable(params)
//    randomInitialize(seed)

    var learningRate = params.learningRate
    for (epoch <- 1 to params.numEpoch) {
      val epochStartTime = System.currentTimeMillis()
      val (lossSum, size) = corpus.mapPartitions {
        iterator =>
          iterator.sliding(params.batchSize, params.batchSize)
            .map(batch => {
              (optimize(batch.toArray, params.windowSize, params.negSample, params.maxIndex,
                learningRate, params.sampleRate), batch.length)
            })
      }.reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))
      learningRate = learningRate * params.decayRate //ssScheduler.next().toFloat
      val epochTime = System.currentTimeMillis() - epochStartTime
      println(s"epoch=$epoch loss=${lossSum / size} time=${epochTime.toFloat / 1000}s")
      checkpointAndSaveIfNeed(epoch, params, path)
    }
  }

  /**
    * Write checkpoint or model result if need
    */
  def checkpointAndSaveIfNeed(epoch: Int, params: Word2VecParam, path: String): Unit = {
    var startTs = 0L
    if (epoch % params.checkpointInterval == 0 && epoch < params.numEpoch) {
      LogUtils.logTime(s"Epoch=$epoch, checkpoint the model")
      startTs = System.currentTimeMillis()
      psMatrix.checkpoint(epoch)
      LogUtils.logTime(s"checkpoint use time=${System.currentTimeMillis() - startTs}")
    }

    if (epoch % params.saveModelInterval == 0 && epoch < params.numEpoch) {
      LogUtils.logTime(s"Epoch=$epoch, save the model")
      startTs = System.currentTimeMillis()
      save(path, epoch)
      LogUtils.logTime(s"save use time=${System.currentTimeMillis() - startTs}")
    }
  }

  def save(modelPathRoot: String, epoch: Int): Unit = {
    save(new Path(modelPathRoot, s"CP_$epoch").toString)
  }

  def save(modelPath: String): Unit = {
    LogUtils.logTime(s"saving model to $modelPath")
    val ss = SparkSession.builder().getOrCreate()
    deleteIfExists(modelPath, ss)

    val saveContext = new ModelSaveContext(modelPath)
    saveContext.addMatrix(new MatrixSaveContext(matrixName, classOf[TextLINEModelOutputFormat].getTypeName))
    PSContext.instance().save(saveContext)
  }

  private def deleteIfExists(modelPath: String, ss: SparkSession): Unit = {
    val path = new Path(modelPath)
    val fs = path.getFileSystem(ss.sparkContext.hadoopConfiguration)
    if (fs.exists(path)) {
      fs.delete(path, true)
    }
  }

  def optimize(batch: Array[Array[Int]], windowSize: Int, numNegSample: Int,
               maxIndex: Int, learningRate: Float, sampleRate: Float): Double = {
    val trainBatch = parseBatchData(batch, windowSize, numNegSample, maxIndex, sampleRate)
    if (trainBatch._1.isEmpty || trainBatch._2.isEmpty || trainBatch._3.isEmpty) {
      println(f"Warning: empty src/dst/negtives in trainBatch, ${trainBatch._1.length}, " +
        f"${trainBatch._2.length}, ${trainBatch._3.length}")
      0.0
    } else {
      optimizeOneBatch(batch.length, trainBatch._1, trainBatch._2, trainBatch._3, numNegSample, learningRate)
    }
  }

  def optimizeOneBatch(batchSize: Int, srcNodes: Array[Int], dstNodes: Array[Int], negativeSamples: Array[Array[Int]],
                       numNegSample: Int, learningRate: Float): Double = {
    incCallNum()
    var start = 0L
    start = System.currentTimeMillis()
    // 目前版本，如果拉取的向量为空，这里会超时
    val result = try {
      psMatrix.asyncPsfGet(new LINEGetEmbedding(new LINEGetEmbeddingParam(matrixId, srcNodes, dstNodes,
        negativeSamples, 2, numNegSample))).get(psTimeout, TimeUnit.MILLISECONDS).asInstanceOf[LINEGetEmbeddingResult].getResult
    } catch {
      case e: Exception => {
        e.printStackTrace()
        LogUtils.logTime("distinct id number in a batch: " + (srcNodes.toSet ++ dstNodes.toSet ++ negativeSamples.toSet).size)
        null
      }
    }
    val srcFeats: Int2ObjectOpenHashMap[Array[Float]] = result._1
    val dstFeats: Int2ObjectOpenHashMap[Array[Float]] = result._2
    incPullTime(start)

    // Calculate the gradients
    start = System.currentTimeMillis()
    val dots = dot(srcNodes, dstNodes, negativeSamples, srcFeats, dstFeats, numNegSample)
    var loss = doGrad(dots, numNegSample, learningRate)
    incCalTime(start)
    start = System.currentTimeMillis()
    val (inputUpdates, outputUpdates) = adjust(srcNodes, dstNodes, negativeSamples, srcFeats, dstFeats, numNegSample, dots)
    incCalUpdateTime(start)
    // Push the gradient to ps
    start = System.currentTimeMillis()
    psMatrix.psfUpdate(new LINEAdjust(new LINEAdjustParam(matrixId, inputUpdates, outputUpdates, 2)))
    incPushTime(start)

    loss = loss / dots.length.toLong
    println(s"avgPullTime=$avgPullTime avgGradTime=$avgCalTime " +
      s"avgCalUpdateTime=$avgCalUpdateTime avgPushTime=$avgPushTime loss=$loss")

    loss * batchSize
  }

  def dot(srcNodes: Array[Int], destNodes: Array[Int], negativeSamples: Array[Array[Int]],
          srcFeats: Int2ObjectOpenHashMap[Array[Float]], targetFeats: Int2ObjectOpenHashMap[Array[Float]], negative: Int): Array[Float] = {
    val dots: Array[Float] = new Array[Float]((1 + negative) * srcNodes.length)
    var docIndex = 0
    for (i <- srcNodes.indices) {
      val srcVec = srcFeats.get(srcNodes(i))
      // Get dot value for (src, dst)
      dots(docIndex) = arraysDot(srcVec, targetFeats.get(destNodes(i)))
      docIndex += 1

      // Get dot value for (src, negative sample)
      for (j <- 0 until negative) {
        dots(docIndex) = arraysDot(srcVec, targetFeats.get(negativeSamples(i)(j)))
        docIndex += 1
      }
    }
    dots
  }

  def arraysDot(x: Array[Float], y: Array[Float]): Float = {
    var dotValue = 0.0f
    x.indices.foreach(i => dotValue += x(i) * y(i))
    dotValue
  }

  def doGrad(dots: Array[Float], negative: Int, alpha: Float): Float = {
    var loss = 0.0f
    for (i <- dots.indices) {
      val prob = FastSigmoid.sigmoid(dots(i))
      if (i % (negative + 1) == 0) {
        dots(i) = alpha * (1 - prob)
        loss -= FastSigmoid.log(prob)
      } else {
        dots(i) = -alpha * FastSigmoid.sigmoid(dots(i))
        loss -= FastSigmoid.log(1 - prob)
      }
    }
    loss
  }

  def adjust(srcNodes: Array[Int], destNodes: Array[Int], negativeSamples: Array[Array[Int]],
             srcFeats: Int2ObjectOpenHashMap[Array[Float]], targetFeats: Int2ObjectOpenHashMap[Array[Float]],
             negative: Int, dots: Array[Float]): (Int2ObjectOpenHashMap[Array[Float]], Int2ObjectOpenHashMap[Array[Float]]) = {
    val inputUpdateCounter = new Int2IntOpenHashMap(srcFeats.size())
    val inputUpdates = new Int2ObjectOpenHashMap[Array[Float]](srcFeats.size())

    val outputUpdateCounter = new Int2IntOpenHashMap(targetFeats.size())
    val outputUpdates = new Int2ObjectOpenHashMap[Array[Float]](targetFeats.size())

    var docIndex = 0
    for (i <- srcNodes.indices) {
      // Src node grad
      val neule = new Array[Float](embeddingDim)

      // Accumulate dst node embedding to neule
      val dstEmbedding = targetFeats.get(destNodes(i))
      var g = dots(docIndex)
      axpy(neule, dstEmbedding, g)

      // Use src node embedding to update dst node embedding
      val srcEmbedding = srcFeats.get(srcNodes(i))
      merge(outputUpdateCounter, outputUpdates, destNodes(i), g, srcEmbedding)

      docIndex += 1

      // Use src node embedding to update negative sample node embedding; Accumulate negative sample node embedding to neule
      for (j <- 0 until negative) {
        val negSampleEmbedding = targetFeats.get(negativeSamples(i)(j))
        g = dots(docIndex)

        // Accumulate negative sample node embedding to neule
        axpy(neule, negSampleEmbedding, g)

        // Use src node embedding to update negative sample node embedding
        merge(outputUpdateCounter, outputUpdates, negativeSamples(i)(j), g, srcEmbedding)
        docIndex += 1
      }

      // Use accumulation to update src node embedding, grad = 1
      merge(inputUpdateCounter, inputUpdates, srcNodes(i), 1, neule)
    }

    var iter = inputUpdateCounter.int2IntEntrySet().fastIterator()
    while (iter.hasNext) {
      val entry = iter.next()
      div(inputUpdates.get(entry.getIntKey), entry.getIntValue.toFloat)
    }

    iter = outputUpdateCounter.int2IntEntrySet().fastIterator()
    while (iter.hasNext) {
      val entry = iter.next()
      div(outputUpdates.get(entry.getIntKey), entry.getIntValue.toFloat)
    }

    (inputUpdates, outputUpdates)
  }

  def merge(inputUpdateCounter: Int2IntOpenHashMap, inputUpdates: Int2ObjectOpenHashMap[Array[Float]],
            nodeId: Int, g: Float, update: Array[Float]): Int = {
    var grads: Array[Float] = inputUpdates.get(nodeId)
    if (grads == null) {
      grads = new Array[Float](embeddingDim)
      inputUpdates.put(nodeId, grads)
      inputUpdateCounter.put(nodeId, 0)
    }
    axpy(grads, update, g)
    inputUpdateCounter.addTo(nodeId, 1)
  }

  def axpy(y: Array[Float], x: Array[Float], a: Float): Unit = {
    blas.saxpy(y.length, a, x, 0, 1, y, 0, 1)
    //x.indices.foreach(i => y(i) += a * x(i))
  }

  def div(x: Array[Float], f: Float): Unit = {
    x.indices.foreach(i => x(i) = x(i) / f)
  }

  /* time calculate functions */
  def incPullTime(startTs: Long): Unit = {
    totalPullTime = totalPullTime + (System.currentTimeMillis() - startTs)
  }

  def incPushTime(startTs: Long): Unit = {
    totalPushTime = totalPushTime + (System.currentTimeMillis() - startTs)
  }

  def incCalTime(startTs: Long): Unit = {
    totalCalTime = totalCalTime + (System.currentTimeMillis() - startTs)
  }

  def incCalUpdateTime(startTs: Long): Unit = {
    totalMakeGradTime = totalMakeGradTime + (System.currentTimeMillis() - startTs)
  }

  def incCallNum(): Unit = {
    totalCallNum = totalCallNum + 1
  }

  def avgPullTime: Long = {
    totalPullTime / totalCallNum
  }

  def avgPushTime: Long = {
    totalPushTime / totalCallNum
  }

  def avgCalUpdateTime: Long = {
    totalMakeGradTime / totalCallNum
  }

  def avgCalTime: Long = {
    totalCalTime / totalCallNum
  }

  def load(modelPath: String): Unit = {
    val startTime = System.currentTimeMillis()
    LogUtils.logTime(s"load model from $modelPath")

    val loadContext = new ModelLoadContext(modelPath)
    loadContext.addMatrix(new MatrixLoadContext(psMatrix.name))
    PSContext.getOrCreate(SparkContext.getOrCreate()).load(loadContext)
    LogUtils.logTime(s"model load time=${System.currentTimeMillis() - startTime} ms")
  }

  def merge(updateCounter: Int2IntOpenHashMap, updates: Int2ObjectOpenHashMap[Array[Float]],
            nodeId: Int, update: Array[Float], learningRate: Float): Int = {
    var grads: Array[Float] = updates.get(nodeId)
    if (grads == null) {
      grads = new Array[Float](embeddingDim)
      updates.put(nodeId, grads)
      updateCounter.put(nodeId, 0)
    }
    update.indices.foreach(i => grads(i) += -1.0f * learningRate * update(i))
    updateCounter.addTo(nodeId, 1)
  }

  def incMakeParamTime(startTs: Long): Unit = {
    totalMakeParamTime = totalMakeParamTime + (System.currentTimeMillis() - startTs)
  }

  def avgMakeParamTime: Long = {
    totalMakeParamTime / totalCallNum
  }


  def parseBatchData(sentences: Array[Array[Int]], windowSize: Int, negative: Int, maxIndex: Int,
                     sampleRate: Float, seed: Int = Random.nextInt): (Array[Int], Array[Int], Array[Array[Int]]) = {
    val rand = new Random(seed)
    val srcNodes = new ArrayBuffer[Int]()
    val dstNodes = new ArrayBuffer[Int]()
    for (s <- sentences.indices) {
      val sen = if (sampleRate > 0) {
        sentences(s).map(word => {
          val ran = Math.sqrt(vocab(word).cn / (sampleRate * trainWordsCount) + 1) *
            (sampleRate * trainWordsCount) / vocab(word).cn
          if (ran >= rand.nextFloat()) {
            word
          } else {
            -1
          }
        }).filter(w => w >= 0)
      } else {
        sentences(s)
      }
      for (srcIndex <- sen.indices) {
        var dstIndex = Math.max(srcIndex - windowSize, 0)
        while (dstIndex < Math.min(srcIndex + windowSize + 1, sen.length)) {
          if (srcIndex != dstIndex) {
            srcNodes.append(sen(dstIndex))
            dstNodes.append(sen(srcIndex))
          }
          dstIndex += 1
        }
      }
    }
    val negativeSamples = negativeSample(srcNodes.toArray, dstNodes.toArray, negative, maxIndex, rand.nextInt())
    (srcNodes.toArray, dstNodes.toArray, negativeSamples)
  }

  def negativeSample(srcNodes: Array[Int], dstNodes: Array[Int], sampleNum: Int,
                     maxIndex: Int, seed: Int): Array[Array[Int]] = {
    val rand = new Random(seed)
    val sampleWords = new Array[Array[Int]](srcNodes.length)
    var wordIndex: Int = 0

    for (i <- srcNodes.indices) {
      var sampleIndex: Int = 0
      sampleWords(wordIndex) = new Array[Int](sampleNum)
      while (sampleIndex < sampleNum) {
        val target = table(rand.nextInt(params.tableSize))
        // val target = rand.nextInt(maxIndex)
        if (target != srcNodes(i) && target != dstNodes(i)) {
          sampleWords(wordIndex)(sampleIndex) = target
          sampleIndex += 1
        }
      }
      wordIndex += 1
    }
    sampleWords
  }
}