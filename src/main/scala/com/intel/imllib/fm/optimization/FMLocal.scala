package com.intel.imllib.fm.optimization

import java.io._
import breeze.linalg.{*, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, Vector => BV}
import breeze.stats.distributions.Rand
import com.intel.imllib.metric.AUC
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.mutable

// 目前只支持分类
class FMLocal(var factorMatrix: BDM[Double],
              var weightVector: Option[BV[Double]],
              var intercept: Double) extends Serializable {

  val numFeatures = factorMatrix.cols
  val numFactors = factorMatrix.rows

  require(numFeatures > 0 && numFactors > 0)

  def predictLogit(testData: Vector): (Double, Array[Double]) = {

    require(testData.size == numFeatures)

    var pred = intercept
    if (weightVector.isDefined) {
      testData.foreachActive {
        case (i, v) =>
          pred += weightVector.get(i) * v
      }
    }

    val sum = Array.fill(numFactors)(0.0)
    (0 until numFactors).foreach(f => {
      var sumSqr = 0.0
      testData.foreachActive {
        case (i, v) =>
          val d = factorMatrix(f, i) * v
          sum(f) += d
          sumSqr += d * d
      }
      pred += (sum(f) * sum(f) - sumSqr) * 0.5
    })
    (pred, sum)
  }

  def predict(testData: Vector): (Double, Array[Double]) = {
    val (pred, sum) = predictLogit(testData)
    (1.0 / (1.0 + Math.exp(-pred)), sum)
  }


  def evaluate(data: Seq[LabeledPoint]): (Seq[Double], Double, Double) = {
    val scores = data.map(_.features).map(f => predict(f)).map(_._1)
    val pred = scores.map(x => if (x >= 0.5) 1 else -1).zip(data.map(_.label.toInt))
    val accuracy = pred.count(x => x._1 == x._2).toDouble / scores.size

    val auc = AUC.of(data.map(x => {
      val tmp = x.label.toInt
      if (tmp < 0) 0 else 1
    }).toArray, scores.toArray)

    (scores, accuracy, auc)
  }

}


object FMLocal {
  private var numFeatures: Int = -1

  def train(trainData: Seq[LabeledPoint],
            testData: Seq[LabeledPoint],
            numIterations: Int,
            stepSize: Double,
            dim: (Boolean, Boolean, Int),
            regParam: (Double, Double, Double),
            initStd: Double,
            pretrainModel: FMLocal = null): FMLocal = {
    this.numFeatures = trainData.head.features.size
    require(numFeatures > 0)


    val k0 = dim._1
    val k1 = dim._2
    val k2 = dim._3
    val r0 = regParam._1
    val r1 = regParam._2
    val r2 = regParam._3

    var model = if (pretrainModel == null) {
      generateInitWeights(dim._1, dim._2, dim._3, initStd)
    } else {
      pretrainModel
    }

    (0 until numIterations).foreach(iter => {
      trainData.foreach(labelPoint => {
        val label = labelPoint.label
        val data = labelPoint.features

        val (pred, sum) = model.predictLogit(data)

        val mult = -label * (1.0 - 1.0 / (1.0 + Math.exp(-label * pred)))

        val thisIterStepSize = stepSize /// math.sqrt(iter)

        if (k0) {
          model.intercept -= thisIterStepSize * (mult + r0 * model.intercept)
          //weightsArray(len - 1) -= thisIterStepSize * (mult + r0 * weightsArray(len - 1))
        }

        if (k1) {
          val pos = numFeatures * k2
          data.foreachActive {
            case (i, v) =>
              val tmpArr = model.weightVector.get
              tmpArr(i) -= thisIterStepSize * (v * mult + r1 * tmpArr(i))
            //weightsArray(pos + i) -= thisIterStepSize * (v * mult + r1 * weightsArray(pos + i))
          }
        }

        data.foreachActive {
          case (i, v) =>
            val pos = i * k2
            for (f <- 0 until k2) {
              model.factorMatrix(f, i) -= thisIterStepSize * ((sum(f) * v - model.factorMatrix(f, i) * v * v) * mult +
                r2 * model.factorMatrix(f, i))
//              weightsArray(pos + f) -= thisIterStepSize * ((sum(f) * v - weightsArray(pos + f) * v * v) * mult +
//                r2 * weightsArray(pos + f))
            }
        }
        //        model = createModel(k0, k1, k2, Vectors.dense(weightsArray))
      })
      val (_, acc, auc) = model.evaluate(testData)
      println(s"iter $iter train acc $acc auc $auc")
    })

    model
  }


  private def generateInitWeights(k0: Boolean, k1: Boolean, k2: Int, initStd: Double): FMLocal = {

    (k0, k1) match {
      case (true, true) =>
        new FMLocal(BDM.rand(k2, numFeatures, Rand.gaussian(0, initStd)), Some(BDV.zeros(numFeatures)), 0.0)

      case (true, false) =>
        new FMLocal(BDM.rand(k2, numFeatures, Rand.gaussian(0, initStd)), None, 0.0)

      case (false, true) =>
        new FMLocal(BDM.rand(k2, numFeatures, Rand.gaussian(0, initStd)), Some(BDV.zeros(numFeatures)), 0.0)

      case (false, false) =>
        new FMLocal(BDM.rand(k2, numFeatures, Rand.gaussian(0, initStd)), None, 0.0)
    }
  }

//  private def createModel(k0: Boolean, k1: Boolean, k2: Int, weights: Vector): FMLocal = {
//
//    val values = weights.toArray
//
//    val v = new BDM(k2, numFeatures, values.slice(0, numFeatures * k2))
//
//    val w = if (k1) Some(Vectors.dense(values.slice(numFeatures * k2, numFeatures * k2 + numFeatures))) else None
//
//    val w0 = if (k0) values.last else 0.0
//
//    new FMLocal(v, w, w0)
//  }

  def loadModel(inputStream: InputStream): FMLocal = {
    val reader = new BufferedReader(new InputStreamReader(inputStream, "utf-8"))
    var b = 0.0
    val wArray = mutable.ArrayBuffer[Double]()
    val vArray = mutable.ArrayBuffer[Array[Double]]()
    var recentStat: Int = 0

    var line = reader.readLine
    while (line != null) {
      if (line.startsWith("#global bias")) recentStat = 0
      else if (line.startsWith("#unary interactions")) recentStat = 1
      else if (line.startsWith("#pairwise interactions")) recentStat = 2
      else if (recentStat == 0) b = line.trim.toDouble
      else if (recentStat == 1) wArray += line.trim.toDouble
      else if (recentStat == 2) {
        val vs = line.trim.split(" ").map(_.toDouble)
        vArray += vs
      }
      line = reader.readLine()
    }

    val factorNum = vArray.head.length
    val featNum = wArray.length

    val w = if (wArray.isEmpty) None else Some(BDV(wArray.toArray))
    val v = new BDM(factorNum, featNum, vArray.toArray.flatten)
    new FMLocal(v, w, b)
  }

  def saveModel(fMLocal: FMLocal, outputStream: OutputStream): Unit = {
    val bufferedWriter = new BufferedWriter(new OutputStreamWriter(outputStream, "utf-8"))
    bufferedWriter.write("#global bias W0\n")
    bufferedWriter.write(fMLocal.intercept + "\n")
    bufferedWriter.write("#unary interactions Wj\n")
    if (fMLocal.weightVector.isDefined)
      fMLocal.weightVector.get.toArray.foreach(w => bufferedWriter.write(w + "\n"))
    bufferedWriter.write("#pairwise interactions Vj,f\n")
    fMLocal.factorMatrix(::, *).foreach(w => bufferedWriter.write(w.toArray.mkString(" ") + "\n"))
    bufferedWriter.flush()
    bufferedWriter.close()
//    fMLocal.factorMatrix..foreach(w => bufferedWriter.write(w.toArray.mkString(" ") + "\n"))
  }
}