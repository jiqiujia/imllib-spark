/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.imllib.fm.optimization

import breeze.linalg._
import breeze.stats._
import scala.collection.mutable.ArrayBuffer
import breeze.linalg.{norm, DenseVector => BDV}
import org.apache.spark.annotation.{DeveloperApi, Experimental}
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import com.intel.imllib.fm.regression.FMGradient
import com.intel.imllib.util.vectorUtils._

/**
  * Class used to solve an optimization problem using Gradient Descent.
  *
  * @param gradient Gradient function to be used.
  * @param updater  Updater to be used to update weights after every iteration.
  */
class GradientDescentFM(private var gradient: Gradient, private var updater: Updater)
  extends Optimizer {

  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: (Double, Double, Double) = (0.0, 0.0, 0.0)
  private var miniBatchFraction: Double = 1.0
  private var convergenceTol: Double = 0.001
  private var dims: (Boolean, Boolean, Int) = (true, true, 1)
  private var numFeatures: Int = 1


  /**
    * Set the initial step size of SGD for the first step. Default 1.0.
    * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
    */
  def setStepSize(step: Double): this.type = {
    require(step > 0,
      s"Initial step size must be positive but got ${step}")
    this.stepSize = step
    this
  }

  def setNumFeatures(numFeatures: Int): this.type = {
    require(numFeatures > 0,
      s"numFeatures must be positive but got ${numFeatures}")
    this.numFeatures = numFeatures
    this
  }
  /**
    * :: Experimental ::
    * Set fraction of data to be used for each SGD iteration.
    * Default 1.0 (corresponding to deterministic/classical gradient descent)
    */
  @Experimental
  def setMiniBatchFraction(fraction: Double): this.type = {
    require(fraction > 0 && fraction <= 1.0,
      s"Fraction for mini-batch SGD must be in range (0, 1] but got ${fraction}")
    this.miniBatchFraction = fraction
    this
  }

  /**
    * Set the number of iterations for SGD. Default 100.
    */
  def setNumIterations(iters: Int): this.type = {
    require(iters >= 0,
      s"Number of iterations must be nonnegative but got ${iters}")
    this.numIterations = iters
    this
  }

  /**
    * Set the regularization parameter. Default 0.0.
    */
  def setRegParam(regParam: (Double, Double, Double)): this.type = {
    require(regParam._1 >= 0 && regParam._2 >=0 && regParam._3>=0,
      s"Regularization parameter must be nonnegative but got ${regParam}")
    this.regParam = regParam
    this
  }

  def setDims(dims: (Boolean, Boolean, Int)): this.type = {
    require(dims._3>=0,
      s"dims parameter must be nonnegative but got ${dims}")
    this.dims = dims
    this
  }

  /**
    * Set the convergence tolerance. Default 0.001
    * convergenceTol is a condition which decides iteration termination.
    * The end of iteration is decided based on below logic.
    *
    *  - If the norm of the new solution vector is >1, the diff of solution vectors
    * is compared to relative tolerance which means normalizing by the norm of
    * the new solution vector.
    *  - If the norm of the new solution vector is <=1, the diff of solution vectors
    * is compared to absolute tolerance which is not normalizing.
    *
    * Must be between 0.0 and 1.0 inclusively.
    */
  def setConvergenceTol(tolerance: Double): this.type = {
    require(tolerance >= 0.0 && tolerance <= 1.0,
      s"Convergence tolerance must be in range [0, 1] but got ${tolerance}")
    this.convergenceTol = tolerance
    this
  }

  /**
    * Set the gradient function (of the loss function of one single data example)
    * to be used for SGD.
    */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }


  /**
    * Set the updater function to actually perform a gradient step in a given direction.
    * The updater is responsible to perform the update from the regularization term as well,
    * and therefore determines what kind or regularization is used, if any.
    */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  /**
    * :: DeveloperApi ::
    * Runs gradient descent on the given training data.
    *
    * @param data           training data
    * @param initialWeights initial weights
    * @return solution vector
    */
  @DeveloperApi
  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val (weights, _) = GradientDescentFM.parallelSGD(
      data,
      gradient,
      updater,
      stepSize,
      numIterations,
      regParam,
      miniBatchFraction,
      initialWeights,
      convergenceTol,
      dims,
      numFeatures)

    weights
  }

  override def toString = s"GradientDescentFM(stepSize=$stepSize, numIterations=$numIterations, regParam=$regParam, miniBatchFraction=$miniBatchFraction, convergenceTol=$convergenceTol, dims=$dims, numFeatures=$numFeatures)"
}

/**
  * :: DeveloperApi ::
  * Top-level method to run gradient descent.
  */
@DeveloperApi
object GradientDescentFM {

  /**
    * Run stochastic gradient descent (SGD) in parallel using mini batches.
    * In each iteration, we sample a subset (fraction miniBatchFraction) of the total data
    * in order to compute a gradient estimate.
    * Sampling, and averaging the subgradients over this subset is performed using one standard
    * spark map-reduce in each iteration.
    *
    * @param data              Input data for SGD. RDD of the set of data examples, each of
    *                          the form (label, [feature values]).
    * @param gradient          Gradient object (used to compute the gradient of the loss function of
    *                          one single data example)
    * @param updater           Updater function to actually perform a gradient step in a given direction.
    * @param stepSize          initial step size for the first step
    * @param numIterations     number of iterations that SGD should be run.
    * @param regParam          regularization parameter
    * @param miniBatchFraction fraction of the input data set that should be used for
    *                          one iteration of SGD. Default value 1.0.
    * @param convergenceTol    Minibatch iteration will end before numIterations if the relative
    *                          difference between the current weight and the previous weight is less
    *                          than this value. In measuring convergence, L2 norm is calculated.
    *                          Default value 0.001. Must be between 0.0 and 1.0 inclusively.
    * @return A tuple containing two elements. The first element is a column matrix containing
    *         weights for every feature, and the second element is an array containing the
    *         stochastic loss computed for every iteration.
    */
  def parallelSGD(
                   data: RDD[(Double, Vector)],
                   gradient: Gradient,
                   updater: Updater,
                   stepSize: Double,
                   numIterations: Int,
                   regParam: (Double, Double, Double),
                   miniBatchFraction: Double,
                   initialWeights: Vector,
                   convergenceTol: Double,
                   dims: (Boolean, Boolean, Int),
                   numFeatures: Int): (Vector, Array[Double]) = {

    // convergenceTol should be set with non minibatch settings
    if (miniBatchFraction < 1.0 && convergenceTol > 0.0) {
      println("Testing against a convergenceTol when using miniBatchFraction " +
        "< 1.0 can be unstable because of the stochasticity in sampling.")
    }

    if (numIterations * miniBatchFraction < 1.0) {
      println("Not all examples will be used if numIterations * miniBatchFraction < 1.0: " +
        s"numIterations=$numIterations and miniBatchFraction=$miniBatchFraction")
    }

    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    // Record previous weight and current one to calculate solution vector difference

    var previousWeights: Option[Vector] = None
    var currentWeights: Option[Vector] = None

    val numExamples = data.count()

    // if no data, return initial weights to avoid NaNs
    if (numExamples == 0) {
      println("GradientDescent.runMiniBatchSGD returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }

    if (numExamples * miniBatchFraction < 1) {
      println("The miniBatchFraction is too small")
    }

    // Initialize weights as a column vector
    var weights = Vectors.dense(initialWeights.toArray)
    val n = weights.size
    val slices = data.getNumPartitions


    var converged = false // indicates whether converged based on convergenceTol
    var i = 1
    while (!converged && i <= numIterations) {
      val bcWeights = data.context.broadcast(weights)
      // Sample a subset (fraction miniBatchFraction) of the total data
      // compute and sum up the subgradients on this subset (this is one map-reduce)
      val (gradientSum, lSum, miniBatchSize) = data.sample(false, miniBatchFraction, 42+i)
        .treeAggregate(BDV.zeros[Double](n), 0.0, 0L)(
          seqOp = (c, v) => {
            val (g, loss) = gradient.asInstanceOf[FMGradient].computeFM(v._2, v._1, bcWeights.value,
              fromBreeze(c._1), stepSize, i)
            (g, c._2 + loss, c._3 + 1L)
          },
          combOp = (c1, c2) => {
            (c1._1 + c2._1, c1._2 + c2._2, c1._3 + c2._3)
          }, 2)

      val thisIterStepSize = stepSize / math.sqrt(i)
      val brzWeights: BDV[Double] = BDV(weights.toArray)
      //brzWeights :*= (1.0 - thisIterStepSize * regParam)

      val gradientAvg = gradientSum/miniBatchSize.toDouble
      if (dims._1){
        brzWeights(n-1) *= 1.0 - thisIterStepSize * regParam._1
        println("b gradient " + gradientAvg(n-1))
      }
      val pos = numFeatures * dims._3
      if (dims._2){
        (0 until numFeatures).foreach(i => brzWeights(pos+i) = (1.0 - thisIterStepSize * regParam._2)*brzWeights(pos+i))
        println("w gradient mean " + mean(gradientAvg.slice(pos, n-1)) +" std " + stddev(gradientSum.slice(pos, n-1)))
      }
      (0 until pos).foreach(i => brzWeights(i) = (1.0 - thisIterStepSize * regParam._3)*brzWeights(i))
      println("v gradient mean " + mean(gradientAvg.slice(0, pos)) +" std " + stddev(gradientSum.slice(0, pos)))

      axpy(-thisIterStepSize, gradientAvg, brzWeights)

      // TODO: 加上正则loss
      val iLoss = lSum / miniBatchSize
      stochasticLossHistory += iLoss
      val (bnorm, bRegVal) = if (dims._1){
        val bnorm = brzWeights(n-1)
        (bnorm, 0.5 * regParam._1 * bnorm * bnorm)
      } else (0.0, 0.0)

      val (wnorm, wRegVal) = if (dims._2) {
        val wnorm = norm(brzWeights.slice(pos, n-1))
        (wnorm, 0.5 * regParam._2 * wnorm * wnorm)
      } else (0.0, 0.0)

      val vnorm = norm(brzWeights.slice(0, pos))
      val vRegVal = 0.5 * regParam._3 * vnorm * vnorm

      println(s"iteration $i miniBatchSize $miniBatchSize lsum $lSum loss $iLoss " +
        s"bnorm $bnorm breg $bRegVal wnorm $wnorm wreg $wRegVal vnorm $vnorm vRegVal $vRegVal")

      weights = Vectors.dense(brzWeights.toArray)
      i += 1
    }

    (weights, stochasticLossHistory.toArray)

  }

}
