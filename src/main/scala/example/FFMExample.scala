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

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification._
import org.apache.spark.rdd.RDD

import com.intel.imllib.ffm.classification._

object FFMExample extends App {

  override def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setAppName("FFMExample"))
    
    if (args.length != 8) {
      println("FFMExample <train_file> <k> <n_iters> <eta> <lambda> " + "<normal> <random> <partitions>")
    }

    val data = sc.textFile(args(0)).map(_.split("\\s")).map(x => {
      val y = if(x(0).toInt > 0 ) 1.0 else -1.0
      val nodeArray: Array[FFMNode] = x.drop(1).map(_.split(":")).map(x => {
        val node = new FFMNode; node.f = x(0).toInt; node.j = x(1).toInt; node.v = x(2).toDouble; node
      })
      (y, nodeArray)
    }).repartition(args(7).toInt)

    val splits = data.randomSplit(Array(0.8, 0.2))
    val (training, testing) = (splits(0), splits(1))

    val m = training.flatMap(x=>x._2).map(_.f).collect.reduceLeft(_ max _) //+ 1
    val n = training.flatMap(x=>x._2).map(_.j).collect.reduceLeft(_ max _) //+ 1

    val param: FFMParameter = new FFMParameter().defaultParameter
    param.k = args(1).toInt
    param.n_iters = args(2).toInt
    param.eta = args(3).toDouble
    param.lambda = args(4).toDouble
    param.normalization = args(5).toBoolean
    param.random = args(6).toBoolean

println("dataset feature:" + n + ",field:" + m + ",k:" + param.k + ",stepsize:" + param.eta)

    val ffm: FFMModel = FFMWithAdag.train(training, m, n, param, "adagrad")
    val scores: RDD[(Double, Double)] = testing.map(x => {
      val p = ffm.predict(x._2)
      val ret = if (p >= 0.5) 1.0 else -1.0
      (ret, x._1)
    })
    val accuracy = scores.filter(x => x._1 == x._2).count().toDouble / scores.count()
    println(s"accuracy = $accuracy")
  }
}

