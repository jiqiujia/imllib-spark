package com.tencent.dp.nlp.angel.util

import algorithm.utils.io.DataLoader
import algorithm.utils.{GeneralArgParser, IOHandler}

import scala.collection.JavaConversions._

object MergeAngelEmbedding {
  def main(args: Array[String]): Unit = {
    val argParser = new GeneralArgParser(args)
    val inPath = argParser.getStringValue("inPath")
    val mappingPath = argParser.getStringValue("mappingPath")
    val outPath = argParser.getStringValue("outPath")

    val io = new IOHandler("")
    val sc = io.getSparkContext()

    val mapping = DataLoader.readLines(mappingPath)
      .map(x => x.split("\t", -1))
      .map(x => (x(1), x(0)))
      .toMap

    val data = sc.textFile(inPath)
      .map(x => x.split(":", -1))
      .map(x => mapping(x.head) + " " + x(1))
      .collect()
      .toSeq

    DataLoader.writeLines(data, io.getOuputStream(outPath).get)

  }
}
