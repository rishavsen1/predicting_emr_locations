import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import when

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import RegressionMetrics
import pyspark
import glob
import os
import pandas as pd

def process(params):

    #
    # Initializing Spark session
    #
    conf = pyspark.SparkConf().setAll([('spark.executor.memory', '16g'), ('spark.driver.memory', '16g')])

    sc = pyspark.SparkContext(conf=conf)
    sparkSession = (SparkSession.builder
      .appName("random_forest")
      .getOrCreate())

    #*************************************************
    print("Reading data from train.csv file")
    #*************************************************
    featureCols = ['xd_id', 'speed', 'average_speed', 'travel_time_seconds', 'congestion', 'tstamp',
                   'extreme_congestion', 'type']

    path = os.getcwd() + '/' + params.trainInput
    train_files = glob.glob(os.path.join(path, "*.csv"))
    for idx, f in enumerate(train_files):
        if idx == 0:
            trainInput = (sparkSession.read
              .option("header", "true")
              .option("inferSchema", "true")
              .csv(f)
              .cache()).select(featureCols)
        else:
            temp = (sparkSession.read
              .option("header", "true")
              .option("inferSchema", "true")
              .csv(f)
              .cache()).select(featureCols)
            trainInput = trainInput.union(temp)

    path = os.getcwd() + '/' + params.testInput
    test_files = glob.glob(os.path.join(path, "*.csv"))
    for idx, f in enumerate(test_files):
        if idx == 0:
            testInput = (sparkSession.read
              .option("header", "true")
              .option("inferSchema", "true")
              .csv(f)
              .cache()).select(featureCols)
        else:
            temp = (sparkSession.read
              .option("header", "true")
              .option("inferSchema", "true")
              .csv(f)
              .cache()).select(featureCols)
            testInput = testInput.union(temp)

    #*****************************************
    print("Preparing data for training model")
    #*****************************************

    data = (trainInput.withColumnRenamed("type", "label").sample(False, params.trainSample))

    [trainingData, validationData] = data.randomSplit([0.7, 0.3])

    trainingData.cache()

    validationData.cache()
    testInput.cache()
    testData = testInput.withColumnRenamed("type", "label").sample(False, params.testSample).cache()


    #******************************************
    print("Building Machine Learning pipeline")
    #******************************************

    categNewCol = lambda c: "idx_{0}".format(c) if c else c
    featureCols = ['xd_id', 'speed', 'average_speed', 'travel_time_seconds', 'congestion',
                   'extreme_congestion']
    stringIndexerStages = list(map(lambda c: StringIndexer(inputCol=c, outputCol=categNewCol(c)).fit(trainingData.select(c)), featureCols))
    assembler = VectorAssembler(inputCols=featureCols, outputCol="features")

    #Estimator algorithm
    algo = RandomForestRegressor(featuresCol="features", labelCol="label")
    stages = stringIndexerStages
    stages.append(assembler)
    stages.append(algo)

    #Building the Pipeline for transformations and predictor
    pipeline = Pipeline(stages=stages)


    #*********************************************************
    print("Preparing K-fold Cross Validation and Grid Search")
    #*********************************************************


    paramGrid = (ParamGridBuilder()
      .addGrid(algo.numTrees, params.algoNumTrees)
      .addGrid(algo.maxDepth, params.algoMaxDepth)
      .addGrid(algo.maxBins, params.algoMaxBins)
      .build())
      
    cv = CrossValidator(estimator=pipeline,
                        evaluator=RegressionEvaluator(),
                        estimatorParamMaps=paramGrid,
                        numFolds=params.numFolds)


    #**********************************************************
    print("Training model with RandomForest algorithm")
    #**********************************************************

    cvModel = cv.fit(trainingData)


    #********************************************************************
    print("Evaluating model on train and test data and calculating RMSE")
    #********************************************************************

    #trainPredictionsAndLabels = cvModel.transform(trainingData).select("label", "prediction").rdd
    trainPredictionsAndLabels = cvModel.transform(trainingData).select("label", "prediction")
    trainPredictionsAndLabels = trainPredictionsAndLabels.withColumn("prediction", when(trainPredictionsAndLabels.prediction < 0.5,0.0).otherwise(1.0)).rdd
    validPredictionsAndLabels = cvModel.transform(validationData).select("label", "prediction")
    validPredictionsAndLabels = validPredictionsAndLabels.withColumn("prediction", when(validPredictionsAndLabels.prediction < 0.5,0.0).otherwise(1.0)).rdd

    trainRegressionMetrics = RegressionMetrics(trainPredictionsAndLabels)
    validRegressionMetrics = RegressionMetrics(validPredictionsAndLabels)

    bestModel = cvModel.bestModel
    featureImportances = bestModel.stages[-1].featureImportances.toArray()

    output = ("\n=====================================================================\n" +
      "Param trainSample: {0}\n".format(str(params.trainSample)) +
      "Param testSample: {0}\n".format(str(params.testSample)) +
      "TrainingData count: {0}\n".format(str(trainingData.count())) +
      "ValidationData count: {0}\n".format(str(validationData.count())) +
      "TestData count: {0}\n".format(str(testData.count())) +
      "=====================================================================\n" +
      "Param algoNumTrees = {0}\n".format(",".join(str(params.algoNumTrees))) +
      "Param algoMaxDepth = {0}\n".format(",".join(str(params.algoMaxDepth))) +
      "Param algoMaxBins = {0}\n".format(",".join(str(params.algoMaxBins))) +
      "Param numFolds = {0}\n".format(str(params.numFolds)) +
      "=====================================================================\n" +
      "Training data MSE = {0}\n".format(str(trainRegressionMetrics.meanSquaredError)) +
      "Training data RMSE = {0}\n".format(str(trainRegressionMetrics.rootMeanSquaredError)) +
      "Training data R-squared = {0}\n".format(str(trainRegressionMetrics.r2)) +
      "Training data MAE = {0}\n".format(str(trainRegressionMetrics.meanAbsoluteError)) +
      "Training data Explained variance = {0}\n".format(str(trainRegressionMetrics.explainedVariance)) +
      "=====================================================================\n" +
      "Validation data MSE = {0}\n".format(str(validRegressionMetrics.meanSquaredError)) +
      "Validation data RMSE = {0}\n".format(str(validRegressionMetrics.rootMeanSquaredError)) +
      "Validation data R-squared = {0}\n".format(str(validRegressionMetrics.r2)) +
      "Validation data MAE = {0}\n".format(str(validRegressionMetrics.meanAbsoluteError)) +
      "Validation data Explained variance = {0}\n".format(str(validRegressionMetrics.explainedVariance)) +
      "=====================================================================\n")

    print(output)


    #*****************************************
    print("Run prediction over test dataset")
    #*****************************************
    predictions = cvModel.transform(testData).select("xd_id", "tstamp", "label", "prediction")
    all_results = predictions
    predictions = predictions.withColumn("prediction", when(predictions.prediction < 0.7, 0.0).otherwise(1.0))
    prediction_results = predictions.groupBy('xd_id').sum('prediction')
    true_results = predictions.groupBy('xd_id').sum('label')
    TP = predictions.where((predictions.label == 1) & (predictions.prediction == 1)).count()
    FP = predictions.where((predictions.label == 0) & (predictions.prediction == 1)).count()
    FN = predictions.where((predictions.label == 1) & (predictions.prediction == 0)).count()

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2 * (precision*recall/(precision+recall))
    #accuracy = predictions.where(predictions.label == predictions.prediction).count()/predictions.count()
    output = ("\n=====================================================================\n" +
      "=====================================================================\n" +
      "=====================================================================\n" +
      "F1 Score On Test Set = {0}\n".format(str(F1)) +
      "Precision On Test Set = {0}\n".format(str(precision)) +
      "Recall On Test Set = {0}\n".format(str(recall)) +
      "=====================================================================\n" +
      "=====================================================================\n" +
      "=====================================================================\n")
    print(output)
    if params.outputFile:
        all_results.coalesce(1).write.format("csv").option("header", "true").save('./' + params.outputFile + '/all_include')
        prediction_results.coalesce(1).write.format("csv").option("header", "true").save('./' + params.outputFile + '/prediction')
        true_results.coalesce(1).write.format("csv").option("header", "true").save('./' + params.outputFile + '/true_results')
#
 # entry point - main method
 #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainInput",  help="Path to file/directory for training data", required=True)
    parser.add_argument("--testInput",   help="Path to file/directory for test data", required=True)
    parser.add_argument("--outputFolder",  help="Path to output file")
    parser.add_argument("--algoNumTrees", nargs='+', type=int, help="One or more options for number of trees for RandomForest model. Default: 3", default=[12])
    parser.add_argument("--algoMaxDepth", nargs='+', type=int, help="One or more values for depth limit. Default: 4", default=[16])
    parser.add_argument("--algoMaxBins",  nargs='+', type=int, help="One or more values for max bins for RandomForest model. Default: 32", default=[64])
    parser.add_argument("--numFolds",    type=int,   help="Number of folds for K-fold Cross Validation. Default: 10", default=10)
    parser.add_argument("--trainSample", type=float, help="Sample fraction from 0.0 to 1.0 for train data", default=1.0)
    parser.add_argument("--testSample",  type=float, help="Sample fraction from 0.0 to 1.0 for test data", default=1.0)

    params = parser.parse_args()

    process(params)