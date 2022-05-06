Predicting Location of Emergency Response Centers
=====================================


#### Flow of Execution and Overall Learnings

Accident Data:
For obtaining training and testing data for spark ML, please refer to the jupyternotebook training_data_generation.ipynb. For running jupyter notebook,
please make sure USA_Tennessee.geojson, traffic datasets, and nash_accidents.parquet are saved under the same folder. Traffic datasets should 
be saved under ./traffic/county=Davidson/years. After obtaining training data and testing data, please put them under folder named ./test_input, ./train_input.
The scripts of model training will automatically parse all the data files under the two folders and merge them as the training data and testing data.
The results of predictions will be saved under the outputFolder specified  while running training scripts.


  
#### Running Training Locally
```
Ensure there is a folder has the same name as outputFile

usage: 
spark-submit GBT_regressor.py --trainInput train_input --testInput test_input --outputFolder result
spark-submit random_forest.py --trainInput train_input --testInput test_input --outputFolder result
spark-submit general_regression.py --trainInput train_input --testInput test_input --outputFolder result

Each of models have several parameters which can be tuned for better performance. These parameters are set to default values, which can
be changed by parsing those arguments.
```

#### Running the optimizer to find best possible Emergency Response centers

The predicted values are generated and stored in prediction_results.csv. This contains the sum of incidents occuring on each road segment. We take these values and try to find the optimal EMR location. The optimization algorithm used here is p-median. It is an uncapacitated facilities location problem with exactly p facilities being available. This makes for an interesting challenge to find the optimal road segment amongst the thousands in Davidson county.
The results are also plotted and provided [here](https://github.com/vu-topics-in-big-data-2022/Project-Incident-Team13/blob/main/results/prediction_emr.jpg). 
