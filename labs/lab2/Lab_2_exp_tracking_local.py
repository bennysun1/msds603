# %%
"""
# Experiment Tracking with MLFlow (Local)

In this demo we will see how to use MLFlow for tracking experiments, using a toy data set. In the attached lab (below), you will download a larger dataset and attempt to train the best model that you can.

We should first install mlflow, and add it to the requirements.txt file if not done already.

`pip install mlflow` or `python3 -m pip install mlflow`.

You may also need to `pip install setuptools`.

From here, make sure to save this notebook in a specific folder, and ensure you run all command line commands from the same folder.
"""

# %%
import mlflow
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer


# %%
"""
After loading the libraries, we can first check the mlflow version you have. And, just for fun, let's look at the mlflow UI by running `mlflow ui`. After this, we should do two things:
- set the tracking uri
- create or set the experiment

Setting the tracking uri tells mlflow where to save the results of our experiments. We will first save these locally in a sqlite instance. In a future lab we will set up mlflow to run in GCP.

If you've already created an experiment previously that you'd like to use, you can tell mlflow by setting the experiment. You can also use `set_experiment` even if the experiment has not yet been created - mlflow will first check if the experiment exists, and if not, it will create it for you. 
"""

# %%
mlflow.__version__

# %%
"""
Running the below code will create a sqlite database and an mlruns folder in the current directory.
"""

# %%
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('demo-experiment')

# %%
"""
From here, we can load the wine data from sklearn and take a look at it. Then let's play around with some models, without using mlflow for now, to get a sense of why mlflow might come in handy.
"""

# %%
wine = load_wine()
df_wine = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df_wine.head(3)

# %%
y = wine.target
X = df_wine
dt = DecisionTreeClassifier(max_depth=4)
dt.fit(X, y)

# %%
accuracy_score(y, dt.predict(df_wine))

# %%
"""
## Train a Model Using MLFLow

In this section, let's train a simple decision tree model, where we will now adjust the maximum depth (`max_depth`) of the tree, and save the results of each run of the experiment using mlflow. To do so, we need to tell mlflow to start recording. We do this with `start_run`. 

The things we might want to record in this simple case are:
- the value of `max_depth`
- the corresponding accuracy of the model

We can also tag each run to make it easier to identify them later.

After running the below code, be sure to check the mlflow UI by running the following in the terminal from the same directory as where you saved this notebook:

`mlflow ui` note that just running this you will not see any of your experiments. You must specify the uri (the place where all of your results are being stored)

`mlflow ui --backend-store-uri sqlite:///mlflow.db`
"""

# %%
with mlflow.start_run():
    # log parameters and log metrics
    # parameters: hyperparameters
    # metrics: model performance metrics

    mlflow.set_tags({"Model":"decision-tree", "Train Data": "all-data"})

    tree_depth = 5
    dt = DecisionTreeClassifier(max_depth=tree_depth)
    dt.fit(X, y)
    acc = accuracy_score(y, dt.predict(df_wine))

    mlflow.log_param("max_depth", tree_depth)
    mlflow.log_metric("accuracy", acc)

mlflow.end_run()

# %%
"""
Let's do it again, but this time we'll use a random forest, which has some other hyperparameters we can tune, which makes keeping track of things a little more complex without a tool like mlflow.
"""

# %%
from sklearn.ensemble import RandomForestClassifier

with mlflow.start_run():
    mlflow.set_tags({"Model":"random-forest", "Train Data": "all-data"})

    ntree = 1000
    mtry = 4

    mlflow.log_params({'n_estimators':ntree, 'max_features':mtry})

    rf = RandomForestClassifier(n_estimators = ntree, max_features = mtry, oob_score = True)
    rf.fit(X,y)
    acc = rf.oob_score_
    #acc = accuracy_score(y, rf.predict(X))
    mlflow.log_metric('accuracy', acc)

mlflow.end_run()

# %%
"""
Typically, in a real-world scenario, you wouldn't change your parameter values manually and re-run your code, you would either use a loop to loop through different parameter values, or you'd use a built-in method for doing cross-validation, of which there are a few. First, let's use a simple loop to run the experiment multiple times, and save the results of each run.
"""

# %%
ntrees = [20,40,60,80,100]
mtrys = [3,4,5]
for i in ntrees:
    for j in mtrys:
        with mlflow.start_run():
            mlflow.set_tags({"Model":"random-forest", "Train Data": "all-data"})

            mlflow.log_params({'n_estimators':i, 'max_features':j})

            rf = RandomForestClassifier(n_estimators = i, max_features = j, oob_score = True)
            rf.fit(X,y)
            acc = rf.oob_score_
            #acc = accuracy_score(y, rf.predict(X))
            mlflow.log_metric('accuracy', acc)
        mlflow.end_run()

# %%
"""
## Training a Model with mlflow and hyperopt

One way of tuning your model is to use the `hyperopt` library. `hyperopt` is a library that does hyperparameter tuning, and does so in a way that makes it easy for mlflow to keep track of the results. 

First, install the libraries you don't have, and then load them below. We do not use `hyperopt` much in the class, so if you don't want to add it to your requirements.txt file, you don't have to.

For this exercise, we'll split the data into training and validation, and then we'll train decision trees and random forests and use `hyperopt` to do the hyperparameter tuning and find the best model for us.
"""

# %%
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# %%
"""
From the above we will use `cross_val_score` for our metric, `fmin` is used by `hyperopt` to do the tuning, `tpe` (Tree of Parzen Estimators) is the algorithm used to search the hyperparameter space,  `hp` has methods we need to use for defining our search space, `STATUS_OK` is a status message that each run completed, and `Trials` keeps track of each run.
"""

# %%
def objective(params):
    with mlflow.start_run():
        classifier_type = params['type']
        del params['type']
        if classifier_type == 'dt':
            clf = DecisionTreeClassifier(**params)
        elif classifier_type == 'rf':
            clf = RandomForestClassifier(**params)        
        else:
            return 0
        acc = cross_val_score(clf, X, y).mean()

        mlflow.set_tag("Model", classifier_type)
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.end_run()
        return {'loss': -acc, 'status': STATUS_OK}

search_space = hp.choice('classifier_type', [
    {
        'type': 'dt',
        'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
        'max_depth': hp.choice('dtree_max_depth', [None, hp.randint('dtree_max_depth_int', 1,10)]),
        'min_samples_split': hp.randint('dtree_min_samples_split', 2,10)
    },
    {
        'type': 'rf',
        'n_estimators': hp.randint('rf_n_estimators', 20, 500),
        'max_features': hp.randint('rf_max_features', 2,9),
        'criterion': hp.choice('criterion', ['gini', 'entropy'])
    },
])

algo = tpe.suggest
trials = Trials()

# %%
best_result = fmin(
        fn=objective, 
        space=search_space,
        algo=algo,
        max_evals=32,
        trials=trials)

# %%
best_result

# %%
"""
### Using Autologging

Rather than manually logging parameters and metrics, mlflow has an autolog feature, which is compatible with a subset of python libraries, such as sklearn. Autologging makes it easy to log all of the important stuff, without having to manually write lines of code to log the parameters. However, sometimes you will want to have finer control over what gets logged, and should instead skip autologging.
"""

# %%

with mlflow.start_run():
    mlflow.sklearn.autolog()
    tree_depth = 5
    dt = DecisionTreeClassifier(max_depth=tree_depth)
    dt.fit(X_train, y_train)
    mlflow.sklearn.autolog(disable=True)
mlflow.end_run()

# %%
"""
# Artifact Tracking and Model Registry (Local)

In this section we will save some artifacts from our model as we go through the model development process. There are a few things that might be worth saving, such as datasets, plots, and the final model itself that might go into production later.

## Data

First, let's see how we can store our important datasets, in a compressed format, for use for later, for example, in case we get a new request about our model and need to run some analyses (such as "what is the distribution of this feature, but only for this specific subset of data?" or "how did the model do on these particular observations from your validation set?").
"""

# %%
import os 

os.makedirs('save_data', exist_ok = True)

X_train.to_parquet('save_data/x_train.parquet')

mlflow.log_artifact('save_data/x_train.parquet')

# %%
X_test.to_parquet('save_data/x_test.parquet')

mlflow.log_artifacts('save_data/')

# %%
"""
You can now go to the mlflow UI, click on the latest run, and select the Artifacts tab. You should see something similar to this:
![mlflow1.png](attachment:mlflow1.png)
"""

# %%
"""
## Images

As part of the model dev process you may end up creating visualizations that can be useful for analysis, or for reporting. You can use mlflow to log the important ones and ignore the rest. After creating the below figure, save into a folder called images, and then you can log whatever is in the `images` folder as an artifact.
"""

# %%
%matplotlib inline
os.makedirs('images', exist_ok = True)
X_train.plot.density(subplots = True, figsize = (20,10), layout = (4,4), sharey = False, sharex = False)

# %%
mlflow.log_artifacts('images')
mlflow.end_run()

# %%
"""
Notice how all of the artifacts were saved in the same run of the experiment. We could have added an `mlflow.end_run()` in between our `log_artifacts` lines to separate runs if we wanted to.
"""

# %%
"""
## Model Management and Model Registry

As you are developing your models you may want to save certain versions of the model, or maybe even all of them, so that you don't have to go back and retrain them later. We can do this in mlflow by logging the models, not as artifacts, but as models, using `log_model`. 

In this section we'll log a couple of models to see how mlflow handles model management. Above, we used `hyperopt` to train a bunch of models at once. Let's do this again, and log some of the models that we train.

### Logging as an Artifact

First we can try logging a model as an artifact. To do this, we must first save the model itself, which we can do by using the `pickle` library. We then log the model as an artifact like we did with data and images. 
"""

# %%
import pickle

os.makedirs('../models', exist_ok = True)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

with open('../models/model.pkl','wb') as f:
    pickle.dump(dt,f)

# First we'll log the model as an artifact
mlflow.log_artifact('../models/model.pkl', artifact_path='my_models')

# %%
"""
### Logging as a Model

Logging the model as an artifact only logs the pickle file (the serialized version of the model). It's not really very useful, especially since models contain so much metadata that might be critical to know for deploying the model later. mlflow has a built-in way of logging models specifically, so let's see how to use this, and how it's different from logging models as an artifact.
"""

# %%
# Let's do it again, but this time we will log the model using log_model
mlflow.sklearn.log_model(dt, artifact_path = 'better_models')
mlflow.end_run()

# %%
"""
Ok, so if you go to the mlflow UI at this point you can see the difference in `log_artifact`, which simply logs the pickle file, and `log_model`, which also gives you information about the environment, required packages, and model flavor.

![mlflow2.png](attachment:mlflow2.png)

Let's do this one more time, but this time let's use `hyperopt` and log all of the trained models separately. Let's do this in a new experiment called 'demo-experiment2'. 
"""

# %%
mlflow.set_experiment('demo-experiment2')
def objective(params):
    with mlflow.start_run():
        classifier_type = params['type']
        del params['type']
        if classifier_type == 'dt':
            clf = DecisionTreeClassifier(**params)
        elif classifier_type == 'rf':
            clf = RandomForestClassifier(**params)        
        else:
            return 0
        acc = cross_val_score(clf, X, y).mean()

        mlflow.set_tag("Model", classifier_type)
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, artifact_path = 'better_models')
        mlflow.end_run()
        return {'loss': -acc, 'status': STATUS_OK}
search_space = hp.choice('classifier_type', [
    {
        'type': 'dt',
        'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
        'max_depth': hp.choice('dtree_max_depth', [None, hp.randint('dtree_max_depth_int', 1,10)]),
        'min_samples_split': hp.randint('dtree_min_samples_split', 2,10)
    },
    {
        'type': 'rf',
        'n_estimators': hp.randint('rf_n_estimators', 20, 500),
        'max_features': hp.randint('rf_max_features', 2,9),
        'criterion': hp.choice('criterion', ['gini', 'entropy'])
    },
])

algo = tpe.suggest
trials = Trials()
best_result = fmin(
        fn=objective, 
        space=search_space,
        algo=algo,
        max_evals=32,
        trials=trials)

# %%
"""
### Loading Models

Now that models have been logged, you can load specific models back into python for predicting and further analysis. There are two main ways to do this. The mlflow UI actually gives you some instructions, with code that you copy and paste.
"""

# %%
logged_model = 'runs:/2a8dc11914a64eb2a715b000411d8d1d/better_models' 

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
loaded_model

# %%
sklearn_model = mlflow.sklearn.load_model(logged_model)
sklearn_model

# %%
sklearn_model.fit(X_train, y_train)
preds = sklearn_model.predict(X_test)
preds[:5]

# %%
"""
### Model Registry

Typically, you will **register** your *chosen* model, the model you plan to put into production. But, sometimes, after you've chosen and registered a model, you may need to replace that model with a new version. For example, the model may have gone into production and started to degrade in performance, and so the model needed to be retrained. Or, you go to deploy your model and notice an error or bug, and now have to go back and retrain it.

In this section let's see how we take our logged models and register them in the model registry, which then can get picked up by the production process, or engineer, for deployment. First, I'll demonstrate how this is done within the UI, but then below I'll show how we can use the python API to do the same thing.
"""

# %%
runid = '2a8dc11914a64eb2a715b000411d8d1d'
mod_path = f'runs:/{runid}/artifacts/better_models'
mlflow.register_model(model_uri = mod_path, name = 'wine_model_from_nb')

# %%
"""
## Common Issues

- Nothing is appearing in the MLFlow UI: make sure you start the UI from the directory where your notebook is running.  
- I installed a library, but I'm getting an error loading it: be sure you installed it in the same environment as where your notebook is running.  
- MLFlow doesn't seem to be working at all, I'm just getting errors all over the place: this might be a versioning issue.  
- The UI broke, I can't get back to it: `sudo lsof -i :5000 | awk '{print $2}' | tail -n +2 | xargs kill`

"""

# %%
"""

# Experiment Tracking and Model Registry Lab

## Overview

In this lab you will each download a new dataset and attempt to train a good model, and use mlflow to keep track of all of your experiments, log your metrics, artifacts and models, and then register a final set of models for "deployment", though we won't actually deploy them anywhere yet.

## Goal

Your goal is **not** to become a master at MLFlow - this is not a course on learning all of the ins and outs of MLFlow. Instead, your goal is to understand when and why it is important to track your model development process (tracking experiments, artifacts and models) and to get into the habit of doing so, and then learn at least the basics of how MLFlow helps you do this so that you can then compare with other tools that are available.

## Data

You can choose your own dataset to use here. It will be helpful to choose a dataset that is already fairly clean and easy to work with. You can even use a dataset that you've used in a previous course. We will do a lot of labs where we do different things with datasets, so if you can find one that is interesting enough for modeling, it should work for most of the rest of the course. 

There are tons of places where you can find open public datasets. Choose something that interests you, but don't overthink it.

[Kaggle Datasets](https://www.kaggle.com/datasets)  
[HuggingFace Datasets](https://huggingface.co/docs/datasets/index)  
[Dagshub Datasets](https://dagshub.com/datasets/)  
[UCI](https://archive.ics.uci.edu/ml/datasets.php)  
[Open Data on AWS](https://registry.opendata.aws/)  
[Yelp](https://www.yelp.com/dataset)  
[MovieLens](https://grouplens.org/datasets/movielens/)  
And so many more...

## Instructions

Once you have selected a set of data, create a brand new experiment in MLFlow and begin exploring your data. Do some EDA, clean up, and learn about your data. You do not need to begin tracking anything yet, but you can if you want to (e.g. you can log different versions of your data as you clean it up and do any feature engineering). Do not spend a ton of time on this part. Your goal isn't really to build a great model, so don't spend hours on feature engineering and missing data imputation and things like that.

Once your data is clean, begin training models and tracking your experiments. If you intend to use this same dataset for your final project, then start thinking about what your model might look like when you actually deploy it. For example, when you engineer new features, be sure to save the code that does this, as you will need this in the future. If your final model has 1000 complex features, you might have a difficult time deploying it later on. If your final model takes 15 minutes to train, or takes a long time to score a new batch of data, you may want to think about training a less complex model.

Now, when tracking your experiments, at a *minimum*, you should:

1. Try at least 3 different ML algorithms (e.g. linear regression, decision tree, random forest, etc.).
2. Do hyperparameter tuning for **each** algorithm.
3. Do some very basic feature selection, and repeat the above steps with these reduced sets of features.
4. Identify the top 3 best models and note these down for later.
6. Choose the **final** "best" model that you would deploy or use on future data, stage it (in MLFlow), and run it on the test set to get a final measure of performance. Don't forget to log the test set metric.
7. Be sure you logged the exact training, validation, and testing datasets for the 3 best models, as well as hyperparameter values, and the values of your metrics.  
8. Push your code to Github. No need to track the mlruns folder, the images folder, any datasets, or the sqlite database in git.

### Turning It In

In the MLFlow UI, next to the refresh button you should see three vertical dots. Click the dots and then download your experiments as a csv file. Open the csv file in Excel and highlight the rows for your top 3 models from step 4, highlight the run where you applied your best model to the test set, and then save as an excel file. Take a snapshot of the Models page in the MLFLow UI showing the model you staged in step 6 above. Submit the excel file and the snapshot to Canvas.
"""

# %%
"""

"""

# %%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")

print("Path to dataset files:", path)

# %%
heart_df = pd.read_csv(path + '/heart.csv')
heart_df.head(3)

# %%
X = heart_df.drop('HeartDisease', axis=1)
y = heart_df['HeartDisease']

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# %%
from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# %%
# Save datasets as artifacts
heart_df.to_parquet('/Users/bensunshine/repos/msds603/data/heart_df.parquet')
X_train.to_parquet('/Users/bensunshine/repos/msds603/data/X_train.parquet')
X_val.to_parquet('/Users/bensunshine/repos/msds603/data/X_val.parquet')
X_test.to_parquet('/Users/bensunshine/repos/msds603/data/X_test.parquet')
pd.Series(y_train).to_csv('/Users/bensunshine/repos/msds603/data/y_train.csv')
pd.Series(y_val).to_csv('/Users/bensunshine/repos/msds603/data/y_val.csv')
pd.Series(y_test).to_csv('/Users/bensunshine/repos/msds603/data/y_test.csv')

# %%
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(10, 6))
sns.countplot(x='HeartDisease', data=heart_df)
plt.title('Distribution of Heart Disease')
plt.show()
    
# Only include numeric columns in correlation
numeric_df = heart_df.select_dtypes(include=['number'])
plt.figure(figsize=(12, 10))
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.show()

# Convert categorical variables for additional analysis
# Create a copy to avoid modifying the original dataframe
heart_encoded = heart_df.copy()

# One-hot encode categorical variables
categorical_cols = heart_df.select_dtypes(include=['object']).columns
heart_encoded = pd.get_dummies(heart_df, columns=categorical_cols, drop_first=True)

# Now you can compute correlation with the encoded variables
plt.figure(figsize=(14, 12))
correlation_encoded = heart_encoded.corr()
sns.heatmap(correlation_encoded, annot=True, cmap='coolwarm', annot_kws={"size": 8})
plt.title('Correlation Matrix with Encoded Categorical Features')
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle

# tracking
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('heart-failure-prediction-hyperopt')


def create_preprocessing_pipeline(categorical_cols, numerical_cols):
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor


def objective(params):
    with mlflow.start_run(nested=True):
        # Extract model type
        classifier_type = params['type']
        del params['type']
        
        # Convert float parameters to int where required
        if classifier_type == 'dt':
            if 'min_samples_split' in params:
                params['min_samples_split'] = int(params['min_samples_split'])
            if 'min_samples_leaf' in params:
                params['min_samples_leaf'] = int(params['min_samples_leaf'])
            clf = DecisionTreeClassifier(**params, random_state=42)
            model_name = "DecisionTree"
        elif classifier_type == 'rf':
            if 'n_estimators' in params:
                params['n_estimators'] = int(params['n_estimators'])
            if 'min_samples_split' in params:
                params['min_samples_split'] = int(params['min_samples_split'])
            if 'min_samples_leaf' in params:
                params['min_samples_leaf'] = int(params['min_samples_leaf'])
            clf = RandomForestClassifier(**params, random_state=42)
            model_name = "RandomForest"
        elif classifier_type == 'gb':
            if 'n_estimators' in params:
                params['n_estimators'] = int(params['n_estimators'])
            if 'max_depth' in params:
                params['max_depth'] = int(params['max_depth'])
            if 'min_samples_split' in params:
                params['min_samples_split'] = int(params['min_samples_split'])
            clf = GradientBoostingClassifier(**params, random_state=42)
            model_name = "GradientBoosting"
        else:
            return {'loss': 0, 'status': STATUS_OK}  # Skip invalid model types
        
        # Create pipeline with preprocessing
        pipeline = Pipeline([
            ('preprocessor', create_preprocessing_pipeline(categorical_cols, numerical_cols)),
            ('classifier', clf)
        ])
        
        # Evaluate on validation set instead of cross-validation to avoid errors
        pipeline.fit(X_train, y_train)
        
        # Predict on validation set
        y_val_pred = pipeline.predict(X_val)
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        val_roc_auc = roc_auc_score(y_val, y_val_proba)
        
        mlflow.set_tag("model_type", model_name)
        
        for key, value in params.items():
            mlflow.log_param(key, value)

        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_f1", val_f1)
        mlflow.log_metric("val_roc_auc", val_roc_auc)
        
        mlflow.sklearn.log_model(pipeline, artifact_path=f"{model_name}_model")
        
        with open(f'models/{model_name}_{val_roc_auc:.4f}.pkl', 'wb') as f:
            pickle.dump(pipeline, f)
        
        print(f"Trained {model_name} - Validation ROC AUC: {val_roc_auc:.4f}")
        
        return {'loss': -val_roc_auc, 'status': STATUS_OK, 'model': pipeline, 'model_name': model_name}

search_space = hp.choice('classifier_type', [
    # Decision Tree
    {
        'type': 'dt',
        'criterion': hp.choice('dt_criterion', ['gini', 'entropy']),
        'max_depth': hp.choice('dt_max_depth', [None, 3, 5, 7, 10, 15]),
        'min_samples_split': hp.quniform('dt_min_samples_split', 2, 20, 1),
        'min_samples_leaf': hp.quniform('dt_min_samples_leaf', 1, 10, 1)
    },
    # Random Forest
    {
        'type': 'rf',
        'n_estimators': hp.quniform('rf_n_estimators', 10, 300, 10),
        'max_depth': hp.choice('rf_max_depth', [None, 5, 10, 15, 20]),
        'min_samples_split': hp.quniform('rf_min_samples_split', 2, 10, 1),
        'min_samples_leaf': hp.quniform('rf_min_samples_leaf', 1, 4, 1),
        'criterion': hp.choice('rf_criterion', ['gini', 'entropy'])
    },
    # Gradient Boosting
    {
        'type': 'gb',
        'n_estimators': hp.quniform('gb_n_estimators', 10, 200, 10),
        'learning_rate': hp.loguniform('gb_learning_rate', np.log(0.01), np.log(0.2)),
        'max_depth': hp.quniform('gb_max_depth', 3, 10, 1),
        'min_samples_split': hp.quniform('gb_min_samples_split', 2, 10, 1),
        'subsample': hp.uniform('gb_subsample', 0.7, 1.0)
    }
])


with mlflow.start_run(run_name="HyperparameterOptimization"):
    mlflow.set_tag("step", "HyperparameterOptimization")
    
    # Log artifacts created during EDA
    mlflow.log_artifacts('data', artifact_path='datasets')
    
    # Configure hyperopt
    algo = tpe.suggest
    trials = Trials()
    
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=algo,
        max_evals=32, 
        trials=trials,
        verbose=1
    )
    
    best_trial = trials.best_trial
    best_val = -trials.best_trial['result']['loss']
    best_model_name = best_trial['result']['model_name']
    best_model = best_trial['result']['model']
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best validation ROC AUC: {best_val:.4f}")
    
    # Log best trial info
    mlflow.log_metric("best_val_roc_auc", best_val)
    mlflow.log_param("best_model_type", best_model_name)



# %%
# Feature selection using the best model type from hyperopt
print("\n========== Feature Selection ==========")
with mlflow.start_run(run_name="FeatureSelection"):
    mlflow.set_tag("step", "FeatureSelection")
    
    # Process data for feature selection
    preprocessor = create_preprocessing_pipeline(categorical_cols, numerical_cols)
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Try different k values for SelectKBest
    k_values = [5, 8, 10, 12]
    best_k_score = 0
    best_k_model = None
    best_k = None
    
    for k in k_values:
        with mlflow.start_run(nested=True):
            mlflow.log_param("k_features", k)
            
            # Create a feature selector
            selector = SelectKBest(f_classif, k=k)
            
            # Determine the best model type based on hyperopt results
            if best_model_name == "DecisionTree":
                clf = DecisionTreeClassifier(max_depth=5, random_state=42)
            elif best_model_name == "RandomForest":
                clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            else:  # GradientBoosting
                clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
            
            # Create pipeline with feature selection
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('selector', selector),
                ('classifier', clf)
            ])
            
            # Train and evaluate
            pipeline.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_val_pred = pipeline.predict(X_val)
            y_val_proba = pipeline.predict_proba(X_val)[:, 1]
            
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred)
            val_roc_auc = roc_auc_score(y_val, y_val_proba)
            
            # Log metrics
            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.log_metric("val_f1", val_f1)
            mlflow.log_metric("val_roc_auc", val_roc_auc)
            
            # Log feature indices
            feature_indices = selector.get_support(indices=True)
            mlflow.log_param("selected_features", str(feature_indices))
            
            # Log the model
            mlflow.sklearn.log_model(pipeline, artifact_path=f"feature_selection_k{k}")
            
            print(f"Feature selection with k={k}: Validation ROC AUC = {val_roc_auc:.4f}")
            
            # Keep track of the best k
            if val_roc_auc > best_k_score:
                best_k_score = val_roc_auc
                best_k_model = pipeline
                best_k = k
    
    # Log best k
    mlflow.log_param("best_k", best_k)
    mlflow.log_metric("best_k_val_roc_auc", best_k_score)
    
    # Save best feature selection model
    with open(f'models/best_feature_selection_k{best_k}.pkl', 'wb') as f:
        pickle.dump(best_k_model, f)
    
    print(f"Best feature selection model: k={best_k}, Validation ROC AUC = {best_k_score:.4f}")

# %%
# Find the top 3 models across all runs
print("\n========== Identifying Top 3 Models ==========")

top_models = []

top_models.append(("Best Hyperopt Model", best_model, best_val))

top_models.append((f"Feature Selection (k={best_k})", best_k_model, best_k_score))
sorted_trials = sorted(trials.trials, key=lambda t: t['result']['loss'])
if len(sorted_trials) > 1:  # Make sure we have at least 2 trials
    second_best_trial = sorted_trials[1]  # Get second best trial
    second_best_model = second_best_trial['result']['model']
    second_best_val = -second_best_trial['result']['loss']
    second_best_name = second_best_trial['result']['model_name']
    top_models.append((f"Second Best {second_best_name}", second_best_model, second_best_val))
else:
    third_model = Pipeline([
        ('preprocessor', create_preprocessing_pipeline(categorical_cols, numerical_cols)),
        ('classifier', LogisticRegression(random_state=42))
    ])
    third_model.fit(X_train, y_train)
    y_val_pred = third_model.predict(X_val)
    y_val_proba = third_model.predict_proba(X_val)[:, 1]
    third_val_roc_auc = roc_auc_score(y_val, y_val_proba)
    top_models.append(("Logistic Regression", third_model, third_val_roc_auc))

# Sort top models by performance
top_models.sort(key=lambda x: x[2], reverse=True)

# Print top 3 models
print("\nTop 3 models by validation ROC AUC:")
for i, (name, model, score) in enumerate(top_models):
    print(f"{i+1}. {name}: {score:.4f}")

# Select the best model for final eval
best_model_name, best_model, _ = top_models[0]
print(f"\nSelected best model: {best_model_name}")


# %%
with mlflow.start_run(run_name="FinalEvaluation"):
    mlflow.set_tag("step", "FinalEvaluation")
    mlflow.set_tag("model", best_model_name)
    
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("test_roc_auc", test_roc_auc)
    
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Test Set Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('images/best_model_test_confusion_matrix.png')
    mlflow.log_artifact('images/best_model_test_confusion_matrix.png')
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_test_pred)
    print(report)
    
    # Save the final model
    with open('models/final_best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Register the model in MLflow registry
    registered_model = mlflow.sklearn.log_model(
        best_model, 
        artifact_path="final_model",
        registered_model_name="heart_failure_prediction_model"
    )
    
    print(f"\nFinal test metrics:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"F1-score: {test_f1:.4f}")
    print(f"ROC-AUC: {test_roc_auc:.4f}")
    print(f"\nBest model registered as 'heart_failure_prediction_model'")