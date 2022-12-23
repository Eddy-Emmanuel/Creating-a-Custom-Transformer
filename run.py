import pandas as pd
import Transformer.classes as tc
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

trainer = tc.kfold_validation(n_folds=5)
stratified_model = tc.stratified_kfold_validation(n_folds=5)

data, target = make_classification(n_samples=1000, n_features=10, random_state=42)

data, target = pd.DataFrame(data), pd.Series(target)


# Splitting and training dataset using KFold
print("KFold spit")
trainer.fit(data= data, target=target, model= XGBClassifier , problem_type="classification")
print(trainer.model_accuracy_score())
print(trainer.model_f1_score(), end="\n\n")
print("Final prediction : \n\n", trainer.predict(data)[:10])

# Splitting and training dataset using StratifiedKFold
print("KFold spit")
stratified_model.fit(data= data, target=target, model= XGBClassifier , problem_type="classification")
print(stratified_model.model_accuracy_score())
print(stratified_model.model_f1_score(), end = "\n\n")
print("Final prediction: \n\n", stratified_model.predict(data)[:10])