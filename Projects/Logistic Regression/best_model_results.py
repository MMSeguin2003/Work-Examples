import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from evaluate import evaluate

train_data = pd.read_csv('Train.csv')
test_data = pd.read_csv('Test.csv')


## Rap Best Results ##
best_rap_params = {'penalty': 'l1', 'solver': 'liblinear', 'C': 1000000.0} 
best_rap_features = ['Explicit', 'Playtime', 'Energy', 'Speechiness', 'Danceability', 'Mode', 'Age', 'TimeSignature_1', 'TimeSignature_3', 'TimeSignature_5']
X_train, y_train = train_data[best_rap_features], train_data['in_Rap']
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

rap_model = LogisticRegression(**best_rap_params)
rap_model.fit(X_train_scaled, y_train)

X_test, y_test = test_data[best_rap_features], test_data['in_Rap'].to_numpy()
# Apply same transformation to validation set
X_test_scaled = scaler.transform(X_test)

preds = rap_model.predict(X_test_scaled)
probs = rap_model.predict_proba(X_test_scaled)[:, 1]
coefs = rap_model.coef_[0]
preds_df = pd.DataFrame({"Rap Predictions": preds})
probs_df = pd.DataFrame({"Rap Probabilities": probs})
coefs_df = pd.DataFrame({"Feature": best_rap_features, 
                         "Coefficient": coefs, 
                         "Abs": abs(coefs)}).sort_values(by="Abs", ascending=False)[["Feature", "Coefficient"]]
preds_df.to_csv('RapPredictions.csv', index=False)
probs_df.to_csv('RapProbabilities.csv', index=False)
coefs_df.to_csv('RapCoefficients.csv', index=False)
num_correct, total, rec, rec_total, prec, prec_total, f1 = evaluate(preds, y_test, return_all=True)
acc = round(100*num_correct/total, 2)
recall = round(100*rec/rec_total, 2)
precision = round(100*prec/prec_total, 2)
data = {
    'Accuracy': [f'{num_correct}/{total} ({acc}%)'],
    'Predicted in Playlist / In playlist': [f'{rec}/{rec_total} ({recall}%)'],
    'In playlist / Predicted in Playlist': [f'{prec}/{prec_total} ({precision}%)'],
    'F1': [round(f1,4)]
}
metrics_df = pd.DataFrame.from_dict(data, orient='index')
metrics_df.to_csv('RapMetrics.csv', header=False)

## Bliss Best Results ##
best_bliss_params = {'penalty': 'l1', 'solver': 'liblinear', 'C': 100}
best_bliss_features = ['Explicit', 'Loudness', 'Popularity', 'Energy', 'Positiveness', 'Speechiness', 'Liveliness', 'Instrumentalness', 'Danceability', 'Key', 'TimeSignature_1', 'TimeSignature_3', 'TimeSignature_5']
X_train, y_train = train_data[best_bliss_features], train_data['in_Bliss']
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

bliss_model = LogisticRegression(**best_bliss_params)
bliss_model.fit(X_train_scaled, y_train)

X_test, y_test = test_data[best_bliss_features], test_data['in_Bliss'].to_numpy()
# Apply same transformation to validation set
X_test_scaled = scaler.transform(X_test)

preds = bliss_model.predict(X_test_scaled)
probs = bliss_model.predict_proba(X_test_scaled)[:, 1]
coefs = bliss_model.coef_[0]
preds_df = pd.DataFrame({"Bliss Predictions": preds})
probs_df = pd.DataFrame({"Bliss Probabilities": probs})
coefs_df = pd.DataFrame({"Feature": best_bliss_features, 
                         "Coefficient": coefs,
                         "Abs": abs(coefs)}).sort_values(by="Abs", ascending=False)[["Feature", "Coefficient"]]
preds_df.to_csv('BlissPredictions.csv', index=False)
probs_df.to_csv('BlissProbabilities.csv', index=False)
coefs_df.to_csv('BlissCoefficients.csv', index=False)
num_correct, total, rec, rec_total, prec, prec_total, f1 = evaluate(preds, y_test, return_all=True)
acc = round(100*num_correct/total, 2)
recall = round(100*rec/rec_total, 2)
precision = round(100*prec/prec_total, 2)
data = {
    'Accuracy': [f'{num_correct}/{total} ({acc}%)'],
    'Predicted in Playlist / In playlist': [f'{rec}/{rec_total} ({recall}%)'],
    'In playlist / Predicted in Playlist': [f'{prec}/{prec_total} ({precision}%)'],
    'F1': [round(f1,4)]
}
metrics_df = pd.DataFrame.from_dict(data, orient='index')
metrics_df.to_csv('BlissMetrics.csv', header=False)

## Blend Best Results ##
best_blend_params = {'penalty': 'l1', 'solver': 'liblinear', 'C': 10}
best_blend_features = ['Explicit', 'Popularity', 'Energy', 'Positiveness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Tempo', 'Key']
X_train, y_train = train_data[best_blend_features], train_data['in_Blend']
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

blend_model = LogisticRegression(**best_blend_params)
blend_model.fit(X_train_scaled, y_train)

X_test, y_test = test_data[best_blend_features], test_data['in_Blend'].to_numpy()
# Apply same transformation to validation set
X_test_scaled = scaler.transform(X_test)

preds = blend_model.predict(X_test_scaled)
probs = blend_model.predict_proba(X_test_scaled)[:, 1]
coefs = blend_model.coef_[0]
preds_df = pd.DataFrame({"Blend Predictions": preds})
probs_df = pd.DataFrame({"Blend Probabilities": probs})
coefs_df = pd.DataFrame({"Feature": best_blend_features, 
                         "Coefficient": coefs,
                         "Abs": abs(coefs)}).sort_values(by="Abs", ascending=False)[["Feature", "Coefficient"]]
preds_df.to_csv('BlendPredictions.csv', index=False)
probs_df.to_csv('BlendProbabilities.csv', index=False)
coefs_df.to_csv('BlendCoefficients.csv', index=False)
num_correct, total, rec, rec_total, prec, prec_total, f1 = evaluate(preds, y_test, return_all=True)
acc = round(100*num_correct/total, 2)
recall = round(100*rec/rec_total, 2)
precision = round(100*prec/prec_total, 2)
data = {
    'Accuracy': [f'{num_correct}/{total} ({acc}%)'],
    'Predicted in Playlist / In playlist': [f'{rec}/{rec_total} ({recall}%)'],
    'In playlist / Predicted in Playlist': [f'{prec}/{prec_total} ({precision}%)'],
    'F1': [round(f1,4)]
}
metrics_df = pd.DataFrame.from_dict(data, orient='index')
metrics_df.to_csv('BlendMetrics.csv', header=False)