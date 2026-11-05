
import re, pandas as pd, numpy as np, matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore') 

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from pandas.plotting import parallel_coordinates

from itertools import combinations

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.metrics import RocCurveDisplay
#!pip install kds
# Import kds.
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier

# Analysis step
# ################### general visualisation #################################.
# Analysis step

"""
Pairwise scatter visualization for a set of numeric features.

Use an optional hue column to distinguish class or cluster membership.
"""
def plot_feature_pairs(data, reg_cols, color_col=None): 
    indices_combinations = list(combinations(range(0, len(reg_cols)), 2))
    plt.figure(figsize=(10, 10))
    for pair in enumerate(indices_combinations):
        plt.subplot(len(reg_cols)//2 + 1, 2, pair[0] + 1)
        if color_col:
            sns.scatterplot(data, x=reg_cols[pair[1][0]],
                            y=reg_cols[pair[1][1]], hue=color_col)
        else:
            sns.scatterplot(data, x=reg_cols[pair[1][0]],
                            y=reg_cols[pair[1][1]])
    plt.tight_layout()
    plt.show()


# Analysis step
# ############################# clustering ##################################.
# Analysis step

"""
Compare silhouette distributions over the configured clustering solutions.

The estimator and candidate cluster counts are passed unchanged.
"""
def plot_cluster_silhouettes(data, algorithm, cluster_range):        
    # Render the current analysis view.
    num_plots = len(cluster_range)
    n_rows = (num_plots + 2) // 3  # Render the current analysis view.
    fig, axs = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))    
    # Flatten the axes array for easier iteration.
    axs = axs.flatten()
    # Loop through the specified number of clusters.
    for i, n_clusters in enumerate(cluster_range):
        # Create an instance of the clustering algorithm with the specified number of clusters.
        model = algorithm.set_params(n_clusters=n_clusters)        
        # Keep the target categories consistent across splits.
        cluster_labels = model.fit_predict(data)        
        # Report the configured model or statistical measure.
        silhouette_avg = silhouette_score(data, cluster_labels)
        sample_silhouette_values = silhouette_samples(data, cluster_labels)        
        y_lower = 10
        axs[i].set_xlim([-0.1, 1])
        axs[i].set_ylim([0, len(data) + (n_clusters + 1) * 10])        
        # Render the current analysis view.
        for j in range(n_clusters):
            # Report the configured model or statistical measure.
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == j]
            ith_cluster_silhouette_values.sort()
            size_cluster_j = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j
            color = cm.nipy_spectral(float(j) / n_clusters)
            axs[i].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
            # Render the current analysis view.
            axs[i].text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))
            # Render the current analysis view.
            y_lower = y_upper + 10  # 10 for the space between clusters.
        axs[i].set_title(f"Silhouette plot for {n_clusters} clusters")
        axs[i].set_xlabel("Silhouette coefficient values")
        axs[i].set_ylabel("Cluster label")
        # Report the configured model or statistical measure.
        axs[i].axvline(x=silhouette_avg, color="red", linestyle="--")
        axs[i].set_yticks([])  # Keep the target categories consistent across splits.
        axs[i].set_xticks(np.arange(-0.1, 1.1, 0.2))
    # Render the current analysis view.
    for ax in axs[num_plots:]:
        fig.delaxes(ax)  
    # Adjust layout.
    plt.tight_layout()
    plt.show()

# Analysis step
# ############################# classification ##################################.
# Analysis step

# Compare actual and predicted class frequencies.
def display_confusion_heatmap(y_test, predictions):
  # Compare actual and predicted class frequencies.
  confusion_m = confusion_matrix(y_test, predictions)
  # Can also visualise it.
  labels = np.unique(y_test)
  confusion_frame = pd.DataFrame(confusion_m,
                            index=labels, columns=labels)
  plt.figure(figsize=(5, 5))
  sns.heatmap(data=confusion_frame, annot=True, square=True,
              cbar=False)
  plt.xlabel('Predicted Labels')
  plt.ylabel('Actual Labels')
  plt.show()

# Compare actual and predicted class frequencies.
def save_confusion_heatmap(y_test, predictions, image):
    # Compare actual and predicted class frequencies.
    confusion_m = confusion_matrix(y_test, predictions)
    
    # Can also visualize it.
    labels = np.unique(y_test)
    confusion_frame = pd.DataFrame(confusion_m, index=labels, columns=labels)
    
    plt.figure(figsize=(5, 5))
    sns.heatmap(data=confusion_frame, annot=True, square=True, cbar=False)
    
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    
    # Configure the chart layout.
    plt.savefig(f"img/{image}.png")
    
    # Render the current analysis view.
    plt.show()
    
"""
Report cross-validated accuracy, macro precision, macro recall and F1,
including variation across the configured number of folds.
"""
def summarize_stratified_cv(reg_data, labels, clf, cv_=5, print_dict=False):
  # # create the stratified splitter.
  skf = StratifiedKFold(n_splits=cv_, shuffle=True, random_state=43)
  accuracies = []
  precisions = []
  recalls = []
  fscores = []
  splits = skf.split(reg_data, labels)
  # # iterate as many times as the specified cv_.
  for train_index, test_index in splits:
    # Read or prepare the relevant input frame.
    X_train, X_test = reg_data.iloc[train_index], reg_data.iloc[test_index]
    # Keep the target categories consistent across splits.
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    # # train the classifier on the training portion.
    clf.fit(X_train, y_train)
    # Calculate the configured statistical comparison.
    predictions = clf.predict(X_test)
    # # retrieve performance using classification report but in the form of a dictionary.
    performance = classification_report(y_test, predictions, output_dict=True)
    # # optionally you can print it.
    if(print_dict):
      print(performance)
    # Report the configured model or statistical measure.
    accuracies.append(performance['accuracy'] * 100)
    precisions.append(performance['macro avg']['precision'] * 100)
    recalls.append(performance['macro avg']['recall'] * 100)
    fscores.append(performance['macro avg']['f1-score'] * 100)
  # Compute the requested summary statistic.
  print('Mean Accuracy is %0.2f +/- %0.2f'%(
    np.array(accuracies).mean(), np.array(accuracies).std()))
  print('Mean Precision is %0.2f +/- %0.2f'%(
      np.array(precisions).mean(), np.array(precisions).std()))
  print('Mean Recall is %0.2f +/- %0.2f'%(
      np.array(recalls).mean(), np.array(recalls).std()))
  print('Mean F-score is %0.2f +/- %0.2f'%(
      np.array(fscores).mean(), np.array(fscores).std()))
  # Calculate the configured statistical comparison.
  # Report the configured model or statistical measure.
  # Return clf.

def evaluate_classification_model(clf, X_test, y_test, print_report=True, print_dict=False):
    """
Summarize a fitted classifier on a separate feature and label set.

The existing predictions and metrics are calculated without modifying inputs.
"""
    # Ensure numpy arrays.
    if isinstance(X_test, pd.DataFrame):
        X_arr = X_test.values
    else:
        X_arr = X_test

    if isinstance(y_test, (pd.Series, pd.DataFrame)):
        y_arr = y_test.values.ravel()
    else:
        y_arr = y_test

    preds = clf.predict(X_arr)
    report = classification_report(y_arr, preds, output_dict=True)

    if print_report:
        print(classification_report(y_arr, preds))
    if print_dict:
        print(report)

    metrics = {
        'accuracy': [report['accuracy'] * 100],
        'precision': [report['macro avg']['precision'] * 100],
        'recall': [report['macro avg']['recall'] * 100],
        'f1_score': [report['macro avg']['f1-score'] * 100]
    }

    return metrics



def summarize_validation_metrics(X, y, clf, cv=5, print_dict=False):
    """
Summarize stratified validation statistics using the original scorer set.

The returned values and print format are preserved.
"""
    # Ensure we have numpy arrays for indexing.
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = X

    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_arr = y.values.ravel()
    else:
        y_arr = y

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=43)

    accs, precs, recs, f1s = [], [], [], []

    for train_idx, test_idx in skf.split(X_arr, y_arr):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)

        report = classification_report(y_te, preds, output_dict=True)
        if print_dict:
            print(report)

        accs.append(report['accuracy'] * 100)
        precs.append(report['macro avg']['precision'] * 100)
        recs.append(report['macro avg']['recall'] * 100)
        f1s.append(report['macro avg']['f1-score'] * 100)

    def mean_std(lst):
        a = np.array(lst)
        return a.mean(), a.std()

    m_acc, s_acc = mean_std(accs)
    m_prec, s_prec = mean_std(precs)
    m_rec, s_rec = mean_std(recs)
    m_f1, s_f1 = mean_std(f1s)

    print(f"Mean Accuracy:  {m_acc:.2f}% +/- {s_acc:.2f}")
    print(f"Mean Precision: {m_prec:.2f}% +/- {s_prec:.2f}")
    print(f"Mean Recall:    {m_rec:.2f}% +/- {s_rec:.2f}")
    print(f"Mean F1‑score:  {m_f1:.2f}% +/- {s_f1:.2f}")
    print("")
    return {
      'accuracy': accs,
      'precision': precs,
      'recall': recs,
      'f1_score': f1s
    }

"""
Construct a feature-level view of predicted classes and class probabilities.

The feature observations and associated labels are retained.
"""
def build_prediction_details(clf, X_test, y_test):
  confidences = clf.predict_proba(X_test)
  predictions = clf.predict(X_test)
  output = X_test.copy()
  output['true labels'] = y_test
  output['predictions'] = predictions
  for i, class_name in enumerate(clf.classes_):
    output[f'confidence_{class_name}'] = confidences[:, i]
  return output


"""
Compare micro-averaged multiclass ROC curves across the input estimators.

The original split, one-vs-rest fitting and AUC calculations remain intact.
"""
def plot_multiclass_roc_comparison(clfs_dict, X, y, test_size_=0.3, random_state_=43):
  # Display class discrimination across decision thresholds.
  classes = list(range(len(np.unique(y))))
  labels = label_binarize(LabelEncoder().fit_transform(y), classes=classes)
  n_classes = labels.shape[1]
  X_train, X_test, y_train, y_test = train_test_split(
                                X, labels,
                                test_size=test_size_,
                                stratify=labels,
                                random_state=random_state_)
  # Use the one-vs-rest classier to wrap around each classifier.
  y_scores = {}
  # Iterate though each classifier, wrap a one-vs-rest classier around each,.
  # Report the configured model or statistical measure.
  for name, clf in clfs_dict.items():
      clf = OneVsRestClassifier(clf)
      clf.fit(X_train, y_train)
      y_scores[name] = clf.predict_proba(X_test)  
  plt.figure(figsize=(7, 5))
  # Display class discrimination across decision thresholds.
  for name, y_score in y_scores.items(): 
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Display class discrimination across decision thresholds.
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])  
    plt.plot(fpr["micro"], tpr["micro"],
          label=f'{name}: micro-average ROC curve (area = {roc_auc["micro"]:0.2f})')
  # Add the diagonal line for random guess.
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curves for 3 classifiers on Iris Dataset')
  plt.legend(loc="best")
  plt.show()

# Remove fields that reveal the outcome directly.
# Display class discrimination across decision thresholds.
def evaluate_holdout_pipeline(clf, preparations, X_train, y_train, X_test, y_test, imblearn_=False):
  # Add preparation steps and classification step to the pipeline.
  pipeline = None
  if imblearn_:
      from imblearn.pipeline import make_pipeline
      pipeline = make_pipeline(preparations,clf)
  else:
      pipeline = Pipeline(steps=[('preprocessing', preparations), ('classification', clf)])
  pipeline.fit(X_train, y_train)
  predictions = pipeline.predict(X_test)
  return (accuracy_score(y_test, predictions),
          precision_score(y_test, predictions, average='macro'),
          recall_score(y_test, predictions, average='macro'),
          f1_score(y_test, predictions, average='macro'))

# Display class discrimination across decision thresholds.
# Within each fold it calls the above method to do split validation.
def evaluate_crossvalidated_pipeline(clf, preparations, X, y, imblearn_=False, folds=5):
  skf = StratifiedKFold(n_splits=folds)
  accuracies = []
  precisions = []
  recalls = []
  fscores = []
  for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    scores = evaluate_holdout_pipeline(clf, preparations, X_train, y_train, X_test, y_test, imblearn_)
    accuracies.append(scores[0])
    precisions.append(scores[1])
    recalls.append(scores[2])
    fscores.append(scores[3])
  # Compute the requested summary statistic.
  print('Mean Accuracy is %0.2f +/- %0.2f'%(
    np.array(accuracies).mean() * 100, np.array(accuracies).std() * 100))
  print('Mean Precision is %0.2f +/- %0.2f'%(
      np.array(precisions).mean() * 100, np.array(precisions).std() * 100))
  print('Mean Recall is %0.2f +/- %0.2f'%(
      np.array(recalls).mean() * 100, np.array(recalls).std() * 100))
  print('Mean F-score is %0.2f +/- %0.2f'%(
      np.array(fscores).mean() * 100, np.array(fscores).std() * 100))
    

# Analysis step
# ############################# regression ##################################.
# Analysis step

# Function to compute adjusted r2.
def compute_adjusted_r_squared(R2, N, p):
  return 1-((1-R2)*(N-1)/(N-p-1))

# Function to compute tolerance for each independent variable.
def compute_tolerance_scores(X):
    tolerance = {}
    for col in X.columns:
        if col == 'const':  
            tolerance[col] = ''
            continue
        # Regress the current variable on all others.
        other_cols = [c for c in X.columns if c != col]
        model = sm.OLS(X[col], X[other_cols]).fit()        
        # Calculate r2 and tolerance.
        r_squared = model.rsquared
        tolerance[col] = 1 - r_squared
    return pd.Series(tolerance)

# Summarize results over the established validation folds.
def summarize_regression_cv(regressor, X, y, lin=False, return_=False, cv_=5):
  scores = cross_validate(regressor, X, y,
                    scoring=['neg_mean_squared_error', 'explained_variance'],
                    return_estimator=True, cv=cv_)
  AVG_MSE = -scores['test_neg_mean_squared_error'].mean()
  MSE_STD = scores['test_neg_mean_squared_error'].std()
  AVG_RMSE = np.sqrt(AVG_MSE);  RMSE_STD = np.sqrt(MSE_STD)
  AVG_R2 = scores['test_explained_variance'].mean()
  R2_STD = scores['test_explained_variance'].std()
  print(f"Average cross-validated RMSE: {AVG_RMSE:.2f} +/- {RMSE_STD:.2f}")
  print(f"Average cross-validated R-squared: {AVG_R2:.2f} +/- {R2_STD:.2f}")
  if(lin): # # the coef and intercept are only available in linear regression.
    print('Coefficients across all folds....')
    for model in scores['estimator']:
      print("Coeficient: ", model.coef_, ", Intercept: ", model.intercept_)
  if(return_):
    return AVG_RMSE, AVG_R2

# Function to get only the relevant results from statsmodels ols.
import statsmodels.api as sm
def extract_ols_diagnostics(X, y, add_const=True, print_=True):
    if add_const:
        X = sm.add_constant(X, has_constant='add')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
    model = sm.OLS(y_train, X_train).fit()
    if print_:      
        preds = model.predict(X_test)
        print(f"------- Overall StatsModel OLS Metrics -------")
        print(f"R-squared: {model.rsquared:.2f}")
        print(f"Adjusted R-squared: {model.rsquared_adj:.2f} ")
        print(f"AIC: {model.aic:.2f}")
        print(f"RMSE: {root_mean_squared_error(y_test, preds):.2f}")
        print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    df = pd.DataFrame(columns=['coefficient', 'std error', 'std coefficient',
                               't-stat', 'p-value', '0.025', '0.975', 'tolerance'], 
                      index=X.columns)
    df['coefficient'] = model.params
    df['std error'] = model.bse
    df['std coefficient'] = df['coefficient'] * (np.std(X_train)/np.std(y_train))
    df['t-stat'] = model.tvalues # Can also use df['coefficient'] / df['std error'].
    df['p-value'] = model.pvalues
    df['0.025'] = model.conf_int()[0]
    df['0.975'] = model.conf_int()[1]    
    df['tolerance'] = compute_tolerance_scores(X_train).T[X.columns]
    return df.round(2)

# Summarize results over the established validation folds.
def cross_validate_ols(X, y, cv_=5, output_AIC=False):
  kf = KFold(n_splits=cv_, shuffle=True, random_state=43)
  MAEs = []; RMSEs = []; R2s = [];adj_R2s = []; AICs=[]
  X = sm.add_constant(X, has_constant='add')
  for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]  
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]          
    model = sm.OLS(y_train, X_train).fit()
    preds = model.predict(X_test)
    MAEs.append(mean_absolute_error(y_test, preds))
    RMSEs.append(root_mean_squared_error(y_test, preds))
    adj_R2s.append(model.rsquared_adj)
    R2s.append(model.rsquared)
    AICs.append(model.aic)
  print(f"Average Cross-validated R-squared: {np.mean(R2s):.2f} +/- {np.std(R2s):.2f}")
  print(f"Average Cross-validated Adjusted R-squared: {np.mean(adj_R2s):.2f} +/- {np.std(adj_R2s):.2f}")
  if(output_AIC):
    print(f"Average Cross-validated AIC: {np.mean(AICs):.2f} +/- {np.std(AICs):.2f}")
  print(f"Average Cross-validated MAE: {np.mean(MAEs):.2f} +/- {np.std(MAEs):.2f}")
  print(f"Average Cross-validated RMSE: {np.mean(RMSEs):.2f} +/- {np.std(RMSEs):.2f}")
  # Return the df with relevant resutls for model with split val.
  return extract_ols_diagnostics(X, y, add_const=False, print_=False)
  
# Read or prepare the relevant input frame.
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameTypeConverter(BaseEstimator, TransformerMixin):
    def __init__(self, dtype='category'):
        self.dtype = dtype
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Convert object columns to the specified dtype.
        X = X.copy()  # Read or prepare the relevant input frame.
        for col in X.columns:
            X[col] = X[col].astype(self.dtype)
        return X