import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

def remove_borders() -> None:
    """
    Removes borders from an existing graph.
    """
    for spine in plt.gca().spines.values():
        spine.set_visible(False)


def model_performance_review(predictions : pd.DataFrame, real_values : pd.DataFrame, probas: pd.DataFrame, model_name : str) -> None:
    """
    Running model performance analysis with respect to:
    1. Accuracy, Precision & Recall of predictions against real values/
    2. Graphing the ROC Curve between the True Positive Rate & The False Positive Rates
    3. Plotting The Confusion Matrix of provided model
    4. Plotting a tile-based representation of correctly/not correctly predicted outcomes of the give model
    """
    

    print(f"Results for {model_name} model:")
    print(f"Accuracy: {accuracy_score(predictions, real_values)}")
    print(f"Precision: {precision_score(predictions, real_values, zero_division = np.nan)}")
    print(f"Recall: {recall_score(predictions, real_values, zero_division = np.nan)}")

    _, ax = plt.subplots(1, 3, figsize = (20, 5))

    # ROC curve

    fpr, tpr, _ = roc_curve(real_values, probas)
    ax[1].plot(fpr, tpr, label='ROC Curve')
    ax[1].plot([0, 1], [0, 1], linestyle='--', color='red', label='No Skill')
    ax[1].fill_between(fpr, tpr, fpr, color='skyblue', alpha=0.3, label='Area Under ROC Curve gain (AUC)')
    ax[1].set_title(f'ROC Curve for model {model_name}')
    ax[1].legend(loc = 'upper left')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    remove_borders()

    # Confusion Matrix

    cm = confusion_matrix(real_values, predictions)

    sns.heatmap(cm, annot=True,  fmt='d', ax = ax[0])
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')
    ax[0].set_title(f'Confusion Matrix')
    
    # Prediction Results Graph
    
    prg = real_values.to_numpy() == predictions

    if len(prg) % 2 != 0:
        prg = np.append(prg, None)

    df = pd.DataFrame(prg.reshape(10, 20))

    color_map = {True: 'green', False: 'red', None:'white'}
    color_grid = df.map(lambda x: color_map[x]);

    # Plot the grid
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax[2].add_patch(plt.Rectangle((j, i), 1, 1, color=color_grid.iloc[i, j]))

    ax[2].set_xlim(0, df.shape[1])
    ax[2].set_ylim(0, df.shape[0])

    ax[2].set_title('Prediction Results')
    remove_borders()

    plt.show()


    

def shorten_param(param_name : str) -> str:
    """
    Removes Unnecessary titles from the columns of a dataset (particularly for hyperparameter tuning results of models)
    """

    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name

def cross_validation_setup(model, param_dist, training_data, variable_space, target_variable, iterations : int = 30) -> pd.DataFrame:
    
    """
    Performs Cross Validation for given model, for given number of iterations on a specific training dataset & parameter distributions.
    """

    hyperparameter_search = RandomizedSearchCV(
                                model,
                                param_distributions=param_dist,
                                n_iter=iterations,
                                cv=5,
                                verbose=1,
                                scoring='recall',
                                random_state=12
                            )

    hyperparameter_search.fit(training_data[variable_space], training_data[target_variable])
    
    print(f"Best Model Hyperparameters: {hyperparameter_search.best_params_}")
    print(f"Best Model Recall Score: {hyperparameter_search.best_score_}")

    cv_results = pd.DataFrame(hyperparameter_search.cv_results_).rename(shorten_param, axis=1)

    return hyperparameter_search.best_estimator_, cv_results
