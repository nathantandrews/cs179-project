from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import os
import zipfile
import urllib.request
import ssl
import warnings
import argparse
import time

import pandas as pd
import numpy as np
import pathlib as p
import matplotlib.pyplot as plt
import pyGMs as gm
import pyGMs.wmb as wmb

warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context

### Random seed for reproducibility
np.random.seed(42)

### Constants
DATASET_URL: str = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
MODEL_PATH: p.Path = p.Path("model")
LOG_PATH: p.Path = p.Path("logs")
LOG_PATH.mkdir(exist_ok=True)
DATA_PATH: p.Path = p.Path(MODEL_PATH / "data")
RATINGS_PATH: p.Path = p.Path(DATA_PATH / "ratings.csv")
MOVIES_PATH: p.Path = p.Path(DATA_PATH / "movies.csv")
RUN_HISTORY_PATH: p.Path = p.Path(LOG_PATH / "run_history.log")
LABEL_WIDTH = 40
VALUE_WIDTH = 20
WRITTEN_SUBHEADINGS: set[str] = set()

def initialize_parameters() -> tuple:
    """
    Initializes the parameters for testing the AI.

    Returns:
        a tuple containing all the parameters.
    """
    print("Initializing parameters...")
    parser = argparse.ArgumentParser(description="Train an Ising model on MovieLens data")
    
    parser.add_argument('--user-count', type=int, default=1000, help='Number of users to include')
    parser.add_argument('--movie-count', type=int, default=250, help='Number of movies to include')
    parser.add_argument('--c-value', type=float, default=0.045, help='Regularization constant')
    parser.add_argument('--min-rating', type=float, default=4.0, help='Minimum rating to classify movie as liked')

    args = parser.parse_args()
    print("Parameter Initialization Done.")
    return int(args.user_count), int(args.movie_count), float(args.c_value), float(args.min_rating)

def download_data(dataset_url: str, model_path: p.Path, data_path: p.Path) -> None:
    """
    Downloads the dataset and places ratings.csv and movies.csv in a custom folder.

    Args:
        dataset_url (str): The url for the dataset
    """
    zip_file = dataset_url.split("/")[-1]
    zip_path: p.Path = p.Path(model_path / zip_file)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        print("Downloading MovieLens dataset...")
        urllib.request.urlretrieve(dataset_url, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                filename = os.path.basename(member)
                print(filename)
                if not filename or filename not in ["movies.csv", "ratings.csv"]:
                    continue
                with zip_ref.open(member) as source:
                    target_path = data_path / filename
                    with open(target_path, "wb") as target:
                        target.write(source.read())
        print("Removing zip...")
        os.remove(zip_path)
        print("Data Download Done.")
    else:
        print("Dataset already exists.")

def log_run_header(user_count: int, movie_count: int, 
                   c_value: float, file_path: p.Path = RUN_HISTORY_PATH) -> None:
    """
    Logs the header for each run.

    Args:
        user_count (int): Number of users to include
        movie_count (int): Number of movies to include
        c_value (float): Regularization constant
        file_path (p.Path, optional): path to log file. Defaults to RUN_HISTORY_PATH.
    """
    timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, "a") as f:
        f.write("\n" + "=" * (LABEL_WIDTH + VALUE_WIDTH) + "\n")
        f.write(f"{'u_count:':<{LABEL_WIDTH}}{user_count:>{VALUE_WIDTH}}\n")
        f.write(f"{'m_count:':<{LABEL_WIDTH}}{movie_count:>{VALUE_WIDTH}}\n")
        f.write(f"{'c_value:':<{LABEL_WIDTH}}{c_value:>{VALUE_WIDTH}.4f}\n")
        f.write(f"{'Run at:':<{LABEL_WIDTH}}{timestamp:>{VALUE_WIDTH}}\n")
        f.write("-" * (LABEL_WIDTH + VALUE_WIDTH) + "\n")

def log_run_data(model_type: str, output_dict: dict, file_path: p.Path = RUN_HISTORY_PATH) -> None:
    """
    Logs a run of the Ising Model

    Args:
        model_type (str): "Independent" or "Ising" Model.
        output_dict (dict): dictionary containing the outputs to log.
        file_path (p.Path, optional): path to log file. Defaults to RUN_HISTORY_PATH.
    """
    with open(file_path, "a") as f:
        subheading = {
                    "independent": "Independent Model", 
                    "ising": "Ising Model"
                 }.get(model_type.lower(), "Unknown Model")
        if model_type not in WRITTEN_SUBHEADINGS:
            f.write(f"{subheading}\n")
            WRITTEN_SUBHEADINGS.add(model_type)
        for key, value in output_dict.items():
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            f.write(f"{key:<{LABEL_WIDTH}}{formatted_value:>{VALUE_WIDTH}}\n")

def filter_data(user_count: int, movie_count: int, minimum_rating: float):
    """
    Filters the ratings data to get a 2D likes array

    Args:
        user_count (int): Number of users to include
        movie_count (int): Number of movies to include
        minimum_rating (float): Minimum rating to classify movie as liked

    Returns:
        2D likes array
    """
    print("Filtering the data...")
    ratings = pd.read_csv(RATINGS_PATH)
    top_users = ratings['userId'].value_counts().head(user_count).index
    top_movies = ratings['movieId'].value_counts().head(movie_count).index
    filtered = ratings[(ratings['userId'].isin(top_users)) & (ratings['movieId'].isin(top_movies))]
    pivot = filtered.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    # Binary matrix: 1 if liked (rating >= minimum_rating), else 0
    X = (pivot >= minimum_rating).astype(int).to_numpy()
    print("Filtering Done.")
    return X, pivot

def get_short_labels(pivot) -> dict:
    """
    Generates a shortened version of the labels used for movies

    Args:
        pivot (Dataframe): A Dataframe containing the filtered movies

    Returns:
        dict: A dictonary that contains the mappings from movie_id to short_label
    """
    print("Extracting Short Labels...")
    movies = pd.read_csv(MOVIES_PATH)
    id_to_title = dict(zip(movies['movieId'], movies['title']))
    short_labels = {i: id_to_title[mid] for i, mid in enumerate(pivot.columns)}
    print("Extraction Done.")
    return short_labels

def conditional(factor, i, x):
    print("Computing Conditional...")
    result = factor.t[tuple(x[v] if v != i else slice(v.states) for v in factor.vars)]
    print("Conditional Done.")
    return result

def pseudolikelihood(model, X):
    print("Computing Pseudo-Likelihood...")
    LL = np.zeros(X.shape)
    for i in range(X.shape[1]):  # for each variable (movie)
        flist = model.factorsWith(i, copy=False)
        for j in range(X.shape[0]):  # for each data point (user)
            pXi = 1.
            for f in flist:
                pXi *= conditional(f, i, X[j])
            LL[j, i] = np.log(pXi[X[j, i]] / pXi.sum()) # type: ignore
    print("Pseudo-Likelihood Done.")
    return LL.sum(1)

def impute_missing(model, Xobs):
    print("Imputing Missing...")
    m,n = Xobs.shape
    Xhat = np.copy(Xobs)
    for j in range(m):
        x_obs = {i:Xobs[j,i] for i in range(n) if Xobs[j,i] >= 0}
        x_unobs = [i for i in range(n) if Xobs[j,i] < 0]
        cond = gm.GraphModel([f.condition(x_obs) for f in model.factorsWithAny(x_unobs)])
        for x in cond.X:
            if x.states == 0:
                x.states = 1  # fix a bug in GraphModel behavior for missing vars...
        jt = wmb.JTree(cond, weights=1e-6) # 0: for maximization
        x_hat = jt.argmax()
        for i in x_unobs: 
            Xhat[j,i] = x_hat[i]
    print("Imputing Done.")
    return Xhat

def get_likelihood(model_type: str, model: gm.GraphModel):
    """
    Gets the likelihood for the Independent model 
    or pseudo-likelihood for the Ising model

    Args:
        model_type (str): type of model, Independent or Ising
        model (gm.GraphModel): model to get likelihood from
    """
    print("Computing the Training/Testing Likelihood/Pseudo-Likelihood...")
    if model_type == "independent":
        ind_train_ll = np.mean([model.logValue(x) for x in Xtr])
        ind_test_ll = np.mean([model.logValue(x) for x in Xte])
        log_run_data("independent", {"- Log-Likelihood (Train)" : float(ind_train_ll), 
                                    "- Log-Likelihood (Test)" : float(ind_test_ll)})
        print("Likelihood/Pseudo-Likelihood Done.")
        return ind_train_ll, ind_test_ll
    elif model_type == "ising":
        pseudolikelihood_tr: float = float(pseudolikelihood(model, Xtr).mean())
        pseudolikelihood_te: float = float(pseudolikelihood(model, Xte).mean())
        log_run_data("ising", {"- Pseudo-Likelihood (Train)" : pseudolikelihood_tr, 
                                "- Pseudo-Likelihood (Test)" : pseudolikelihood_te})
        print("Likelihood/Pseudo-Likelihood Done.")
        return pseudolikelihood_tr, pseudolikelihood_te
    else:
        raise TypeError(f"Unknown model type: {model_type}")
    
def get_average_connectivity(nbrs: list) -> tuple:
    """_summary_

    Args:
        nbrs (list): _description_

    Returns:
        tuple: _description_
    """
    print("Computing the Average Connectivity...")
    average_connectivity = np.mean([len(nn) for nn in nbrs])
    std_dev = np.std([len(nn) for nn in nbrs])
    log_run_data("ising", {"- Average Connectivity" : f"{average_connectivity:.4f} +/- {std_dev:.4f}"})
    print("Average Connectivity Done.")
    return average_connectivity, std_dev

def independent_model(Xtr, movie_count: int) -> gm.GraphModel:
    """
    Creates an independent model on the training data.
    
    Args:
        Xtr: Training data.
    Returns:
        model0 (gm.GraphModel): The independent model.
    """
    pXi = np.mean(Xtr, axis=0)
    model0 = gm.GraphModel([gm.Factor([gm.Var(i, 2)], 
                                      [1 - pXi[i], pXi[i]]) for i in range(movie_count)]) # type: ignore
    return model0

def ising_model(Xtr, movie_count: int, c_value: float) -> tuple[gm.GraphModel, list]:
    nbrs, th_ij, th_i = [None] * movie_count, [None] * movie_count, np.zeros((movie_count,))
    Xtmp = np.copy(Xtr)
    for i in range(movie_count):
        Xtmp[:, i] = 0.
        lr = LogisticRegression(penalty='l1', C=c_value, solver='liblinear').fit(Xtmp, Xtr[:, i])
        nbrs[i] = np.where(np.abs(lr.coef_) > 1e-6)[1] # type: ignore
        th_ij[i] = lr.coef_[0, nbrs[i]] / 2. # type: ignore
        th_i[i] = lr.intercept_ / 2.
        Xtmp[:, i] = Xtr[:, i]
    factors = [gm.Factor(gm.Var(i, 2), [-t, t]).exp() for i, t in enumerate(th_i)] # type: ignore
    for i in range(movie_count):
        for j, n in enumerate(nbrs[i]): # type: ignore
            scope = [gm.Var(i, 2), gm.Var(int(n), 2)]
            t = th_ij[i][j] # type: ignore
            factors.append(gm.Factor(scope, [[t, -t], [-t, t]]).exp()) # type: ignore
    model1 = gm.GraphModel(factors)
    model1.makeMinimal()
    return model1, nbrs

def get_error_rate(model: gm.GraphModel) -> float:
    print("Computing Error Rate...")
    Xte_missing = np.copy(Xte)
    missing_proportion = 0.2
    mask = np.random.rand(*Xte.shape) < missing_proportion
    Xte_missing[mask] = -1
    Xte_hat = impute_missing(model, Xte_missing)
    error_rate = np.mean(Xte_hat[mask] != Xte[mask]) * 100
    log_run_data("ising", {"- Error Rate" : f"{error_rate:.4f}%"})
    print("Error Rate Done.")
    return error_rate

if __name__ == "__main__":
    user_count, movie_count, c_value, minimum_rating = initialize_parameters()
    log_run_header(user_count, movie_count, c_value)
    download_data(DATASET_URL, MODEL_PATH, DATA_PATH)
    X, pivot = filter_data(user_count, movie_count, minimum_rating)
    # short_labels = get_short_labels(pivot)
    Xtr, Xte = train_test_split(X, test_size=0.2, random_state=42)
    ind_model = independent_model(Xtr, movie_count)
    isi_model, nbrs = ising_model(Xtr, movie_count, c_value)








