import os
import numpy as np
from flask import Flask, render_template, request
import pickle
import random

# Tambahkan kelas SimpleRandomForestRegressor

class SimpleRandomForestRegressor:
    def __init__(self, n_estimators=29, max_depth=None, random_state=1234):
        # Initialize number of trees, maximum depth (default is None), and random seed
        random.seed(random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        # Fit the random forest to the training data
        self.trees = []
        for _ in range(self.n_estimators):
            indices = [random.randint(0, len(X) - 1) for _ in range(len(X))]
            X_sample = [X[i] for i in indices]
            y_sample = [y[i] for i in indices]
            tree = self._build_tree(X_sample, y_sample, depth=0)
            self.trees.append(tree)

    def _build_tree(self, X, y, depth):
        # Build a single decision tree
        if not y:
            return 0
        # Stop splitting if max_depth is None (allow unlimited depth) or if stopping criteria are met
        if self.max_depth is not None and depth >= self.max_depth or len(set(y)) == 1:
            return sum(y) / len(y)
        best_split = self._best_split(X, y)
        if best_split is None:
            return sum(y) / len(y)
        left_indices = [i for i in range(len(X)) if float(X[i][best_split[0]]) < float(best_split[1])]
        right_indices = [i for i in range(len(X)) if float(X[i][best_split[0]]) >= float(best_split[1])]
        left_tree = self._build_tree([X[i] for i in left_indices], [y[i] for i in left_indices], depth + 1)
        right_tree = self._build_tree([X[i] for i in right_indices], [y[i] for i in right_indices], depth + 1)
        return (best_split[0], best_split[1], left_tree, right_tree)

    def _best_split(self, X, y):
        # Find the best feature and value to split on
        best_split = None
        best_mse = float("inf")
        for feature_idx in range(len(X[0])):
            values = set(row[feature_idx] for row in X)
            for val in values:
                left_y = [y[i] for i in range(len(y)) if float(X[i][feature_idx]) < float(val)]
                right_y = [y[i] for i in range(len(y)) if float(X[i][feature_idx]) >= float(val)]
                if not left_y or not right_y:
                    continue
                mse = (self._mse(left_y) * len(left_y) + self._mse(right_y) * len(right_y)) / len(y)
                if mse < best_mse:
                    best_mse = mse
                    best_split = (feature_idx, val)
        return best_split

    def _mse(self, y):
        # Calculate mean squared error
        if not y:
            return 0
        mean_y = sum(y) / len(y)
        return sum((yi - mean_y) ** 2 for yi in y) / len(y)

    def predict(self, X):
        # Predict using the random forest
        predictions = []
        for x in X:
            tree_predictions = [self._predict_tree(x, tree) for tree in self.trees]
            predictions.append(sum(tree_predictions) / self.n_estimators)
        return predictions

    def _predict_tree(self, x, tree):
        # Predict using a single decision tree
        if not isinstance(tree, tuple):
            return tree
        feature_idx, val, left_tree, right_tree = tree
        if float(x[feature_idx]) < float(val):
            return self._predict_tree(x, left_tree)
        else:
            return self._predict_tree(x, right_tree)

app = Flask(__name__, template_folder='templates')

@app.route('/')
def student():
    return render_template("index.html")

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 8)
    try:
        # Periksa apakah file model tersedia
        if not os.path.exists("model.pkl"):
            # Jika file model tidak ada, kembalikan pesan error
            return "Error: File model 'model.pkl' tidak ditemukan."

        # Buka dan muat model
        with open("model.pkl", "rb") as model_file:
            load_model = pickle.load(model_file)
        
        # Periksa apakah model memiliki metode 'predict'
        if not hasattr(load_model, 'predict'):
            return "Error: Model tidak memiliki metode 'predict'."

        # Prediksi
        result = load_model.predict(to_predict)
        return round(result[0], 2)

    except (pickle.UnpicklingError, EOFError) as e:
        return f"Error loading the model: File model mungkin rusak atau tidak lengkap. Detail error: {e}"
    except Exception as e:
        return f"Terjadi error lain saat memuat model: {e}"

@app.route('/', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        try:
            to_predict_list = request.form.to_dict()

            if any(value.strip() == '' for value in to_predict_list.values()):
                return render_template("index.html", result_text="Error: Semua input harus diisi.")
            
            to_predict_list = list(to_predict_list.values())
            try:
                to_predict_list = list(map(float, to_predict_list))
            except ValueError:
                return render_template("index.html", result_text="Error: Pastikan semua input adalah angka.")
            
            result = ValuePredictor(to_predict_list)
            if isinstance(result, str):
                return render_template("index.html", result_text=result)
            
            return render_template("index.html", result_text=f'Prediksi Hasil Panen Anda: {result} kg')
        
        except Exception as e:
            return render_template("index.html", result_text=f"Terjadi error: {e}")
    
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
