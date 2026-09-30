import numpy as np
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class MRF_Graph:
    def __init__(self, hidden_vars, observed_vars):
        self.hidden_vars = hidden_vars
        self.observed_vars = observed_vars
        self.all_vars = hidden_vars + observed_vars
        self.unary_weights = {}
        self.pairwise_weights = {}
        self.graph = defaultdict(list)

    def add_edge(self, var1, var2):
        if var2 not in self.graph[var1]:
            self.graph[var1].append(var2)
        if var1 not in self.graph[var2]:
            self.graph[var2].append(var1)

    def init_weights_random(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        for h in self.hidden_vars:
            self.unary_weights[h] = {
                0: np.random.randn() * 0.1,
                1: np.random.randn() * 0.1,
            }
        edges_done = set()
        for var1 in self.graph:
            for var2 in self.graph[var1]:
                edge_key = tuple(sorted([var1, var2]))
                if edge_key not in edges_done:
                    edges_done.add(edge_key)
                    self.pairwise_weights[edge_key] = {
                        (0, 0): np.random.randn() * 0.1,
                        (0, 1): np.random.randn() * 0.1,
                        (1, 0): np.random.randn() * 0.1,
                        (1, 1): np.random.randn() * 0.1,
                    }

    def init_weights_expert(self):
        for h in self.hidden_vars:
            self.unary_weights[h] = {0: 0.5, 1: -0.5}
        edges_done = set()
        for var1 in self.graph:
            for var2 in self.graph[var1]:
                edge_key = tuple(sorted([var1, var2]))
                if edge_key not in edges_done:
                    edges_done.add(edge_key)
                    if (var1 in self.hidden_vars and var2 in self.observed_vars) or (
                        var2 in self.hidden_vars and var1 in self.observed_vars
                    ):
                        self.pairwise_weights[edge_key] = {
                            (0, 0): 1.0,
                            (0, 1): -1.0,
                            (1, 0): -1.0,
                            (1, 1): 1.5,
                        }
                    else:
                        self.pairwise_weights[edge_key] = {
                            (0, 0): 1.0,
                            (0, 1): -0.5,
                            (1, 0): -0.5,
                            (1, 1): 0.5,
                        }

    def save_weights(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "unary_weights": self.unary_weights,
                    "pairwise_weights": self.pairwise_weights,
                },
                f,
            )
        print(f"Веса сохранены в {filepath}")

    def load_weights(self, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.unary_weights = data["unary_weights"]
            self.pairwise_weights = data["pairwise_weights"]
        print(f"Веса загружены из {filepath}")

    def compute_energy(self, h_values, o_values):
        energy = 0
        for h_name, h_val in h_values.items():
            if h_name in self.unary_weights:
                energy += self.unary_weights[h_name][h_val]
        for (var1, var2), weights in self.pairwise_weights.items():
            val1 = h_values[var1] if var1 in h_values else o_values[var1]
            val2 = h_values[var2] if var2 in h_values else o_values[var2]
            energy += weights[(val1, val2)]
        return energy

    def compute_probability(self, h_values, o_values):
        energy = self.compute_energy(h_values, o_values)
        return np.exp(-energy)

    def compute_conditional_probability_h(self, h_name, h_val, other_h, o_values):
        """Вычисляет условную вероятность P(H_i = h_val | все остальные)"""
        h_copy = other_h.copy()
        h_copy[h_name] = h_val
        energy_h1 = self.compute_energy(h_copy, o_values)
        h_copy[h_name] = 1 - h_val
        energy_h0 = self.compute_energy(h_copy, o_values)
        prob = np.exp(-energy_h1) / (np.exp(-energy_h1) + np.exp(-energy_h0) + 1e-10)
        return prob

    def gibbs_sample(self, n_burnin=100):
        """Семплирование одной конфигурации из распределения модели методом Гиббса"""
        h_current = {h: np.random.choice([0, 1]) for h in self.hidden_vars}
        o_current = {o: np.random.choice([0, 1]) for o in self.observed_vars}
        for _ in range(n_burnin):
            for h_name in self.hidden_vars:
                prob = self.compute_conditional_probability_h(
                    h_name, 1, h_current, o_current
                )
                h_current[h_name] = 1 if np.random.random() < prob else 0
            for o_name in self.observed_vars:
                prob = self.compute_conditional_probability_o(
                    o_name, 1, o_current, h_current
                )
                o_current[o_name] = 1 if np.random.random() < prob else 0
        return h_current.copy(), o_current.copy()

    def compute_conditional_probability_o(self, o_name, o_val, other_o, h_values):
        """Вычисляет условную вероятность P(O_i = o_val | все остальные)"""
        o_copy = other_o.copy()
        o_copy[o_name] = o_val
        energy_o1 = self.compute_energy(h_values, o_copy)
        o_copy[o_name] = 1 - o_val
        energy_o0 = self.compute_energy(h_values, o_copy)
        prob = np.exp(-energy_o1) / (np.exp(-energy_o1) + np.exp(-energy_o0) + 1e-10)
        return prob

    def compute_model_expectations(self):
        h_names = self.hidden_vars
        n_hidden = len(h_names)
        hidden_configs = []
        for bits in itertools.product([0, 1], repeat=n_hidden):
            config = {h_names[i]: bits[i] for i in range(n_hidden)}
            hidden_configs.append(config)
        o_names = self.observed_vars
        n_obs = len(o_names)
        observed_configs = []
        for bits in itertools.product([0, 1], repeat=n_obs):
            config = {o_names[i]: bits[i] for i in range(n_obs)}
            observed_configs.append(config)
        Z = 0
        for hc in hidden_configs:
            for oc in observed_configs:
                Z += self.compute_probability(hc, oc)
        h_expectations = {h: {0: 0.0, 1: 0.0} for h in self.hidden_vars}
        pairwise_expectations = defaultdict(float)
        for hc in hidden_configs:
            for oc in observed_configs:
                prob = self.compute_probability(hc, oc) / Z
                for h_name, h_val in hc.items():
                    h_expectations[h_name][h_val] += prob
                for var1, var2 in self.pairwise_weights.keys():
                    val1 = hc[var1] if var1 in hc else oc[var1]
                    val2 = hc[var2] if var2 in hc else oc[var2]
                    key = (tuple(sorted([var1, var2])), (val1, val2))
                    pairwise_expectations[key] += prob
        return h_expectations, pairwise_expectations, Z

class EM_Algorithm:
    def __init__(self, mrf_graph, learning_rate=0.1, epsilon=1e-10):
        self.mrf = mrf_graph
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def enumerate_hidden_configs(self):
        h_names = self.mrf.hidden_vars
        n_hidden = len(h_names)
        configs = []
        for bits in itertools.product([0, 1], repeat=n_hidden):
            config = {h_names[i]: bits[i] for i in range(n_hidden)}
            configs.append(config)
        return configs

    def e_step_single_observation(self, observed_values):
        hidden_configs = self.enumerate_hidden_configs()
        unnormalized_probs = []
        for h_config in hidden_configs:
            prob = self.mrf.compute_probability(h_config, observed_values)
            unnormalized_probs.append(prob)
        total = sum(unnormalized_probs) + self.epsilon
        posterior_probs = [p / total for p in unnormalized_probs]
        h_stats = {h: {0: 0.0, 1: 0.0} for h in self.mrf.hidden_vars}
        pairwise_stats = defaultdict(float)
        for h_config, post_prob in zip(hidden_configs, posterior_probs):
            for h_name, h_val in h_config.items():
                h_stats[h_name][h_val] += post_prob
            for var1, var2 in self.mrf.pairwise_weights.keys():
                val1 = (
                    h_config[var1]
                    if var1 in self.mrf.hidden_vars
                    else observed_values[var1]
                )
                val2 = (
                    h_config[var2]
                    if var2 in self.mrf.hidden_vars
                    else observed_values[var2]
                )
                key = (tuple(sorted([var1, var2])), (val1, val2))
                pairwise_stats[key] += post_prob
        return h_stats, pairwise_stats

    def e_step(self, observations):
        n_samples = len(observations)
        total_h_stats = {h: {0: 0.0, 1: 0.0} for h in self.mrf.hidden_vars}
        total_pairwise_stats = defaultdict(float)
        for obs in observations:
            h_stats, pw_stats = self.e_step_single_observation(obs)
            for h_name, stats in h_stats.items():
                total_h_stats[h_name][0] += stats[0]
                total_h_stats[h_name][1] += stats[1]
            for key, val in pw_stats.items():
                total_pairwise_stats[key] += val
        for h_name in total_h_stats:
            total_h_stats[h_name][0] /= n_samples
            total_h_stats[h_name][1] /= n_samples
        for key in total_pairwise_stats:
            total_pairwise_stats[key] /= n_samples
        return total_h_stats, total_pairwise_stats

    def m_step(self, data_expectations, model_expectations):
        h_data, pw_data = data_expectations
        h_model, pw_model, Z = model_expectations
        for h_name in self.mrf.hidden_vars:
            for val in [0, 1]:
                grad = h_data[h_name][val] - h_model[h_name][val]
                self.mrf.unary_weights[h_name][val] += self.learning_rate * grad
        for edge_key in self.mrf.pairwise_weights.keys():
            for comb in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                key = (edge_key, comb)
                grad = pw_data.get(key, 0) - pw_model.get(key, 0)
                self.mrf.pairwise_weights[edge_key][comb] += self.learning_rate * grad
        return self.mrf

    def fit(self, observations, n_iterations=20, verbose=True):
        log_likelihoods = []
        for iteration in range(n_iterations):
            data_expectations = self.e_step(observations)
            model_expectations = self.mrf.compute_model_expectations()
            self.m_step(data_expectations, model_expectations)
            log_lik = self.compute_log_likelihood(observations)
            log_likelihoods.append(log_lik)
            if verbose:
                print(
                    f"Iteration {iteration+1}/{n_iterations}, Log-Likelihood: {log_lik:.4f}"
                )
            if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < 1e-6:
                if verbose:
                    print(f"Сходимость достигнута на итерации {iteration+1}")
                break
        return log_likelihoods

    def compute_log_likelihood(self, observations):
        log_lik = 0
        for obs in observations:
            hidden_configs = self.enumerate_hidden_configs()
            total_prob = 0
            for h_config in hidden_configs:
                total_prob += self.mrf.compute_probability(h_config, obs)
            log_lik += np.log(total_prob + self.epsilon)
        return log_lik / len(observations)

    def diagnose(self, observed_values):
        current_h = {h: np.random.choice([0, 1]) for h in self.mrf.hidden_vars}
        current_energy = self.mrf.compute_energy(current_h, observed_values)
        best_h = current_h.copy()
        best_energy = current_energy
        T_start = 10.0
        T_end = 0.1
        n_steps = 500
        cooling_rate = (T_end / T_start) ** (1 / n_steps)
        T = T_start
        for step in range(n_steps):
            h_name = np.random.choice(self.mrf.hidden_vars)
            new_h = current_h.copy()
            new_h[h_name] = 1 - new_h[h_name]
            new_energy = self.mrf.compute_energy(new_h, observed_values)
            delta_E = new_energy - current_energy
            if delta_E < 0:
                accept = True
            else:
                accept = np.random.random() < np.exp(-delta_E / T)
            if accept:
                current_h = new_h
                current_energy = new_energy
                if current_energy < best_energy:
                    best_h = current_h.copy()
                    best_energy = current_energy
            T *= cooling_rate
        return best_h, best_energy

def generate_synthetic_data_gibbs(mrf, n_samples=200, n_burnin=100):
    """Генерация синтетических данных методом Гиббса (честное семплирование)"""
    observations = []
    true_hiddens = []
    for _ in range(n_samples):
        h_true, o_true = mrf.gibbs_sample(n_burnin=n_burnin)
        observations.append(o_true)
        true_hiddens.append(h_true)
    return observations, true_hiddens

def validate_mrf_on_synthetic_data(mrf, em, n_synthetic=200):
    """Валидация на синтетике, сгенерированной через Гиббс"""
    print()
    print("ВАЛИДАЦИЯ НА СИНТЕТИЧЕСКИХ ДАННЫХ (Гиббс)")
    print()
    synth_obs, true_hiddens = generate_synthetic_data_gibbs(
        mrf, n_samples=n_synthetic, n_burnin=100
    )
    correct_predictions = {h: 0 for h in mrf.hidden_vars}
    total = len(synth_obs)
    for obs, true_h in zip(synth_obs, true_hiddens):
        best_h, _ = em.diagnose(obs)
        for h in mrf.hidden_vars:
            if best_h.get(h, 0) == true_h.get(h, 0):
                correct_predictions[h] += 1
    print(f"Точность восстановления на {total} синтетических примерах:")
    for h in mrf.hidden_vars:
        acc = correct_predictions[h] / total
        print(f"  {h}: {acc:.3f} ({acc*100:.1f}%)")
    return correct_predictions

def compare_with_logistic_regression(observations):
    X = [[obs["O1"], obs["O2"], obs["O3"], obs["O4"]] for obs in observations]
    y = [obs["O5"] for obs in observations]
    split_idx = int(len(observations) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print()
    print("СРАВНЕНИЕ С ЛОГИСТИЧЕСКОЙ РЕГРЕССИЕЙ")
    print()
    print(f"Точность логистической регрессии: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print("Коэффициенты модели:")
    feature_names = ["O1", "O2", "O3", "O4"]
    for name, coef in zip(feature_names, lr.coef_[0]):
        print(f"  {name}: {coef:.4f}")
    return lr, accuracy

def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path)
    observed_names = [
        "О1 Состояние баланса (отрицательный =1) Признак установлен",
        "О2 Состояние просроченной дебиторки(target) (рост=1) Признак установлен",
        "О3 Состояние денежных ресурсов по отношению к расчетной прибыли (падает =1) Признак установлен",
        "О4 Состояние расчетной прибыли (падение =1) Признак установлен",
        "О5 Темп роста новых контрактов (низкий=1) Признак установлен",
    ]
    observed_short = ["O1", "O2", "O3", "O4", "O5"]
    observations = []
    for idx, row in df.iterrows():
        obs = {}
        for long_name, short_name in zip(observed_names, observed_short):
            val = row.get(long_name, 0)
            if pd.isna(val):
                val = 0
            obs[short_name] = int(val)
        if all(v in [0, 1] for v in obs.values()):
            observations.append(obs)
    print(f"Загружено {len(observations)} наблюдений")
    o5_values = [obs["O5"] for obs in observations]
    print(
        f"Низкий темп новых контрактов (O5=1): {sum(o5_values)} ({sum(o5_values)/len(o5_values)*100:.1f}%)"
    )
    print(f"Норма (O5=0): {len(o5_values) - sum(o5_values)}")
    return observations

def create_full_graph():
    hidden_vars = ["H1", "H2", "H3", "H4", "H5"]
    observed_vars = ["O1", "O2", "O3", "O4", "O5"]
    mrf = MRF_Graph(hidden_vars, observed_vars)
    mrf.add_edge("H1", "O1")
    mrf.add_edge("H2", "O3")
    mrf.add_edge("H3", "H1")
    mrf.add_edge("H3", "O3")
    mrf.add_edge("H4", "O2")
    mrf.add_edge("H4", "O3")
    mrf.add_edge("H5", "O4")
    for o in ["O1", "O2", "O3", "O4"]:
        mrf.add_edge(o, "O5")
    print(
        f"Создан граф со {len(hidden_vars)} скрытыми и {len(observed_vars)} наблюдаемыми узлами"
    )
    return mrf

def visualize_results(mrf, best_h, observed_values):
    G = nx.Graph()
    for node in mrf.hidden_vars:
        color = "lightcoral" if best_h.get(node, 0) == 1 else "lightgreen"
        G.add_node(node, type="hidden", color=color)
    for node in mrf.observed_vars:
        color = "lightsalmon" if observed_values.get(node, 0) == 1 else "lightblue"
        G.add_node(node, type="observed", color=color)
    for v1, v2 in mrf.pairwise_weights.keys():
        G.add_edge(v1, v2)
    pos = nx.spring_layout(G, seed=42, k=2)
    colors = [G.nodes[n]["color"] for n in G.nodes]
    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=colors,
        node_size=2500,
        font_size=10,
        font_weight="bold",
        edge_color="gray",
        width=1.5,
    )
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor="lightgreen", label="Скрытая: норма"),
        plt.Rectangle((0, 0), 1, 1, facecolor="lightcoral", label="Скрытая: проблема"),
        plt.Rectangle((0, 0), 1, 1, facecolor="lightblue", label="Наблюдаемая: норма"),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="lightsalmon", label="Наблюдаемая: симптом"
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.title("Диагностическая карта")
    plt.tight_layout()
    plt.show()

def main():
    print()
    print("ЗАГРУЗКА ДАННЫХ И ОБУЧЕНИЕ MRF МОДЕЛИ")
    print()
    file_path = "RMF -data1.xlsx"
    try:
        observations = load_and_prepare_data(file_path)
    except FileNotFoundError:
        print(f"Файл {file_path} не найден.")
        return
    mrf = create_full_graph()
    weights_file = "mrf_weights.pkl"
    try:
        mrf.load_weights(weights_file)
        print("Веса загружены из файла, обучение не требуется")
        em = EM_Algorithm(mrf, learning_rate=0.1)
    except FileNotFoundError:
        print("Инициализация весов с нуля...")
        mrf.init_weights_expert()
        print(f"Количество ребер: {len(mrf.pairwise_weights)}")
        print()
        print("ЗАПУСК EM-АЛГОРИТМА")
        print()
        em = EM_Algorithm(mrf, learning_rate=0.1)
        log_likelihoods = em.fit(observations, n_iterations=20, verbose=True)
        mrf.save_weights(weights_file)
        plt.figure(figsize=(10, 5))
        plt.plot(log_likelihoods, marker="o")
        plt.xlabel("Итерация")
        plt.ylabel("Средний логарифм правдоподобия")
        plt.title("Сходимость EM-алгоритма")
        plt.grid(True)
        plt.show()
    compare_with_logistic_regression(observations)
    print()
    print("ЗАПУСК ВАЛИДАЦИИ НА СИНТЕТИЧЕСКИХ ДАННЫХ (Гиббс)")
    print()
    val_results = validate_mrf_on_synthetic_data(mrf, em, n_synthetic=200)
    print()
    print("ДИАГНОСТИКА (MAP-ВЫВОД)")
    print()
    test_symptoms = (
        observations[-1]
        if observations
        else {"O1": 1, "O2": 1, "O3": 1, "O4": 1, "O5": 1}
    )
    print(f"Наблюдаемые симптомы: {test_symptoms}")
    best_h, best_energy = em.diagnose(test_symptoms)
    print(f"Наиболее вероятные причины (энергия = {best_energy:.4f}):")
    for h in ["H1", "H2", "H3", "H4", "H5"]:
        status = "ПРОБЛЕМА" if best_h.get(h, 0) == 1 else "норма"
        print(f"  {h}: {best_h.get(h, 0)} ({status})")
    print("Маргинальные вероятности P(H_i=1 | O):")
    h_stats, _ = em.e_step_single_observation(test_symptoms)
    for h in ["H1", "H2", "H3", "H4", "H5"]:
        prob = h_stats[h][1]
        print(f"  P({h}=1 | O) = {prob:.3f} ({prob*100:.1f}%)")
    print("Визуализация диагностической карты...")
    visualize_results(mrf, best_h, test_symptoms)
    print()
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print()

if __name__ == "__main__":
    main()
