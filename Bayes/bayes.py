import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pgmpy.estimators import HillClimbSearch, K2Score, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork


# Función para cargar los datos
def load_data(file_path, sample_size=100):
    data = pd.read_csv(file_path)
    # Tomar una muestra aleatoria de los datos
    data = data.sample(n=sample_size, random_state=42)
    return data[['Age', 'Gender', 'Lymphocytes T', 'Lymphocytes B', 'Red Cells', 'Background']]


# Función para aprender la estructura del modelo bayesiano
def learn_structure(data):
    estimator = HillClimbSearch(data)
    model = estimator.estimate(scoring_method=K2Score(data), max_indegree=6, max_iter=int(1e5))
    return model


# Función para ajustar los parámetros del modelo bayesiano
def learn_parameters(model, data):
    bayesian_model = BayesianNetwork(model.edges())
    bayesian_model.fit(data, estimator=MaximumLikelihoodEstimator)
    return bayesian_model


# Función para graficar la red bayesiana aprendida y guardar como imagen
def plot_network(model, output_file="bayesian_network.png"):
    G = nx.DiGraph()
    for edge in model.edges():
        G.add_edge(edge[0], edge[1])

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos, with_labels=True, node_size=3000, node_color='skyblue',
        font_size=12, font_weight='bold', arrows=True, edge_color='black', width=2
    )
    plt.title("Learned Bayesian Network")
    plt.savefig(output_file)  # Guardar el gráfico como imagen
    print(f"Gráfico guardado como '{output_file}'")
    plt.close()  # Cerrar la figura para liberar memoria


# Generar ejemplos aleatorios a partir del modelo aprendido
def generate_random_example(model, num_samples=1):
    samples = model.simulate(num_samples)
    return samples


def main():
    # Ruta al dataset
    data_path = './Dataset/patients_dataset.csv'

    # Cargar el dataset (usando solo una muestra más pequeña)
    data = load_data(data_path, sample_size=100)
    
    # Verificar que los datos se están cargando correctamente
    print("Datos cargados:")
    print(data.head())
    
    # Aprender la estructura de la red bayesiana
    model = learn_structure(data)
    print("Learned structure:", model.edges())

    # Si el modelo aprendió una estructura, ajustar los parámetros
    if model.edges():
        # Ajustar los parámetros de la red bayesiana
        model = learn_parameters(model, data)
        
        # Graficar la red y guardar el gráfico como imagen
        plot_network(model, output_file="./Images/bayesian_network.png")
        
        # Mostrar las variables que el modelo ha aprendido
        print("Variables en el modelo bayesiano:", model.nodes())

        # Generar ejemplos aleatorios basados en el modelo aprendido
        random_example = generate_random_example(model, num_samples=5)
        print("Random examples:\n", random_example)

        # Ajustar el orden y las columnas para que coincidan con el modelo
        new_data = pd.DataFrame({
            'Age': ['Middle-Aged', 'Young', 'Senior'],
            'Gender': [1, 0, 1],  # 1 = masculino, 0 = femenino
            'Lymphocytes T': ['Medium', 'Low', 'High'],
            'Lymphocytes B': ['Medium', 'High', 'Low'],
            'Red Cells': ['High', 'Medium', 'Low'],
            'Background': [0, 1, 0]  # Añadimos la columna 'Background' ya que está en el modelo
        })

        # Ajustar el orden y las columnas para que coincidan con el modelo
        new_data = new_data[['Age', 'Gender', 'Lymphocytes T', 'Lymphocytes B', 'Red Cells', 'Background']]

        # Predecir las probabilidades para los nuevos datos
        predicted_probabilities = model.predict(new_data)
        print("Predicted probabilities:\n", predicted_probabilities)
    else:
        print("El modelo no aprendió ninguna estructura.")


if __name__ == '__main__':
    main()
