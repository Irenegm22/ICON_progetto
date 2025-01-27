import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from pyswip import Prolog  # Importar la librería pyswip para interactuar con Prolog

from Pre_procesing import utils
from Pre_procesing.outliners_manager import remove_outliners_iqr
from Supervised_training.supervised_learning import train_models_with_cv

# Función para obtener el diagnóstico desde Prolog
def get_diagnosis_from_prolog(LymphocytesT, LymphocytesB, RedCells, Background, prolog_instance):
    query = f"disease_type({LymphocytesT}, {LymphocytesB}, {RedCells}, {Background}, Diagnoses)"
    result = list(prolog_instance.query(query))
    if result:
        # Convertir Diagnoses a una lista de enfermedades separadas por comas
        diagnoses = result[0]["Diagnoses"]
        return ", ".join(diagnoses)  # Convertir Diagnoses a una cadena separada por comas
    return "0" # no diagnosis

def assign_class_numbers(dataset):
    # Diccionario de mapeo para las clases
    class_mapping = {
        'linfoma': 1,
        'leucemia': 2,
        'anemia, leucemia': 3,
        'anemia, linfoma': 4,
        'posible cancer': 5,
        'anemia, posible cancer': 6
    }
    
    # Asignar un número entero a cada clase
    dataset['Diagnosis'] = dataset['Diagnosis'].map(class_mapping)
    #map_dataset = df.to_csv('./Dataset/patients_data_mapped.csv', index=False)
    
    # Verificar que se han asignado correctamente los números
    print(dataset['Diagnosis'].value_counts())
    return dataset

def preprocess_data(patients_data, verbose=False, min_samples=10):
    # Seleccionar las columnas relevantes
    selected_columns = patients_data[['Age', 'Gender', 'Lymphocytes T', 'Lymphocytes B', 'Red Cells', 'Background', 'Diagnosis']]

    if verbose:
        plot_column_statistics(selected_columns, ['Age', 'Lymphocytes T', 'Lymphocytes B', 'Red Cells'])

    # Eliminar valores atípicos de las columnas numéricas
    for column in ['Age', 'Lymphocytes T', 'Lymphocytes B', 'Red Cells']:
        selected_columns = remove_outliners_iqr(selected_columns, column)

    if verbose:
        plot_column_statistics(selected_columns, ['Age', 'Lymphocytes T', 'Lymphocytes B', 'Red Cells'])
        show_stats_table_text(selected_columns)

    # Filtrar clases con menos de 'min_samples' muestras
    class_counts = selected_columns['Diagnosis'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    selected_columns = selected_columns[selected_columns['Diagnosis'].isin(valid_classes)]

    if verbose:
        print(f"Distribución de las clases después de filtrar: {selected_columns['Diagnosis'].value_counts()}")

    # Guardar el dataset preprocesado
    selected_columns.to_csv('./Dataset/patients_data_outliners_removed.csv', index=False)

    return selected_columns

def train_and_predict(processed_dataset, apply_smote=False):
    # Crear las variables predictoras (X) y la variable objetivo (y)
    X = processed_dataset[['Age', 'Gender', 'Lymphocytes T', 'Lymphocytes B', 'Red Cells', 'Background']]
    y = processed_dataset['Diagnosis']

    # Convertir todas las columnas a tipo float para evitar errores de tipo
    X = X.astype(float)

    # Asegurarse de que y esté en formato numérico
    y = y.astype(int)

    # Aplicar SMOTE si está habilitado
    if apply_smote:
        smote = SMOTE()
        X, y = smote.fit_resample(X, y)

    print(f"SMOTE aplicado: {apply_smote}. Distribución de la variable objetivo: {y.value_counts()}")

    # Entrenar modelos con validación cruzada
    best_models = train_models_with_cv(X, y)

    # Realizar predicciones con los mejores modelos
    dataset_with_predictions = processed_dataset.copy()

    for model in best_models:
        print(f"Mejor modelo para {model}: {best_models[model]['best_estimator']}")
        print(f"Mejores parámetros para {model}: {best_models[model]['best_params']}")
        dataset_with_predictions[model] = best_models[model]['best_estimator'].predict(X)

    dataset_with_predictions.to_csv('./Dataset/patients_data_predictions.csv', index=False)
    return dataset_with_predictions

def main():
    # Cargar el dataset original
    raw_dataset = pd.read_csv('./Dataset/patients_dataset.csv')

    # Cargar la base de conocimientos de Prolog
    prolog = Prolog()
    prolog.consult("KB_cancer.pl")  
    
    # Calcular la columna Diagnosis usando Prolog
    raw_dataset['Diagnosis'] = raw_dataset.apply(
        lambda row: get_diagnosis_from_prolog(
            row['Lymphocytes T'], row['Lymphocytes B'], row['Red Cells'], row['Background'], prolog
        ),
        axis=1
    )

    # Asignar números a las clases
    raw_dataset = assign_class_numbers(raw_dataset)

    # Preprocesar los datos (incluyendo el filtro de clases)
    processed_dataset = preprocess_data(raw_dataset)
    
    # Escalar las características numéricas
    scaler = MinMaxScaler(feature_range=(0, 1))
    numeric_columns = processed_dataset.select_dtypes(include=['float64', 'int64']).columns
    processed_dataset[numeric_columns] = scaler.fit_transform(processed_dataset[numeric_columns])

    # Entrenar y predecir sin SMOTE
    dataset_with_predictions_NO_SMOTE = train_and_predict(processed_dataset, apply_smote=False)

    # Entrenar y predecir con SMOTE
    dataset_with_predictions_SMOTE = train_and_predict(processed_dataset, apply_smote=True)

    # Guardar los datasets resultantes
    dataset_with_predictions_SMOTE.to_csv('./Dataset/patients_data_predictions_SMOTE.csv', index=False)
    dataset_with_predictions_NO_SMOTE.to_csv('./Dataset/patients_data_predictions_NO_SMOTE.csv', index=False)

if __name__ == '__main__':
    main()