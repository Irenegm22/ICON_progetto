import pandas as pd
import numpy as np

# Function to generate random numbers within a specific range
def generate_random_numbers_float(low, high, size):
    return np.round(np.random.uniform(low, high+1, size), 2)

def generate_random_numbers(low, high, size):
    return np.random.randint(low, high+1, size)

# Generate data for 500 patients
num_patients = 2000
age = generate_random_numbers(18, 90, num_patients)
gender = np.random.choice([0, 1], size=num_patients)            # 0 = female. 1 = male
lymphocytesT = generate_random_numbers(400, 2000, num_patients)
lymphocytesB = generate_random_numbers(480, 3300, num_patients)
redcells = generate_random_numbers_float(3.2, 8.0, num_patients)
background = np.random.choice([0, 1], size=num_patients)        # 0 = no cancer history. 1 = cancer history
diagnosis = '-'

# Create a pandas DataFrame
data = {'Age': age, 'Gender': gender, 'Lymphocytes T': lymphocytesT, 'Lymphocytes B': lymphocytesB, 'Red Cells': redcells, 'Background': background, 'Diagnosis': diagnosis}
df = pd.DataFrame(data)

# Show the first 5 rows of the DataFrame
print(df.head())

# Save the DataFrame to a CSV file
df.to_csv('./Dataset/patients_dataset.csv', index=False)