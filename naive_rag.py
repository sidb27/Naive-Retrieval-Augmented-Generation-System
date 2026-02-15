import pandas as pd
import os
import google.generativeai as genai
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Gemini API is configured with the API Key
if not os.getenv("API_KEY"):
    raise ValueError("Missing API_KEY environment variable. Please set API_KEY before running.")
genai.configure(api_key=os.getenv("API_KEY"))

# File paths
data_file_path = "data/data_scientist_salaries.csv"
preprocessed_file_path = "data/preprocessed_data.csv"
faiss_index_path = "data/vector_index.faiss"  # FIXED (was wrongly pointing to CSV)

# Ensuring the directory for FAISS index file exists
faiss_index_dir = os.path.dirname(faiss_index_path)
if not os.path.exists(faiss_index_dir):
    os.makedirs(faiss_index_dir)
    print(f"Directory created for FAISS index: {faiss_index_dir}")

# Preprocesses the original dataset if not already preprocessed
def preprocess_data(input_path, output_path):
    print("Preprocessing the dataset...")
    data = pd.read_csv(input_path)

    # Retain relevant columns
    relevant_columns = [
        "Hobby", "OpenSource", "Country", "Student", "Employment",
        "FormalEducation", "UndergradMajor", "CompanySize", "DevType",
        "YearsCoding", "Salary", "SalaryType", "ConvertedSalary"
    ]
    data = data[relevant_columns]

    # Drop rows with missing values in critical columns
    critical_columns = ["ConvertedSalary", "YearsCoding", "DevType"]
    data = data.dropna(subset=critical_columns)

    # Save the preprocessed dataset
    data.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved to: {output_path}")
    return data

# Checks if the preprocessed data exists. If it does not it preprocess the original dataset
if not os.path.exists(preprocessed_file_path):
    if os.path.exists(data_file_path):
        filtered_data = preprocess_data(data_file_path, preprocessed_file_path)
    else:
        raise FileNotFoundError(f"Original data file not found at: {data_file_path}")
else:
    filtered_data = pd.read_csv(preprocessed_file_path)
    print(f"Loaded preprocessed dataset from: {preprocessed_file_path}")

# Embeddings are generated for the dataset
def generate_embeddings(data, column, model_name="all-MiniLM-L6-v2"):
    """
    Generate embeddings for a specific column in the dataset.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(data[column].tolist(), show_pr_
