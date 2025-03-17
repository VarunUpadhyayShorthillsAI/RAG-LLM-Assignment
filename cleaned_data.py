import pandas as pd  

# Load the CSV file (preserving line breaks)
csv_file_path = "final_medicalDataSet.csv"  # Replace with your file path
df = pd.read_csv(csv_file_path, dtype=str, keep_default_na=False, encoding="utf-8", encoding_errors="replace")


# Strip leading/trailing spaces and convert column names to lowercase
df.columns = df.columns.str.strip().str.lower()

# Debug: Print column names and first few rows
print("Column names in the DataFrame:", df.columns.tolist())  
print("\nFirst few rows of the DataFrame:")  
print(df.head())  

# Filter out rows where the "answer" column starts with "I'm unsure"
if 'answer' in df.columns:
    df = df[~df['answer'].astype(str).str.startswith("I'm unsure")]
else:
    print("Warning: 'answer' column not found in CSV!")

# Save the cleaned data back to a new CSV file
cleaned_csv_file_path = "cleaned_file_with_context.csv"  # Replace with your desired output file path
df.to_csv(cleaned_csv_file_path, index=False)  

print(f"Cleaned data saved to {cleaned_csv_file_path}")  
