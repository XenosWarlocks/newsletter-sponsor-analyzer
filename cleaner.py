import pandas as pd
import os

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
input_file_path = os.path.join(script_dir, "processed_sponsors.csv")
output_file_path = os.path.join(script_dir, "cleaned_file.csv") # Explicit output path

# --- Load Data ---
try:
    df = pd.read_csv(input_file_path)
    print(f"Successfully loaded data from {input_file_path}. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Input file not found at {input_file_path}")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- Data Cleaning and Type Conversion ---
# Convert dates, handle potential errors by setting invalid parsing as NaT (Not a Time)
df["sponsorship_date"] = pd.to_datetime(df["sponsorship_date"], errors='coerce')
df["processed_at"] = pd.to_datetime(df["processed_at"], errors='coerce')

# --- Normalization ---
# Create normalized columns to avoid modifying original data directly if needed later
# Normalize issue_number (convert to string and strip)
df['issue_number_norm'] = df['issue_number'].astype(str).str.strip()

# Normalize company_name (strip, lowercase, remove common suffixes like 'inc.', 'inc', '.')
df['company_name_norm'] = df['company_name'].str.strip().str.lower()
# Remove optional space + optional '.' + 'inc' + optional '.' at the end of the string
df['company_name_norm'] = df['company_name_norm'].str.replace(r'\s*\.?inc\.?$', '', regex=True)
# Remove trailing periods that might be left
df['company_name_norm'] = df['company_name_norm'].str.replace(r'\.$', '', regex=True).str.strip()
# Add more rules here if needed (e.g., removing 'ltd', 'llc', etc.)

print("Normalization applied.")
# print("\nUnique normalized company names (sample):")
# print(df['company_name_norm'].unique()[:20])


# --- Deduplication ---
# Sort BEFORE dropping duplicates to control which row is kept.
# Sort by the identifying columns (issue, company) and then by the 'keep' criteria column (processed_at).
df_sorted = df.sort_values(
    by=['issue_number_norm', 'company_name_norm', 'processed_at'],
    ascending=[True, True, False], # Sort issue/company ascending, processed_at descending
    na_position='last' # Put rows with missing processed_at last
)

# Drop duplicates based on the normalized columns, keeping the 'first' row
# which is now the one with the latest 'processed_at' time due to the sort.
df_cleaned = df_sorted.drop_duplicates(
    subset=['issue_number_norm', 'company_name_norm'],
    keep='first'
)
print(f"Duplicates identified based on 'issue_number' and normalized 'company_name'. Kept the most recent entry based on 'processed_at'.")
print(f"Shape after deduplication: {df_cleaned.shape}")

# --- Final Output Preparation ---
# Remove the temporary normalization columns
df_cleaned = df_cleaned.drop(columns=['issue_number_norm', 'company_name_norm'])

# Convert issue_number back to numeric for proper sorting if they are all numbers, otherwise keep as string
try:
    df_cleaned['issue_number'] = pd.to_numeric(df_cleaned['issue_number'])
    df_cleaned = df_cleaned.sort_values(by=['issue_number', 'company_name'], ascending=[False, True])
except ValueError:
     # Handle cases where issue_number might not be purely numeric
     print("Warning: 'issue_number' contains non-numeric values. Sorting as string.")
     df_cleaned = df_cleaned.sort_values(by=['issue_number', 'company_name'], ascending=[False, True])


# --- Save Cleaned Data ---
try:
    df_cleaned.to_csv(output_file_path, index=False)
    print(f"\nCleaned file saved successfully as '{output_file_path}'.")
except Exception as e:
    print(f"Error saving cleaned file: {e}")
