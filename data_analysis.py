import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_name):
    """Load CSV file and return DataFrame."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, file_name)
    logging.info(f"Attempting to load {file_name} from: {file_path}")
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise

def handle_missing_data(df):
    """Handle missing data for both numeric and non-numeric columns."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

    non_numeric_imputer = SimpleImputer(strategy='most_frequent')
    df[non_numeric_columns] = non_numeric_imputer.fit_transform(df[non_numeric_columns])

    return df

def filter_salary(df, threshold):
    """Filter DataFrame based on salary threshold."""
    return df[df['Salaries'] < threshold]

def calculate_days_between(start_date, end_date):
    """Calculate days between two dates."""
    return (end_date - start_date).days

def format_date(date_obj):
    """Format date in different ways."""
    return {
        'YYYY-MM': date_obj.strftime('%Y-%m'),
        'YYYY-DDD': date_obj.strftime('%Y-%j'),
        'MONTH (YYYY)': date_obj.strftime('%B (%Y)')
    }

def analyze_cities(df):
    """Perform various analyses on the cities DataFrame."""
    # Top 10 states in female-male sex ratio
    top_states_sex_ratio = df.groupby('state_name')['sex_ratio'].mean().sort_values(ascending=False).head(10)

    # Top 10 cities in total number of graduates
    top_cities_graduates = df.nlargest(10, 'total_graduates')[['name_of_city', 'total_graduates']]

    # Top 10 cities and their locations in respect of total effective literacy rate
    top_cities_literacy = df.nlargest(10, 'effective_literacy_rate_total')[['name_of_city', 'location', 'effective_literacy_rate_total']]

    return top_states_sex_ratio, top_cities_graduates, top_cities_literacy

def plot_data(df):
    """Create various plots from the data."""
    # Histogram of Total Literates
    plt.figure(figsize=(10, 6))
    plt.hist(df['literates_total'], bins=30)
    plt.title('Histogram of Total Literates')
    plt.xlabel('Total Literates')
    plt.ylabel('Frequency')
    plt.show()

    # Scatter plot of Male vs Female Graduates
    plt.figure(figsize=(10, 6))
    plt.scatter(df['male_graduates'], df['female_graduates'])
    plt.title('Male Graduates vs Female Graduates')
    plt.xlabel('Male Graduates')
    plt.ylabel('Female Graduates')
    plt.show()

    # Boxplot of Total Effective Literacy Rate
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df['effective_literacy_rate_total'])
    plt.title('Boxplot of Total Effective Literacy Rate')
    plt.show()

def main():
    # Load datasets
    df_data = load_data('Data.csv')
    df_cities = load_data('Indian_cities.csv')

    # Print data types and first few rows of df_data
    logging.info("Data types of df_data:")
    logging.info(df_data.dtypes)
    logging.info("\nFirst few rows of df_data:")
    logging.info(df_data.head())

    # Handle missing data
    df_data = handle_missing_data(df_data)
    
    # Filter salary
    df_filtered = filter_salary(df_data, 70000)
    logging.info(f"\nNumber of rows after filtering salary: {len(df_filtered)}")

    # Date calculations
    hurricane_start = date(2007, 5, 9)
    hurricane_end = date(2007, 12, 13)
    days_elapsed = calculate_days_between(hurricane_start, hurricane_end)
    logging.info(f"\nDays elapsed between hurricanes: {days_elapsed}")

    # Date formatting
    andrew_date = date(1992, 8, 26)
    formatted_dates = format_date(andrew_date)
    for format_name, formatted_date in formatted_dates.items():
        logging.info(f"{format_name}: {formatted_date}")

    # Analyze cities data
    top_states_sex_ratio, top_cities_graduates, top_cities_literacy = analyze_cities(df_cities)
    logging.info("\nTop 10 states in female-male sex ratio:")
    logging.info(top_states_sex_ratio)
    logging.info("\nTop 10 cities in total number of graduates:")
    logging.info(top_cities_graduates)
    logging.info("\nTop 10 cities and their locations in respect of total effective literacy rate:")
    logging.info(top_cities_literacy)

    # Plot data
    plot_data(df_cities)

    # Handle null values
    null_counts = df_cities.isnull().sum()
    logging.info("\nNumber of null values in each column:")
    logging.info(null_counts)

    df_cities_clean = df_cities.dropna()
    logging.info(f"\nRows in original dataset: {len(df_cities)}")
    logging.info(f"Rows after removing null values: {len(df_cities_clean)}")

if __name__ == "__main__":
    main()