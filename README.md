# Flood Probability Prediction using Linear Regression

This project demonstrates a simple machine learning pipeline to predict flood probability based on a given dataset (`flood.csv`). It uses Python with libraries like Pandas for data manipulation, Scikit-learn for machine learning tasks (preprocessing, model training, evaluation), and Matplotlib for visualization.

## Project Workflow

The script follows these main steps:

1.  **Import Libraries:** Imports necessary libraries: `pandas`, `numpy`, `matplotlib.pyplot`, and specific modules from `sklearn`.
2.  **Load Data:** Reads the dataset from the `flood.csv` file into a Pandas DataFrame.
3.  **Initial Exploration:**
    *   Prints the first few rows of the DataFrame (`df.head()`) to give a preview of the data.
    *   Checks and prints the count of missing values for each column before handling them.
4.  **Data Preprocessing:**
    *   **Handle Missing Values:** Fills any missing *numeric* values in the DataFrame with the mean of their respective columns.
    *   **Feature Removal:** Drops the `RiverManagement` column from the DataFrame, potentially because it's irrelevant, redundant, or problematic for the model.
    *   **Re-check Missing Values:** Prints the count of missing values again to confirm they've been handled (or removed via column drop). *Note: The second `fillna` call after dropping the column might be redundant if the first one handled all numeric NaNs and the dropped column was the only remaining source of NaNs.*
    *   **Feature Scaling:** Separates features (X) and the target variable (y - assumed to be the last column, likely 'FloodProbability'). It then uses `MinMaxScaler` to scale the features (X) to a range between 0 and 1. This helps ensure that all features contribute equally to the model training process.
5.  **Train-Test Split:** Splits the scaled feature data (X) and the target variable (y) into training (70%) and testing (30%) sets. `random_state=42` ensures reproducibility of the split.
6.  **Model Training:**
    *   Initializes a `LinearRegression` model.
    *   Trains the model using the training data (`X_train`, `y_train`).
7.  **Prediction:** Uses the trained model to make predictions on the unseen test data (`X_test`).
8.  **Model Evaluation:**
    *   Calculates three common regression metrics to evaluate the model's performance:
        *   **Root Mean Squared Error (RMSE):** Measures the average magnitude of the errors (lower is better).
        *   **R-squared (R²) Score:** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables (closer to 1 is better).
        *   **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted values (lower is better).
    *   Prints these performance metrics.
9.  **Visualization:**
    *   Generates a scatter plot comparing the actual flood probability values (`y_test`) against the predicted values (`y_pred`).
    *   Includes an "Ideal Fit" line (y=x) in red dashed style for reference. Points lying close to this line indicate accurate predictions.
    *   Adds labels, a title, a legend, and a grid for better readability.
    *   Displays the plot.

## Requirements

*   Python 3.x
*   Pandas
*   NumPy
*   Matplotlib
*   Scikit-learn

You can install the required libraries using pip:
```bash
pip install pandas numpy matplotlib scikit-learn
```

![image](https://github.com/user-attachments/assets/a2c0a8bc-e5cb-4264-867a-2ea5db6d9925)
