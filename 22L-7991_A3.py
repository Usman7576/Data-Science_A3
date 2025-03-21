import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib as mpl
mpl.rcParams['axes.formatter.use_mathtext'] = True
def load_and_process_data(electricity_path, weather_path):
    logs = {
        'electricity_files': 0,
        'weather_files': 0,
        'electricity_records': 0,
        'weather_records': 0,
        'merged_records': 0,
        'anomalies': []
    }

    # Load and process electricity data
    electricity_dfs = []
    for file in glob.glob(os.path.join(electricity_path, '*.json')):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data['response']['data'])
            electricity_dfs.append(df)
            logs['electricity_files'] += 1
        except (KeyError, json.JSONDecodeError) as e:
            logs['anomalies'].append(f"Invalid file structure in {os.path.basename(file)}: {str(e)}")
            continue

    if not electricity_dfs:
        raise ValueError("No valid electricity data found")
    
    df_electricity = pd.concat(electricity_dfs, ignore_index=True)
    logs['electricity_records'] = len(df_electricity)

    try:
        df_electricity['period'] = pd.to_datetime(df_electricity['period'], format='%Y-%m-%dT%H')
        df_electricity['value'] = pd.to_numeric(df_electricity['value'], errors='coerce')
    except KeyError as e:
        raise KeyError(f"Missing critical column in electricity data: {str(e)}")

    # Load and process weather data
    weather_dfs = []
    for file in glob.glob(os.path.join(weather_path, '*.csv')):
        try:
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(file, encoding=encoding)
                    weather_dfs.append(df)
                    logs['weather_files'] += 1
                    break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            logs['anomalies'].append(f"Failed to load {os.path.basename(file)}: {str(e)}")
            continue

    if not weather_dfs:
        raise ValueError("No valid weather data found")
    
    df_weather = pd.concat(weather_dfs, ignore_index=True)
    
    try:
        df_weather['date'] = pd.to_datetime(
            df_weather['date'],
            format='ISO8601',
            utc=True,
            errors='coerce'
        ).dt.tz_convert(None)
        
        # Remove duplicates and invalid dates
        df_weather = df_weather.drop_duplicates(subset=['date'])
        df_weather = df_weather.dropna(subset=['date'])
        df_weather['temperature_2m'] = pd.to_numeric(df_weather['temperature_2m'], errors='coerce')
        logs['weather_records'] = len(df_weather)
    except KeyError as e:
        raise KeyError(f"Missing critical column in weather data: {str(e)}")

    # Merge datasets with validation
    df_merged = pd.merge(
        df_electricity,
        df_weather,
        left_on='period',
        right_on='date',
        how='inner'
    ).drop(columns=['date'], errors='ignore')
    
    logs['merged_records'] = len(df_merged)
    logs['validation'] = {
        'missing_electricity_values': df_electricity['value'].isna().sum(),
        'missing_weather_temps': df_weather['temperature_2m'].isna().sum(),
        'electricity_duplicates': df_electricity.duplicated().sum(),
        'weather_duplicates': df_weather.duplicated().sum(),
        'merged_missing_values': df_merged[['value', 'temperature_2m']].isna().sum().sum()
    }

    return df_merged, logs

def preprocess_data(df, logs):
    processed_df = df.copy()
    
    # Handle missing data
    processed_df['value'] = processed_df['value'].ffill()
    processed_df['temperature_2m'] = processed_df['temperature_2m'].interpolate(method='linear')
    
    # Feature engineering
    processed_df['hour'] = processed_df['period'].dt.hour
    processed_df['day_of_week'] = processed_df['period'].dt.dayofweek
    processed_df['is_weekend'] = processed_df['day_of_week'].isin([5, 6]).astype(int)
    processed_df['month'] = processed_df['period'].dt.month
    processed_df['season'] = (processed_df['month'] % 12 + 3) // 3
    
    # Handle duplicates
    initial_rows = len(processed_df)
    processed_df = processed_df.drop_duplicates(subset=['period', 'subba'])
    logs['duplicates_removed'] = initial_rows - len(processed_df)
    
    # Outlier detection and handling
    numeric_cols = ['value', 'temperature_2m']
    logs['outliers'] = {}
    os.makedirs('eda_results', exist_ok=True)
    for col in numeric_cols:
        # Store original distribution stats
        original_stats = {
            'mean': processed_df[col].mean(),
            'std': processed_df[col].std(),
            'min': processed_df[col].min(),
            'max': processed_df[col].max()
        }
        
        # IQR Method
        q1 = processed_df[col].quantile(0.25)
        q3 = processed_df[col].quantile(0.75)
        iqr = q3 - q1
        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr
        iqr_outliers = processed_df[(processed_df[col] < iqr_lower) | (processed_df[col] > iqr_upper)]
        
        # Z-score Method
        z_scores = np.abs(stats.zscore(processed_df[col].dropna()))
        z_threshold = 3
        z_outliers = processed_df[(z_scores > z_threshold)]
        
        # Store detection results
        logs['outliers'][col] = {
            'IQR': {
                'lower_bound': iqr_lower,
                'upper_bound': iqr_upper,
                'outlier_count': len(iqr_outliers),
                'percentage': len(iqr_outliers)/len(processed_df)*100
            },
            'Z_score': {
                'threshold': z_threshold,
                'outlier_count': len(z_outliers),
                'percentage': len(z_outliers)/len(processed_df)*100
            },
            'original_stats': original_stats
        }
        
        # Visualization: Before handling
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(x=processed_df[col])
        plt.title(f'Original {col} Distribution')
        
        # Apply IQR-based capping
        if col == 'value' and iqr_lower < 0:
            iqr_lower = 0
        processed_df[col] = processed_df[col].clip(iqr_lower, iqr_upper)
        
        # Visualization: After handling
        plt.subplot(1, 2, 2)
        sns.boxplot(x=processed_df[col])
        plt.title(f'Processed {col} Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join('eda_results', f'{col}_outlier_comparison.png'))
        plt.close()
        
        # Store final stats
        logs['outliers'][col]['final_stats'] = {
            'mean': processed_df[col].mean(),
            'std': processed_df[col].std(),
            'min': processed_df[col].min(),
            'max': processed_df[col].max()
        }
    
    logs['data_conservation'] = {
        'original': len(df),
        'final': len(processed_df),
        'percentage': len(processed_df)/len(df)*100
    }
    
    return processed_df, logs
def perform_eda(df, output_dir='eda_results'):
    os.makedirs(output_dir, exist_ok=True)
    report = []
    
    # 1. Statistical Summary with Full Metrics
    numerical_cols = df.select_dtypes(include=np.number).columns
    stats_df = df[numerical_cols].agg(['mean', 'median', 'std', 'skew', 'kurtosis']).T
    stats_df.columns = ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis']
    report.append("=== Statistical Summary ===\n" + stats_df.to_string())
    
    # 2. Enhanced Time Series Analysis with Proper Date Handling
    plt.figure(figsize=(15, 8))
    ax = df.set_index('period')['value'].plot(title='Electricity Demand Analysis')
    
    # Convert dates to numerical format correctly
    x_ordinal = np.array([d.toordinal() for d in df['period']])
    z = np.polyfit(x_ordinal, df['value'], 1)
    p = np.poly1d(z)
    ax.plot(df['period'], p(x_ordinal), 'r--', label='Trend Line')
    
    # Seasonal annotations
    for year in df['period'].dt.year.unique():
        ts = pd.Timestamp(f'{year}-07-01')
        if ts >= df['period'].min() and ts <= df['period'].max():
            ax.axvline(ts, color='g', linestyle=':', alpha=0.5)
            ax.text(ts, df['value'].max()*0.9, 'Summer', 
                   rotation=90, verticalalignment='top')
    
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'annotated_demand.png'))
    plt.close()
    for col in numerical_cols:
        if col in ['value', 'temperature_2m']:
            # Z-score distribution plot
            plt.figure(figsize=(12, 6))
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            sns.histplot(z_scores, bins=30)
            plt.axvline(3, color='r', linestyle='--')
            plt.title(f'{col} Z-score Distribution')
            plt.savefig(os.path.join(output_dir, f'{col}_zscore_dist.png'))
            plt.close()

    # 3. Robust Univariate Analysis with Error Handling
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        plt.figure(figsize=(15, 5))
        
        # Skip non-numeric columns accidentally included
        if not np.issubdtype(df[col].dtype, np.number):
            continue
            
        # Histogram with KDE
        plt.subplot(1, 3, 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'{col} Distribution')
        
        # Boxplot
        plt.subplot(1, 3, 2)
        sns.boxplot(x=df[col])
        plt.title(f'{col} Spread')
        
        # Q-Q Plot with validation
        plt.subplot(1, 3, 3)
        try:
            stats.probplot(df[col].dropna(), plot=plt)
            plt.title(f'{col} Q-Q Plot')
        except Exception as e:
            plt.close()
            report.append(f"\nQ-Q Plot failed for {col}: {str(e)}")
            continue
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{col}_analysis.png'))
        plt.close()
        
        # Distribution analysis
        skew = stats_df.loc[col, 'Skewness']
        kurt = stats_df.loc[col, 'Kurtosis']
        report.append(f"\n--- {col} Distribution ---")
        report.append(f"Skewness: {skew:.2f} ({'Normal' if -0.5<skew<0.5 else 'Moderate' if -1<skew<1 else 'Extreme'})")
        report.append(f"Kurtosis: {kurt:.2f} ({'Normal' if -2<kurt<2 else 'High' if kurt>2 else 'Low'})")

    # 4. Correlation Analysis with Validation
    corr_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(12, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix (Lower Triangle)')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    
    # 5. Time Series Decomposition with Adaptive Frequency
    try:
        freq = 24*365 if len(df) > 24*365*2 else 24*30
        decomposition = seasonal_decompose(
            df.set_index('period')['value'],
            model='additive',
            period=freq
        )
        plt.figure(figsize=(12, 8))
        decomposition.plot()
        plt.suptitle('Time Series Decomposition')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'decomposition.png'))
        plt.close()
    except Exception as e:
        report.append(f"\nDecomposition Error: {str(e)}")
    
    # 6. Stationarity Test with Clear Reporting
    adf_result = adfuller(df['value'].dropna())
    report.append("\n=== Stationarity Test Results ===")
    report.append(f"ADF Statistic: {adf_result[0]:.4f}")
    report.append(f"p-value: {adf_result[1]:.4f}")
    report.append("Critical Values:")
    for key, val in adf_result[4].items():
        report.append(f"{key}: {val:.3f}")
    conclusion = "Stationary" if adf_result[1] < 0.05 else "Non-Stationary"
    report.append(f"Conclusion: {conclusion}")

    # Save comprehensive report
    with open(os.path.join(output_dir, 'full_analysis_report.txt'), 'w') as f:
        f.write("\n".join(report))

    return report
def build_regression_model(df, output_dir='model_results'):
    """
    Build and evaluate regression model to predict electricity demand
    Returns model metrics and generates evaluation visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics = {}
    
    # 1. Feature Selection
    features = ['hour', 'day_of_week', 'month', 'season', 'temperature_2m']
    target = 'value'
    
    X = df[features]
    y = df[target]
    
    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Model Development
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 4. Model Evaluation
    metrics['mse'] = mean_squared_error(y_test, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['r2'] = r2_score(y_test, y_pred)
    
    # 5. Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Demand')
    plt.ylabel('Predicted Demand')
    plt.title('Actual vs Predicted Electricity Demand')
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
    plt.close()
    
    # 6. Residual Analysis
    residuals = y_test - y_pred
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, color='red')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.subplot(1, 2, 2)
    stats.probplot(residuals, plot=plt)
    plt.title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_analysis.png'))
    plt.close()
    
    # 7. Save metrics
    with open(os.path.join(output_dir, 'model_metrics.txt'), 'w') as f:
        f.write("Regression Model Metrics:\n")
        f.write(f"MSE: {metrics['mse']:.2f}\n")
        f.write(f"RMSE: {metrics['rmse']:.2f}\n")
        f.write(f"R² Score: {metrics['r2']:.2f}\n")
    
    return metrics, model
if __name__ == "__main__":

    ELECTRICITY_PATH = r'F:\DataScience\Pre_processing\electricity_raw_data'
    WEATHER_PATH = r'F:\DataScience\Pre_processing\weather_raw_data'
    OUTPUT_PATH = r'F:\DataScience\Pre_processing\processed_data.csv'

    try:
        # Data processing
        merged_df, logs = load_and_process_data(ELECTRICITY_PATH, WEATHER_PATH)
        processed_df, logs = preprocess_data(merged_df, logs)
        
        # Save results
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)  # Added directory creation
        processed_df.to_csv(OUTPUT_PATH, index=False)
        
        # Generate reports
        print("\n=== Processing Report ===")
        print(f"Merged Records: {logs['merged_records']}")
        print(f"Final Records: {len(processed_df)}")
        print(f"Data Conservation: {logs['data_conservation']['percentage']:.1f}%")
        
        print("\n=== Quality Checks ===")
        if logs['anomalies']:
            print("Warnings:")
            for warning in logs['anomalies']:
                print(f"- {warning}")
        else:
            print("No significant issues found")
        
        # Perform EDA
        perform_eda(processed_df)
        print("\n=== EDA Completed ===")
        print("Check 'eda_results' folder for analysis outputs")
        # New Regression Modeling
        model_metrics, model = build_regression_model(processed_df)
        
        print("\n=== Regression Model Results ===")
        print(f"Mean Squared Error: {model_metrics['mse']:.2f}")
        print(f"Root Mean Squared Error: {model_metrics['rmse']:.2f}")
        print(f"R² Score: {model_metrics['r2']:.2f}")
        print("Check 'model_results' folder for evaluation visualizations")

    except Exception as e:
        print(f"Error: {str(e)}")
