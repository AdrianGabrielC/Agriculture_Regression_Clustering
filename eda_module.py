import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def visualize_data_distribution(df):
    for col in df.select_dtypes(include=['number']).columns.drop(["Season", "Repetition"]):
        plt.figure()
        sns.displot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

def show_best_cultivars_gy(df):
    best_gy = df[['Cultivar', 'GY']].groupby("Cultivar").mean()
    best_gy_sorted = best_gy.sort_values(by='GY', ascending=False)  # Sort the data by GY values

    plt.figure(figsize=(24, 6))  # Increase figure size
    sns.barplot(x='Cultivar', y='GY', data=best_gy_sorted)
    plt.xticks(rotation=90)  # Rotate x-axis labels by 45 degrees and align to the right
    plt.xlabel('Cultivar')
    plt.ylabel('Average GY')
    plt.tight_layout()
    plt.show()

def show_delta(df, column):
    # First chart for GY
    s1 = df[df.Season == 1][['Cultivar', 'MHG', 'GY']]
    s1_cultivar_avg = s1.groupby('Cultivar').mean()

    s2 = df[df.Season == 2][['Cultivar', 'MHG', 'GY']]
    s2_cultivar_avg = s2.groupby('Cultivar').mean()

    s1_cultivar_avg.reset_index(inplace=True)
    s2_cultivar_avg.reset_index(inplace=True)

    merged_df = pd.merge(s1_cultivar_avg, s2_cultivar_avg, on='Cultivar', suffixes=('_season1', '_season2'))
    merged_df['GY_difference'] = abs(merged_df['GY_season2'] - merged_df['GY_season1'])
    merged_df['MHG_difference'] = abs(merged_df['MHG_season2'] - merged_df['MHG_season1'])

    merged_df.plot(x="Cultivar", y=[f'{column}_season1', f'{column}_season2'], kind='bar', figsize=(22, 10))
    if column == 'GY':plt.ylabel('Grain Yield (kg/ha)')
    else: plt.ylabel('Thousand Seed Weight (g)')
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Cultivar', y=f'{column}_difference', data=merged_df)
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.xlabel('Cultivar')
    plt.ylabel(f'Difference in {column} values')
    plt.title(f'Difference in {column} values between two datasets for each Cultivar')
    plt.tight_layout()
    plt.show()



def remove_outliers(df):
    # Remove outliers
    for column in df.select_dtypes(include=['number']).columns:
        # Find the limits of GY
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        # Compute the upper and lower limits
        lower_limit = q1 - (1.5 * iqr)
        upper_limit = q3 + (1.5 * iqr)

        # Replace the outliers to the max/min limit
        df.loc[(df[column] > upper_limit), column] = upper_limit
        df.loc[(df[column] < lower_limit), column] = lower_limit
    df.boxplot()
