import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def generate_synthetic_data(df, season):
    cultivar_groups = df.groupby('Cultivar')
    cultivar_dataframes = []
    synthetic_data_result = []
    for _, group_df in cultivar_groups:
        cultivar_dataframes.append(group_df.copy())
    for cultivar_df in cultivar_dataframes:
        cultivar = cultivar_df['Cultivar'].iloc[0]
        data = cultivar_df.drop(columns=['Cultivar', 'Season', 'Repetition'])
        parameters = {}
        for column in data.columns:
            mean_estimate = data[column].mean()
            std_estimate = data[column].std()
            parameters[column] = {'mean': mean_estimate, 'std': std_estimate}
        num_samples = 5
        synthetic_data = {}
        for column in data.columns:
            synthetic_data[column] = np.random.normal(loc=parameters[column]['mean'],
                                                      scale=parameters[column]['std'],
                                                      size=num_samples)
        synthetic_df = pd.DataFrame(synthetic_data)
        synthetic_df['Cultivar'] = cultivar
        synthetic_df['Season'] = season
        synthetic_data_result.append(synthetic_df)

    combined_df = pd.concat(synthetic_data_result, ignore_index=True)
    return combined_df

def preprocess(dataframe):
    df = pd.get_dummies(dataframe, columns=['Cultivar'], drop_first=True)
    df = pd.get_dummies(df, columns=['Season'], drop_first=True)
    return df

def regression(df):
    Y = df['MHG']
    X = df.drop(columns=['MHG'])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)
    print('Mean Square error:', mse)
    print("R2 score:", r2)
    return model

def predict(model, dataframe):
    df = preprocess(dataframe)
    Y = df['MHG']
    X = df.drop(columns=['MHG'])

    y_predict = model.predict(X)
    mse = mean_squared_error(Y, y_predict)
    r2 = r2_score(Y, y_predict)
    print(mse)
    print(r2)

    for i in range(10):
        print("Real Y:", Y[i])
        print("Predicted Y:" , y_predict[i])

    return (y_predict, mse, r2)

