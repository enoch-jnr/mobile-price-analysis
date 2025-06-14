
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


df=pd.read_csv('Mobiles Dataset (2025).csv',encoding='iso-8859-1')
df.set_index('Launched Year',inplace=True)
df.head(10)
print(df.columns)
print(df.describe())
print(df.shape)
print(df.isnull().sum())
print(df.info())

print(df.index.name)



prices_column=['Launched Price (USA)','Launched Price (China)',
               'Launched Price (India)','Launched Price (Pakistan)',
               'Launched Price (Dubai)']
for col in prices_column:
    df[col]=df[col].replace(r'[^\d.]','',regex=True)
    df[col]=pd.to_numeric(df[col],errors='coerce')
conversions={'Launched Price (USA)':1,
             'Launched Price (China)':7.1865,
             'Launched Price (India)':85.7956,
             'Launched Price (Pakistan)':282.33,
             'Launched Price (Dubai)':3.6725
             }
for col in prices_column:
    df[col]=df[col]/(conversions[col])

    Apple_Phones = df[df['Company Name'] == 'Apple'].copy()
    Apple_Phones_2020_Subset = Apple_Phones[['Model Name', 'Launched Price (USA)', 'Launched Price (China)',
                                             'Launched Price (India)', 'Launched Price (Pakistan)',
                                             'Launched Price (Dubai)',
                                             'RAM', 'Back Camera']
    ]
    Apple_Phones_Subset = Apple_Phones_2020_Subset.melt(
        id_vars=['Model Name', 'RAM', 'Back Camera'],
        var_name='Country',
        value_name='Price'
    )
    fig = px.scatter(Apple_Phones_Subset,
                     x='Model Name',
                     y='Price',
                     color='Country',
                     hover_name='Back Camera',
                     title='Iphone Prices In Different Countries in 2020',
                     )
    fig.update_layout(
        xaxis_title='Model Name',
        yaxis_title='Price',
        xaxis_tickangle=45,
    )
    fig.show()

    Apple_Phones_2021 = Apple_Phones.loc[2021]
    Apple_Phones_2021_Subset = Apple_Phones[[
        'Model Name', 'Launched Price (China)', 'Launched Price (India)',
        'Launched Price (Dubai)', 'Launched Price (USA)', 'RAM', 'Back Camera'
        , 'Launched Price (Pakistan)'
    ]]
    Apple_Phones_2021_Subset = Apple_Phones_2021_Subset.melt(
        id_vars=['Model Name', 'RAM', 'Back Camera'],
        var_name='Country',
        value_name='Price'
    )
    fig = px.line(Apple_Phones_Subset,
                  x='Model Name',
                  y='Price',
                  color='Country',
                  hover_name='RAM',
                  title='Iphone Prices In Different Countries in 2021',
                  )
    fig.update_layout(
        xaxis_title='Model Name',
        yaxis_title='Price In USD',
        xaxis_tickangle=45,

    )
    fig.show()

    Samsung = df[df['Company Name'] == 'Samsung'].copy()

    print(Samsung.shape)
    Samsung_prices = Samsung[['Model Name', 'Launched Price (China)', 'Launched Price (India)',
                              'Launched Price (Dubai)', 'Launched Price (USA)', 'RAM', 'Back Camera'
        , 'Launched Price (Pakistan)'
                              ]]
    Samsung_prices_Merge = Samsung_prices.melt(
        id_vars='Model Name',
        var_name='Country',
        value_name='Price'
    )
    print(Samsung_prices_Merge.head(10))
    fig = px.bar(Samsung_prices_Merge,
                 x='Model Name',
                 y='Price',
                 title='Samsung Prices In Different Countries in 2021',
                 color='Country',

                 barmode='group')
    fig.update_layout(
        xaxis_title='Model Name',
        yaxis_title='Price In USD',
        xaxis_tickangle=45,
    )
    fig.show()
    Samsung_2021 = Samsung.loc[2021]
    print(Samsung_2021.shape)
    print(Samsung_2021.head(10))

    Samsung_Apple = pd.concat([Samsung, Samsung_2021, Apple_Phones])
    Samsung_Apple_2024 = Samsung_Apple.loc[2024]
    Samsung_Apple_2024.tail(20)
    Samsung_Apple_2024_Prices = Samsung_Apple_2024[['Model Name', 'Launched Price (China)', 'Launched Price (India)',
                                                    'Launched Price (Dubai)', 'Launched Price (USA)', 'RAM',
                                                    'Back Camera'
        , 'Launched Price (Pakistan)'
                                                    ]].copy()
    print(Samsung_Apple_2024_Prices.head(10))
    Samsung_Apple_2024_Prices['RAM'] = Samsung_Apple_2024_Prices['RAM'].replace('GB', '', regex=True)
    Samsung_Apple_2024_Prices['RAM'] = pd.to_numeric(Samsung_Apple_2024_Prices['RAM'], errors='coerce')
    print(Samsung_Apple_2024_Prices['RAM'].dtypes)
    Samsung_Apple_2024_Prices_melted = Samsung_Apple_2024_Prices.melt(
        id_vars=['Model Name', 'RAM', 'Back Camera'],
        var_name='Country',
        value_name='Price'
    )
    print(Samsung_Apple_2024_Prices_melted.head())

    fig = px.scatter(Samsung_Apple_2024_Prices_melted,
                     x='Model Name',
                     y='Price',
                     title='Samsung vs Apple Prices In Different Countries in 2024',
                     color='Country',
                     hover_name='Back Camera',
                     size='RAM',

                     )
    fig.update_layout(
        xaxis_title='Model Name',
        yaxis_title='Price In USD',
        xaxis_tickangle=45,
    )
    fig.show()
    print(Samsung_Apple_2024_Prices_melted.tail(10))

    Samsung_Apple_2024_Prices_melted_10 = Samsung_Apple_2024_Prices_melted.loc[::10]
    fig = px.sunburst(
        Samsung_Apple_2024_Prices_melted_10,
        path=['Model Name', 'RAM', 'Country'],
        values='Price'
    )
    fig.show()

    Samsung_Apple_2024_Prices_melted = Samsung_Apple_2024_Prices_melted[
        Samsung_Apple_2024_Prices_melted['Model Name'].isin(['Galaxy S24 Ultra 256GB', 'iPhone 16 Pro Max 512GB'])]
    Samsung_Apple_2024_Prices_melted.head(10)
    fig = px.scatter(Samsung_Apple_2024_Prices_melted_10,
                     x='Model Name',
                     y='Price',
                     color='Model Name',
                     hover_name='Back Camera',
                     size='RAM',
                     )
    fig.update_layout(
        xaxis_title='Model Name',
        yaxis_title='Price In USD',
        xaxis_tickangle=45,
    )
    fig.show()

    Samsung_Apple_2024_Prices_melted_10 = Samsung_Apple_2024_Prices_melted_10.copy()
    le = LabelEncoder()
    Samsung_Apple_2024_Prices_melted_10['Model Name Encoded'] = le.fit_transform(
        Samsung_Apple_2024_Prices_melted_10['Model Name'])
    X = Samsung_Apple_2024_Prices_melted_10[['RAM']]
    Y = Samsung_Apple_2024_Prices_melted_10['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(Samsung_Apple_2024_Prices_melted_10)
    Samsung_Apple_2024_Prices_melted_10['Model Name'] = le.inverse_transform(
        Samsung_Apple_2024_Prices_melted_10['Model Name Encoded'])
    print(Samsung_Apple_2024_Prices_melted_10['Model Name'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=Samsung_Apple_2024_Prices_melted_10['Model Name'],
        y=Samsung_Apple_2024_Prices_melted_10['Price'],
        mode='markers',
        name='Actual Price',
    ))
    fig.add_trace(go.Scatter(
        x=Samsung_Apple_2024_Prices_melted_10.loc[y_test.index, 'Model Name'],
        y=y_pred,
        mode='markers+text',
        name='Predicted Price',
    ))
    fig.update_layout(
        xaxis_title='Model Name',
        yaxis_title='Price In USD',
        xaxis_tickangle=45,
        showlegend=True,

    )
    fig.show()
    print(y_pred)
    Mean_squared_error = mean_squared_error(y_test, y_pred)
    Mean_absolute_error = mean_absolute_error(y_test, y_pred)
    R2 = r2_score(y_test, y_pred)
    print('The mean square error of the prediction is', Mean_squared_error)
    print('The mean absolute error of the prediction is', Mean_absolute_error)
    print('The r2 score of the prediction is', R2)
