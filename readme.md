# **PROJECT: ANALYZE WITH CLUSTERING**

![image](https://user-images.githubusercontent.com/131565885/235119922-ccceab26-cd2f-4a7d-b0d9-35f01f9fa7a8.png)

# **I. Dataset Overview**
**Let's take a look at the dataset**
```php
df_country = pd.read_csv('Country_Data.csv')
df_country
```
![image](https://user-images.githubusercontent.com/131565330/234447826-3425b238-c99a-49e9-8125-1852ab87cd98.png)

**Visualize data to see Life Expectancy according to Country**
```php
import plotly.express as px
fig = px.choropleth(df_country, locations="Country", locationmode='country names', color='Life Expectancy (Year)', hover_name="Country", color_continuous_scale="tealrose", width=1300, height=500)
fig.update_layout(title_text = '<b>Life Expectancy (Year)<b>', title_x = 0.5)
fig.show()
```
![image](https://user-images.githubusercontent.com/131565330/234448079-07c5b28a-ebfd-4b54-887b-f4fa0e3df9b7.png)

# **II. Data Visualization**
**Visualize data to see the correlation of the Life Expectancy (Year)  with other factors**

**By Scatter Chart**
```php
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.set_facecolor("white")
sns.scatterplot(ax=axes[0, 0], data=df_country, x='CO2 emissions (Billion tons)', y='Life Expectancy (Year)', s=200, alpha=0.7,color='red')    
sns.scatterplot(ax=axes[0, 1], data=df_country, x='GDP per capita ($)', y='Life Expectancy (Year)', s=200, alpha=0.7, color='green')
sns.scatterplot(ax=axes[0, 2], data=df_country, x='Rate of using basic drinking water (%)', y='Life Expectancy (Year)', s=200, alpha=0.7, color='#fe00fa')
sns.scatterplot(ax=axes[1, 0], data=df_country, x='Obesity among adults (%)', y='Life Expectancy (Year)', s=200, alpha=0.7, color='#17becf')
sns.scatterplot(ax=axes[1, 1], data=df_country, x='Beer consumption per capita (Liter)', y='Life Expectancy (Year)', s=200, alpha=0.7, color='#ff9900')
```
![image](https://user-images.githubusercontent.com/131565330/234448416-737be06f-cf12-4a36-8c10-bfdc0735c955.png)

**By Heat map**
```php
plt.figure(figsize=(15,10))
plt.title('CORRELATION ')
corre= df_country.corr()
sns.heatmap(corre, square= True, annot= True, fmt= '.2f', annot_kws= {'size':10}, linecolor='white', linewidths=0.5);
```
![image](https://user-images.githubusercontent.com/131565330/234448958-8ef3e0e2-1316-478e-ae22-b29c0c8d4290.png)

# **III. K-Means**
**I will need to find the optimal value of K for K-means**

**1. Data for training after drop column 'Country'**
```php
data= df_country.drop(['Country'], axis='columns')
X = data.iloc[:,:-1].values
```

**2. Transform data to fit the input of computational functions**
```php
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)
```

**3. Use Elbow method and Silhouette score**

**Elbow method**
```php
k_war= {'init':'random','n_init': 10,'max_iter': 300,'random_state': 42}
sse= []
for i in range(1,11):
   kmeans= KMeans(n_clusters=i, **k_war)
   kmeans.fit(scaled_features)
   sse.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.style.use('fivethirtyeight')
with plt.style.context("seaborn-white"):
    fig, ax = plt.subplots()  
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE");
```
![image](https://user-images.githubusercontent.com/131565330/234451627-c976669c-a761-4e65-bf09-f1c41332431d.png)

**Silhouette score**
```php
silhouette= []
for i in range(2,11):
    kmeans= KMeans(n_clusters=i, **k_war)
    kmeans.fit(scaled_features)
    silhouette_sc= silhouette_score(scaled_features, kmeans.labels_)
    silhouette.append(silhouette_sc)
plt.figure(figsize=(12,6))
plt.style.use('fivethirtyeight')
with plt.style.context("seaborn-white"):
    fig, ax = plt.subplots()  
plt.plot(range(2,11), silhouette)
plt.xticks(range(2,11))
plt.xlabel("Number of Clusters")
plt.ylabel("silhouette_score");
```
![image](https://user-images.githubusercontent.com/131565330/234451721-476a9b4d-c928-40b7-8a77-d50b2226408d.png)
 
**4. Use KneeLocator to determine elbow**

![image](https://user-images.githubusercontent.com/131565330/234452209-e9365ec8-7518-4595-a4db-4923635f30fd.png)

# **IV. Build model**
**Now I have Clusters using K-MEANS**
```php
kmeans = KMeans(n_clusters = 3)
y_kmeans = kmeans.fit_predict(X)
y_kmeans
df_country['class']= y_kmeans
import plotly.express as px
fig = px.choropleth(df_country, locations="Country", locationmode='country names', color='class', hover_name="Country", color_continuous_scale="tealrose")
fig.update_layout(title_text = 'Life Expectancy by Country(K-MEANS)', title_x = 0.5)
fig.show()
```
![image](https://user-images.githubusercontent.com/131565330/234453090-a09104cd-d3fc-4007-b904-c3ada07a641b.png)

# **V. Data Analysis**
**1. [Life Expectancy] between Class**
```php
plt.figure(figsize=(12,6))
plt. grid(False)
sns.violinplot(data=df_country, x='class', y='Life Expectancy (Year)');
```
![image](https://user-images.githubusercontent.com/131565330/234453384-7cbccd2b-9b59-47da-b42d-40d200634e70.png)

We can see that
* Group 1 has the most variety range of life expectancy
* While the lowest life expectancy of Group 0 is about 72 years old, the lowest life expectancy of Group 2 is 77 years old

**2. Correlations between [CO2 emissions] and Life expectancy across different Income groups**
```php
fig = px.scatter(df_country, x = 'CO2 emissions (Billion tons)', y ='Life Expectancy (Year)',
                    size ='GDP per capita ($)', color = 'class', title='Correlations between CO2 emissions and Life expectancy across different Income groups',
                 template='plotly_white',
                 labels={ # replaces default labels by column name
                 "CO2 emissions (Billion tons)": "<b>CO2 emissions (Billion tons)<b>",'Life Expectancy (Year)':'<b>Life Expectancy (Year)<b>'},)
fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
fig.update_layout(title=dict(font=dict(color='black', size=20)))
fig.show()
```
![newplot](https://user-images.githubusercontent.com/131565330/234457624-a39b6b65-bcc5-444f-a0a1-7650847ecc17.png)

High-income countries have a long life expectancy and also emit high amounts of CO2 emissions while poor countries have a lower life expectancy and emit less CO2.

**3. Correlations between [Rate of using basic drinking water] and Life expectancy across different Income groups**
```php
fig = px.scatter(df_country, x = 'Rate of using basic drinking water (%)', y ='Life Expectancy (Year)',
                    size ='GDP per capita ($)', color = 'class', title='Correlations between Rate of using basic drinking water and Life expectancy across different Income groups',
                 template='plotly_white',
                 labels={ # replaces default labels by column name
                 "Rate of using basic drinking water (%)": "<b>Rate of using basic drinking water (%)<b>",'Life Expectancy (Year)':'<b>Life Expectancy (Year)<b>'},)
fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
fig.update_layout(title=dict(font=dict(color='black', size=20)))
fig.show()
```
![newplot (1)](https://user-images.githubusercontent.com/131565330/234458065-f5af835d-eb12-40df-849b-360346b3d8b1.png)

High-income countries have high access to clean drinking water and therefore they have a long life expectancy. In contrast, poor countries have lower access to safe drinking water and lower life expectancy as a result.

**4. Correlations between [Obesity among adults] and Life expectancy across different Income groups**
```php
fig = px.scatter(df_country, x = 'Obesity among adults (%)', y ='Life Expectancy (Year)',
                    size ='GDP per capita ($)', color = 'class', title='Correlations between Obesity among adults and Life expectancy across different Income groups',
                 template='plotly_white',
                 labels={ # replaces default labels by column name
                 "Rate of using basic drinking water (%)": "<b>Obesity among adults (%)<b>",'Life Expectancy (Year)':'<b>Life Expectancy (Year)<b>'},)
fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
fig.update_layout(title=dict(font=dict(color='black', size=20)))
fig.show()
```
![newplot (2)](https://user-images.githubusercontent.com/131565330/234458391-de15725f-0e7e-4462-b988-2b9eab0a5904.png)

We can see that the higher income, the higher the obesity rate among adults. Poor countries tend to have lower obesity rates and this might be due to low income which leads to a lack of food.

**5. Correlations between [Beer consumption per capita] and Life expectancy across different Income groups**
```php
fig = px.scatter(df_country, x = 'Beer consumption per capita (Liter)', y ='Life Expectancy (Year)',
                    size ='GDP per capita ($)', color = 'class', title='Correlations between Beer consumption per capita and Life expectancy across different Income groups',
                 template='plotly_white',
                 labels={ # replaces default labels by column name
                 "Beer consumption per capita (Liter)": "<b>Beer consumption per capita (Liter)<b>",'Life Expectancy (Year)':'<b>Life Expectancy (Year)<b>'},)
fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
fig.update_layout(title=dict(font=dict(color='black', size=20)))
fig.show()
```

![newplot (3)](https://user-images.githubusercontent.com/131565330/234460988-e100f1d6-c7c7-446d-97b4-7ccf5d587621.png)

As can be seen in the chart above, it seems like beer consumption per capita does not have much impact on the life expectancy of classes.
