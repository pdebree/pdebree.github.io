<div id="container" style="position:relative;">
<div style="float:left"><h1> Assumptions and Non-Parametric Statistics </h1></div>
<div style="position:relative; float:right"><img style="height:65px" src ="https://drive.google.com/uc?export=view&id=1EnB0x-fdqMp6I5iMoEBBEuxB_s7AmE2k" />
</div>
</div>


```python
import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

The hypothesis tests and modelling techniques we have learned so far are known as parametric tests/models. This is because the methods have some underlying assumption that the samples we are working with have been drawn from a population that follows some particular probability distribution, usually the normal distribution. 

### Why is it important that the assumptions are met?

Cast your mind back to the hypothesis testing lecture. When we say that we are testing at the 5% significance level in a t-test, we are saying that we will accept at most a 5% chance of a false positive (Type I error). This is the case when we reject $H_0$ when $H_0$ is actually true. We look for a p-value that is less than 0.05 as the p-value gives us the probability that the result we found happened by random chance.

However, the test statistics that we use to calculate the final p-value are based on some assumptions that we will discuss in detail. When these don't hold it can be shown that the real false positive rate when running these test may be higher than we expect! This will lead us to make false conclusions. 


## i.i.d

**Applies to**:
    
   - Most parts of statistics when working with samples!

i.i.d is one of the key assumptions for much of statistics. It stands for "independent and identically distributed". 

When rolling some dice or flipping a coin, people will often guess that one result is 'due' if it has not occurred for a while. However, this is not how the dice or coins behave, each flip or roll is independent from the previous one. 

This is necessary for much of statistics. Each new data point should be free to vary based on its own parameters, not the previous outcomes. Compare this to Monopoly. While in Monopoly you are more likely to land on certain squares based on where you start your turn, you are not more or less likely to roll a certain number based on previous rolls.

Similarly, we cannot expect the variance of the process under study to change as we get new samples. This would mean that we are unable to use the central limit theorem.


**Question:** Do you think sampling without replacement is i.i.d.?

---

## Normality

**Applies to:** </font>
    
   - t-tests (including Pearsons-r test of correlation significance)
   - OLS model residuals

Many of the statistical tests we have used have an assumption of normality. In t-tests, we assume that the sample means are normally distributed (CLT applies here). For Linear Regression, there is no specific assumption that our residuals are normally distributed, however we need them to be normally distributed in order to trust the p-values that have been calculated for our regression coefficients (the $\beta$ values).  

### How can we test for normality?

#### Shapiro-Wilk test

If we ever need to check the normality of data, one possibility is to use the Shapiro-Wilk test.

Shapiro-Wilk tests have a null hypothesis that data is normal and an alternate hypothesis that the data is not.

$$ H_0 : \text{Data is normally distributed}  \quad vs. \quad H_1 : \text{Data is not normally distributed} $$

We can run a Shapiro-Wilk test using scipy:


```python
np.random.seed(1234)
# we randomly generate normally distributed data
data1 = np.random.randn(5000) 
plt.hist(data1,bins=200)

# calculating the shapiro test statisic and its p-value
stats.shapiro(data1)
```




    ShapiroResult(statistic=0.999592125415802, pvalue=0.3965257704257965)




    
![png](output_5_1.png)
    


**Question: Does this mean that if we have a p-value of higher than 0.05, the data is normally distributed?**


```python
np.random.seed(1234)

data2 = np.array([1,2,3]) # we know this is definitely not normally distributed

stats.shapiro(data2) #we can't reject the null
```




    ShapiroResult(statistic=1.0, pvalue=0.9999986886978149)



The Shapiro-Wilk test is only designed to identify if a given data set is **not** normal. If we cannot reject the null hypothesis, it means we didn't have enough evidence to say that it wasn't normal.

The steps when performing a Shapiro-Wilk test are: 

   * We start off with no knowledge of our data's distribution.
   * We suspect it may not be normal, so we decide to perform a Shapiro-Wilk test:
       * If we get a low p-value, we can reject the null hypothesis and conclude that our data is **not** normally distributed.
       * If we get a large p-value, we cannot make conclusions about our data's normality. We are back to having no concrete knowledge of our data's distribution.

The Shapiro-Wilk test is very sensitive, and with large sample sizes, it can sometimes detect even trivial differences from the normal distribution as being significant.

Remember our dice rolling example? Let's say we roll 20 dice at the same time and we take the sum of all 20 dice. We do this 1000 times. As we saw previously, the outcome is supposed to be normally distributed, but sometimes the Shapiro-Wilk test will tell us to reject. 


```python
np.random.seed(123) #this makes sure we get the same numbers!
possiblerolls = list(range(1,7)) # rolling any number from 1 to 6
```


```python
# Randomly generating dice rolls and adding them up
random_dice_rolls1 = np.random.choice(possiblerolls, replace = True, size = (500,20)).sum(axis =1)

plt.figure()
plt.hist(random_dice_rolls1, bins=20)
plt.xlabel('Sum of dice rolls')
plt.show()

print("Shapiro-Wilk Test: ", stats.shapiro(random_dice_rolls1))
```


    
![png](output_10_0.png)
    


    Shapiro-Wilk Test:  ShapiroResult(statistic=0.9957106709480286, pvalue=0.18906182050704956)


The Shapiro-Wilk test conclusion is that we cannot detect non-normality. 


```python
# Randomly generating dice rolls AGAIN and adding them up
random_dice_rolls2 = np.random.choice(possiblerolls, replace = True, size = (500,20)).sum(axis =1)

plt.figure()
plt.hist(random_dice_rolls2, bins=20)
plt.xlabel('Sum of dice rolls')
plt.show()

print("Shapiro-Wilk Test: ", stats.shapiro(random_dice_rolls2))
```


    
![png](output_12_0.png)
    


    Shapiro-Wilk Test:  ShapiroResult(statistic=0.9939136505126953, pvalue=0.04248642921447754)


With this randomly generated data, the p-value is less than 0.05, leading us to conclude that this data is not normally distributed. But it was generated from a normal distribution!

#### Normal Q-Q Plot
A more reasonable method, but one that relies on judgment is to plot out a prob-plot. The prob-plot is also known as a normal Q-Q plot.

In a normal Q-Q plot, we are plotting the sorted values from our data against the expected quantiles from a normal distribution. 

- If the data is normally distributed, we expect to get a straight line.
- If we see skew either in the tails or globally our data is not normal.


```python
# ?stats.probplot
```

Let's get a normal Q-Q plot of the dice rolls where Shapiro-Wilk test detected non-normality. 


```python
plt.figure()
stats.probplot(random_dice_rolls2, dist="norm", plot = plt);
plt.show()
```


    
![png](output_17_0.png)
    


There is definitely some skewness in the tails, but overall, this data looks very normal. 

The normal Q-Q plot also looks very differently when we have different sample sizes. Let's explore this with $n=20,200$ and $2000$.


```python
np.random.seed(12345)
plt.figure()
stats.probplot(np.random.randn(20), dist="norm", plot = plt);
plt.show();
```


    
![png](output_20_0.png)
    



```python
np.random.seed(12345)
plt.figure()
stats.probplot(np.random.randn(200), dist="norm", plot = plt);
plt.show();
```


    
![png](output_21_0.png)
    



```python
np.random.seed(12345)
stats.probplot(np.random.randn(2000), dist="norm", plot = plt);
plt.show();
```


    
![png](output_22_0.png)
    


We can see that the more data we have, the more normal the plot looks. 

There is a substantial amount of subjectivity involved in assessing a Q-Q plot. In reality, we always want to try out multiple methods and make conclusions about normality based on all of our findings.

To see what a non-normal distribution would look like, let's look at how an exponential distribution would look like. Feel free to try out out distributions as well. 


```python
# Exponential
plt.figure()
plt.hist(np.random.exponential(size=100))
plt.title('Histogram for an exponential distribution')
plt.show()


plt.figure()
stats.probplot(np.random.exponential(size = 100), dist="norm", plot = plt);
plt.show()
```


    
![png](output_25_0.png)
    



    
![png](output_25_1.png)
    


#### Practical Implementation:

- Before running a hypothesis test which has a normality assumption, such as the unpaired t-test, you should check for normality of your samples using the Shapiro-Wilk test and the Q-Q plot. It is also worth plotting a histogram of the sample as well. 

- One of the results of the central limit theorem states that the distribution of sample means drawn from any underlying distribution will be approximately normal as sample size $n$ increases. There is no hard and fast rule for what a large $n$ is, however a conservative limit would be $n \geq 50$ for less skewed distributions and $n \geq 100$ for highly skewed distributions. So if you have a large sample size, you can reasonably continue to use a t-test.  

- After running a linear regression model, you need to check that the model residuals are normally distributed. This would involve plotting a histogram, Q-Q plot, and running the Shapiro-Wilk test.
---

### Homoscedasticity

**Applies to:**
   
   - Linear Regression model residuals

One of the key assumptions for a linear regression model to be considered a 'good' method to modelling a particular dependent variable is that your residuals must be **homoscedastic**.

**What does it mean?**

Homoscedastic errors means that the variance in your model residuals should be **constant** as the independent variable(s) change.
 
**Checking for homoscedasticity:**

First we need to fit a model to get some residuals.
    
Let's fit two regression models, one to data we know is heteroscedastic and one to data we know is homoscedastic.


```python
# heteroscedastic 
y = pd.read_csv('data/nonequal_var_data.csv')
y_new = pd.read_csv('data/equal_var_data.csv')

plt.subplots(1,2, figsize=(15,5))
plt.subplot(1,2,1)
plt.scatter(y.index, y)
plt.title('Heteroscedastic data')

plt.subplot(1,2,2)
plt.scatter(y_new.index, y_new)
plt.title('Homoscedastic data')

plt.show()
```


    
![png](output_28_0.png)
    



```python
# These models do not have any real world context, we're just modelling the y 
# as a function of the index of each data point

#Heteroscedastic model 
X = y.index
y = y
X_const = sm.add_constant(X)

#Instantiate and fit a model
hetero_reg = sm.OLS(y,X_const).fit()

# Homoscedastic model 
X_new = y_new.index
y_new = y_new
X_new_const = sm.add_constant(X_new)

#Instantiate and fit a model
homoscedastic_reg = sm.OLS(y_new,X_new_const).fit()
```

**Plotting the residuals:**

To check for heteroscedasticity, we can make a scatter plot where we plot the fitted values of the model (the predictions) on the x-axis and the model residuals on the y-axis. Remember, the residuals are the prediction errors, the difference between the true value for $y$ and the predicted value $\hat{y}$.


```python
# We can pull out the model residuals and fitted values (predictions for x datapoints) 
# using the resid and fittedvalues attributes of the fitted model object

plt.subplots(1,2, figsize=(15,5))
plt.subplot(1,2,1)
plt.scatter(hetero_reg.fittedvalues,hetero_reg.resid)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Heteroscedastic residuals')

plt.subplot(1,2,2)
plt.scatter(homoscedastic_reg.fittedvalues,homoscedastic_reg.resid)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Homoscedastic residuals')

plt.show()

```


    
![png](output_31_0.png)
    


The graph on the left is a classic example of heteroscedastic errors. What it shows us is that when the model is predicting $\hat{y}$ values which are low, the error term is very small. However, for high values of $\hat{y}$ the error terms are massive.

We want to see something like we have on the right. In this graph, the errors tend to vary within the same rough band for all values of X. 

**What does this mean for your regression model?**

If you have heteroscedastic errors, the standard errors (SE) for the calculated regression coefficients are likely to be incorrect and not reflective of their true population values. This also means that tests of coefficient significance (and other tests related to the regression model) cannot be trusted as they use the SE in the denominator when calculating the test statistic.
    
In a more practical sense, consider the following scenario:
    
   - For low values of your independent variable $x$, the variance of the residuals of your regression model are very small(~ 1). 
    
   - For high values of your independent variable $x$, the variance residuals of your regression model are very high (~ 100).

Think about the predictions your model will give you for different values of the independent variable. You can be fairly confident in the predictions $\hat{y}$ given for low values of $x$. But can you be comfortable that the predictions for $\hat{y}$ for high values of $x$? Probably not...

#### Practical implementation: 

- After fitting a regression model, plot out a scatterplot of the fitted values against residuals and check that the pattern matches what you expect for homoscedasticity.  

---
#### Exercise 1

In many retail scenarios, we model sales as a response of other factors, like price, rain, holidays, competitor sales, etc. However, its not always the case that sales is linearly related to these factors. Let's investigate what happens when we try to fit a linear model to data that does not have linear relationships. 


Use the data from [here.](https://drive.google.com/file/d/1AR21EACLjQ8iDsBWrAFauS-5bO_0aA4J/view?usp=sharing )

1. Plot the data on a linear scale - how does sales total vary with price?
2. Carry out a linear regression on the untransformed data - what can we conclude from the analysis?
3. Check the model residuals of the linear regression. Are they normal and homoscedastic? 
4. Take the log of the sales data and plot log(sales) and price. 
5. Carry out a linear regression on the price and transformed sales data and check the residuals? How do these compare to the residuals of your initial model?
6. How would you interpret the coefficients for this model?

---

### Transformations

This isn't an assumption, but rather a potential solution for when you have heteroscedastic errors. 

We saw how applying a transformation to our data can help fix issues we may have with non-normal and heteroscedastic residuals. Modelling using variables that do not have linear relationships with the dependent is one of the main causes of heteroscedasticity.

Transforming data is a very deep topic area and we will be covering more robust techniques later in the course. The main gist behind transformations is to apply some form of mathematical function to all values in a variable to yield a *transformed* variable. There is a wide range of popular functions commonly applied when modelling data that have been shown to empircally improve model performance. It is not always clear which transformation is best. In Exercise 1, it was quite clear that the relationship was exponential. In many cases this may not be the case so we may need to try a range of transformations and compare results.

**What about interpretability?**

The major downside to transforming variables is the impact on interpretability. As we saw from Exercise 1, we no longer can say a unit increase in price leads to a $\beta_i$ increase in sales . Instead we have to say a unit increase in price leads to a $\beta_i$ increase in $log(sales)$. This is far less intuitive than standard linear regression interpretation. We would then need to take the reverse transformation to get our actual $\hat{y}$. 

**Example from Exercise 1:**

**Step 1** : Original regression equation

$$ y = \beta_0 + \beta_1 x_1 $$ </inline>

**Step 2**: Regression equation after taking the log of sales

$$ log(y) = \beta_0 + \beta_1 x_1 $$

**Step 3**: Take the inverse of the log function to  get the actual value of our sales prediction $\hat{y}$.

$$ \hat{y} = e^{(\hat{\beta_0} + \hat{\beta_1} x_1)} $$

#### Practical implementation:

- Plotting scatterplots of your independent variables against the dependent variable will help you visually identify whether there is a non-linear relationship. It may also help you work out which transformation to apply.

---

### Multicollinearity

**Applies to:**

   - Linear & Logistic regression models

**What is it?**

Multicollinearity is the situation where your independent variables are highly correlated with *each other*.

**What does it do to our models?**

 - We often see certain independent variables be given much lower coefficients then what the correlation between that variable and the dependent variable would suggest. It may also go against our prior subject matter knowledge.
 
 - The coefficient standard errors may also be very high, suggesting that the model is unsure about its prediction.
 
 - As a result of both of the above, p-values are likely to be highly skewed, and may lead us to make incorrect conclusions about the significance of the variable. 
 
**Why does this happen?**

Fundamentally, linear regression models use how an independent variable varies with a dependent variable to estimate the coefficient value that describes the change in the $y$. This is why correlation between the independent variable and dependent variables is important to have. If there is a strong relationship the model can look at the data and say "Ah, if $x$ increases by 1, then $y$ generally increases by ~2"

In multiple linear regression, where we have multiple independent variables, we also need to factor in how the independent variables vary with *each other* as well as the dependent variable. Consider this extreme scenario:

   - Two independent variables $x_1$ and $x_2$ have very strong correlation with each other, |$\rho|\geq 0.9 $
   - Both variables variables share strong positive correlation with the dependent $y$, $\rho \approx 0.7 $

In the scenario above, the model will be unsure about which independent variable is the one that actually driving the relationship with the dependent variable. Ultimately both independent variables seem to move in the same way. 

The result of this confusion is that the model will assign the majority of the relationship to one of the variables, and assign little to the other. As a result, one coefficient value will be high and the other very low! This may not match up with what we would expect and lead us to make conclusions about the relationship between the independent variables and the dependent variable that do not reflect the true relationship.


#### Detecting Multicollinearity

The first signs of multicollinearity can be detected in the **correlation coefficients** between the independent variables. If a regression model is to be fitted, this should always be part of the exploratory phase. If the correlation between some predictors is high, it is a sign of multicollinearity.

Let us load in the dataset about cirrhosis death rates which we looked at the other day


```python
drinking = pd.read_csv('data/drinking.csv')
drinking.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Urban_pop</th>
      <th>Late_births</th>
      <th>Wine_consumption_per_capita</th>
      <th>Liquor_consumption_per_capita</th>
      <th>Cirrhosis_death_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>44</td>
      <td>33.2</td>
      <td>5</td>
      <td>30</td>
      <td>41.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43</td>
      <td>33.8</td>
      <td>4</td>
      <td>41</td>
      <td>31.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48</td>
      <td>40.6</td>
      <td>3</td>
      <td>38</td>
      <td>39.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>52</td>
      <td>39.2</td>
      <td>7</td>
      <td>48</td>
      <td>57.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>71</td>
      <td>45.5</td>
      <td>11</td>
      <td>53</td>
      <td>74.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = drinking[['Urban_pop', 'Late_births', 'Wine_consumption_per_capita',
       'Liquor_consumption_per_capita']]
y = drinking['Cirrhosis_death_rate']
```


```python
# lets check the correlations, this should always be done prior to regression 
# modelling In particular we are going to check the correlations between the 
# independent variables

plt.figure()
sns.heatmap(X.corr(), cmap='coolwarm', vmin=-1, vmax=1, annot=True, lw=1)
plt.show()
```


    
![png](output_39_0.png)
    


We can immediately see that our independent variables have relatively strong positive correlation with eachother. This is a good indicator of multicollinearity in our data.


```python
# Lets fit a regression model to this data
X_withconst = sm.add_constant(X)

sm.OLS(y, X_withconst).fit().summary()
```

    /Users/borisshabash/opt/anaconda3/lib/python3.9/site-packages/statsmodels/tsa/tsatools.py:142: FutureWarning: In a future version of pandas all arguments of concat except for the argument 'objs' will be keyword-only
      x = pd.concat(x[::order], 1)





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>Cirrhosis_death_rate</td> <th>  R-squared:         </th> <td>   0.814</td>
</tr>
<tr>
  <th>Model:</th>                     <td>OLS</td>         <th>  Adj. R-squared:    </th> <td>   0.795</td>
</tr>
<tr>
  <th>Method:</th>               <td>Least Squares</td>    <th>  F-statistic:       </th> <td>   44.75</td>
</tr>
<tr>
  <th>Date:</th>               <td>Thu, 26 May 2022</td>   <th>  Prob (F-statistic):</th> <td>1.95e-14</td>
</tr>
<tr>
  <th>Time:</th>                   <td>09:58:33</td>       <th>  Log-Likelihood:    </th> <td> -171.25</td>
</tr>
<tr>
  <th>No. Observations:</th>        <td>    46</td>        <th>  AIC:               </th> <td>   352.5</td>
</tr>
<tr>
  <th>Df Residuals:</th>            <td>    41</td>        <th>  BIC:               </th> <td>   361.6</td>
</tr>
<tr>
  <th>Df Model:</th>                <td>     4</td>        <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>        <td>nonrobust</td>      <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
                <td></td>                   <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                         <td>  -13.9631</td> <td>   11.400</td> <td>   -1.225</td> <td> 0.228</td> <td>  -36.987</td> <td>    9.060</td>
</tr>
<tr>
  <th>Urban_pop</th>                     <td>    0.0983</td> <td>    0.244</td> <td>    0.403</td> <td> 0.689</td> <td>   -0.395</td> <td>    0.591</td>
</tr>
<tr>
  <th>Late_births</th>                   <td>    1.1484</td> <td>    0.583</td> <td>    1.970</td> <td> 0.056</td> <td>   -0.029</td> <td>    2.326</td>
</tr>
<tr>
  <th>Wine_consumption_per_capita</th>   <td>    1.8579</td> <td>    0.401</td> <td>    4.634</td> <td> 0.000</td> <td>    1.048</td> <td>    2.668</td>
</tr>
<tr>
  <th>Liquor_consumption_per_capita</th> <td>    0.0482</td> <td>    0.133</td> <td>    0.361</td> <td> 0.720</td> <td>   -0.221</td> <td>    0.317</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.887</td> <th>  Durbin-Watson:     </th> <td>   2.549</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.143</td> <th>  Jarque-Bera (JB):  </th> <td>   1.988</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.211</td> <th>  Prob(JB):          </th> <td>   0.370</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.073</td> <th>  Cond. No.          </th> <td>    688.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



We have some odd results here. In particular the model is suggesting that `liquor_consumption_per_capita` does not have a significant coefficient so is not useful to predicitng cirrohsis death rates. This doesn't really match up with general medical evidence on the causes of cirrohsis. This is a good example of the effects of multicollinearity. The model is unsure what coefficient to assign to this variable.

**Variance Inflation Factors**

Another common way of detecting multicollinearity is using the **Variance Inflation Factor** (VIF). The VIF is calculated for each predictor. 

In order to calculate it, we build a regression model of each independent variable against the other independent variables and look at the $R^2$. The VIF for each predictor is defined as 
$$
\text{VIF}_i = \frac{1}{1-R_i^2}
$$

In a perfect scenario of no multicollinearity, the VIF for each predictor should be 1 (hence, the $R^2$ from each model is 0). 

By common convention, any VIF value higher than 5 indicates high collinearity.

Let's look at an example. 


```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
```


```python
# This gives us the VIF for column 1 (Urban_pop)
variance_inflation_factor(X_withconst.values, 1)
```




    5.910332534192018



Let's look at the VIF for all columns.


```python
pd.Series([variance_inflation_factor(X_withconst.values, i) 
               for i in range(X_withconst.shape[1])], 
              index=X_withconst.columns)[1:] # leaving out the constant
```




    Urban_pop                        5.910333
    Late_births                      6.748416
    Wine_consumption_per_capita      3.080737
    Liquor_consumption_per_capita    3.488172
    dtype: float64



A high VIF means that the independent variable in question has a higher level of colinearity with the other independent variables.

An obvious step would be to just drop the columns with the highest VIF numbers. One thing to note however is that dropping one variable will change the VIFs for *all the independent variables*.

Lets check the VIFs after dropping `Late_births`.


```python
# drop late births
X_new = X.drop('Late_births', axis=1)

# add constant
X_new_withconst = sm.add_constant(X_new)

#calculate VIF
pd.Series([variance_inflation_factor(X_new_withconst.values, i) 
               for i in range(X_new_withconst.shape[1])], 
              index=X_new_withconst.columns)[1:] # leaving out the constant
```

    /Users/borisshabash/opt/anaconda3/lib/python3.9/site-packages/statsmodels/tsa/tsatools.py:142: FutureWarning: In a future version of pandas all arguments of concat except for the argument 'objs' will be keyword-only
      x = pd.concat(x[::order], 1)





    Urban_pop                        1.855813
    Wine_consumption_per_capita      2.754480
    Liquor_consumption_per_capita    1.843305
    dtype: float64



The remaining columns have seen their VIFs drop substantially!

This result is pretty important. It means that we can reduce VIFs without having to drop a variable we think will be useful for predicting the dependent. Instead, drop a different variable which has high correlation with the variable we want to keep. Let's refit our model from above but without the `Late_births` variable.



```python
# fit model without late births

sm.OLS(y, X_new_withconst).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>Cirrhosis_death_rate</td> <th>  R-squared:         </th> <td>   0.796</td>
</tr>
<tr>
  <th>Model:</th>                     <td>OLS</td>         <th>  Adj. R-squared:    </th> <td>   0.781</td>
</tr>
<tr>
  <th>Method:</th>               <td>Least Squares</td>    <th>  F-statistic:       </th> <td>   54.62</td>
</tr>
<tr>
  <th>Date:</th>               <td>Thu, 26 May 2022</td>   <th>  Prob (F-statistic):</th> <td>1.50e-14</td>
</tr>
<tr>
  <th>Time:</th>                   <td>09:59:05</td>       <th>  Log-Likelihood:    </th> <td> -173.33</td>
</tr>
<tr>
  <th>No. Observations:</th>        <td>    46</td>        <th>  AIC:               </th> <td>   354.7</td>
</tr>
<tr>
  <th>Df Residuals:</th>            <td>    42</td>        <th>  BIC:               </th> <td>   362.0</td>
</tr>
<tr>
  <th>Df Model:</th>                <td>     3</td>        <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>        <td>nonrobust</td>      <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
                <td></td>                   <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                         <td>    3.8706</td> <td>    7.162</td> <td>    0.540</td> <td> 0.592</td> <td>  -10.582</td> <td>   18.324</td>
</tr>
<tr>
  <th>Urban_pop</th>                     <td>    0.4965</td> <td>    0.141</td> <td>    3.512</td> <td> 0.001</td> <td>    0.211</td> <td>    0.782</td>
</tr>
<tr>
  <th>Wine_consumption_per_capita</th>   <td>    1.6008</td> <td>    0.392</td> <td>    4.085</td> <td> 0.000</td> <td>    0.810</td> <td>    2.392</td>
</tr>
<tr>
  <th>Liquor_consumption_per_capita</th> <td>    0.2286</td> <td>    0.100</td> <td>    2.281</td> <td> 0.028</td> <td>    0.026</td> <td>    0.431</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 2.072</td> <th>  Durbin-Watson:     </th> <td>   2.568</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.355</td> <th>  Jarque-Bera (JB):  </th> <td>   1.616</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.459</td> <th>  Prob(JB):          </th> <td>   0.446</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.979</td> <th>  Cond. No.          </th> <td>    375.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The results look more appropriate here. Liquor and wine consumption are both significant. You can also see that the coefficient for Liquor consumption has increased and the standard error has come down slightly. This is a result of removing some of the multicollinearity that had been present. 

#### Practical Implementation:

- Investigate the correlations of your independent variables against each other. Note which ones appear to have strong correlations.
- Calculate the VIFs to quantify the level of collinearity.
- Be methodical and systematic when dropping columns and refitting models.

### Variable Selection
You will hopefully find that some combinations of variables will give almost the same performance using fewer predictors and hence, less multicollinearity. 


Even if there isn't any collinearity present, it's common to try and find a simpler model that will have almost the same performance as a more complicated model. This is usually done by adding or removing variables until all are significant (p-value < 0.05). 

The technical term for this is called variable selection. In fact there's a few way to do this, some of which are: 
- Forward Selection: Starting with the intercept only, at each step, select the candidate variable that increases R-Squared the most
- Backward Selection: Starting with a full model (all predictors), at each step, the variable that is the least significant is removed. In the case of multicollinearity, the variable removed may be the one causing the most collinear issues. 
- Stepwise Selection: Combination of the above two, after each step in which a predictor was added, all predictor candidates in the model are checked to see if their significance has been reduced below 0.05

We will not be going through examples of this, but it is something that you might come across in your career.

### Non-Parametric Statistics

When you are faced with the following two problems:
   - Your sample is non-normally distributed.
   - The sample size is too small. Which means you cannot rely upon the result of the CLT that states that the distribution of sample means taken from any random population distribution will approach normal as $n$ increases.

It is not appropriate to run a parametric test such as the t-test. Instead we must look towards a group of testing methods known as non-parametric tests.

Non-parametric tests generally do not make assumptions that your variables come from a particular distribution and have less stringent requirements. Instead, they work by ranking values and comparing ranks (although there are many other methods). They are extremely useful when analyzing data that may not have an underlying distribution, like ranking of movies in order of preference, or cases where we know that the relationship between variables is not normal.



We have non-parametric alternatives for a range of tests. In general they are less [powerful](https://en.wikipedia.org/wiki/Power_of_a_test) and more likely to produce a Type II error, but they are more robust in scenarios where the assumptions of a parametric test do not hold.

**Parametric Test Alternatives:**

| Parametric test  | Non-Parametric test  |
|:---:|:---:|
| One-Sample t-test | [One-Sample Wilcoxon Signed Rank test](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test) or [One-Sample Sign test](https://en.wikipedia.org/wiki/Sign_test)   | 
| Two Sample Paired t-test | [Wilcoxon Signed Rank test](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test) | 
| Two Sample Unpaired t-test | [Mann-Whitney U test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test) |
| Pearson's Correlation  | [Spearman's Rank Correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) |  

For most cases there are different options you could use, however the examples in the above table are the most commonly used non-parametric test alternatives.

For linear and logistic regression alternatives, there are many non-parametric models that are able to predict both categorical and continuous values (KNN, Decision Trees, etc.), which are considered machine learning models and will be studied later!

---

## Supplementary - Simulation Case Study

### Hypothesis Test for $\mu$

**Example: The Vancouver Morning Commute**  
Let's consider a study where we are trying to estimate the average amount of time it takes for a Vancouverite to get to work. Our speculation is that the average morning commute is 0.5 hours. To test this we conducted a random survey of size n for people throughout Vancouver. To put statistically:  
- **Population** : City of Vancouver residents  
- **Population Parameter** : The mean morning commute time of everyone in Vancouver  
- **Sampling** : simple random sample of size n  
- **Point estimate** : The sample morning commute time  
- **Null Hypothesis** ($H_0$) : $\mu = 0.5$ hrs
- **Alternative Hypothesis** ($H_a$) : $\mu \ne 0.5$ hrs

*The Vancouver Morning Commute* example is made up for this exercise to give context. Our goal given this context is actually to evaluate the quality of each of the hypothesis test techniques under a variety of conditions. To do this we need to create a theoretical world where we know the true parameters. We can then sample our theoretical population and compare the performance of each technique. 

Given the problem set up, the approach we will want to consider is a one sample hypothesis test for the population mean. Tabulated below are a few techniques and their corresponding assumptions.

|Assumption|Description| t test | Bootstrap-t |
|:---|:---|:---:|:---:|
|Independent and Identically Distributed (i.i.d)| The observations of the sample come from the same distribution and are not dependent on previous observations | X | X |
|Normality | When the sample is small, we require that the observations come from a normally distributed population. The larger the sample size the more we can relax this assumption | X | |


For our simulation, lets consider the following world:
- Population is a heavily right skewed normal distribution
- The true population average commute time is 30 minutes
- The true population standard deviation is 15 minutes
- Our sample size is 20 observations

In this world, we see that the true mean parameter is 0.5 hours. This is consistent with our initial speculation, therefore whichever test we use will hopefully fail to reject the null hypothesis.


```python
# Calculate the skewnorm arguments in order to acheive the desired 
# mean and standard deviation for our simulated population
# https://en.wikipedia.org/wiki/Skew_normal_distribution

# Desired population parameters
σ2 = 15**2 # Variance
μ = 30     # Mean
α = 10     # Shape

# Calculated arguments for skewnorm function
# They are a function of our population parameters
δ = α/(1+α**2)**(1/2)
ω = (σ2/(1-(2*(δ**2)/np.pi)))**(1/2)  # scale
ξ = μ - ω*δ*(2/np.pi)**(1/2)          # location
```


```python
# Here we simulate our population with an arbitrarily large N
# The population distribution is a right skewed normal distribution
# The shape, location, and scale have been calculated above.

np.random.seed(12345)

# Simulate population
pop_exp1 = stats.skewnorm.rvs(a = α, loc = ξ, scale = ω, size = 100000)


# Plot population
ax = sns.distplot(pop_exp1)
plt.axvline(x = np.mean(pop_exp1), color = "black")
plt.text(32, 0.025, 
         "μ: " + str(round(np.mean(pop_exp1),2)) + "\nσ: " + str(round(np.std(pop_exp1),2)), 
         horizontalalignment = 'left')
ax.set_title("Population Distribution - Simulation 1")
ax.set_xlabel("Commute Time (min)");
```

    /Users/borisshabash/opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)



    
![png](output_57_1.png)
    



```python
np.random.seed(6)

# Sampling our population we get 20 random observations
sample_exp1 = np.random.choice(pop_exp1, size = 20)

# It doesn't look very normal...
sns.distplot(sample_exp1, kde = False, bins = 30);
```

    /Users/borisshabash/opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)



    
![png](output_58_1.png)
    



```python
# We can also test the Normality by looking at a probability plot
probplot = stats.probplot(sample_exp1, plot = sns.mpl.pyplot)
```


    
![png](output_59_0.png)
    


The Q-Q plot doesn't look particularly normal. We can conclude that this is a non-normal sample. 

#### Student t-test

The one-sample student t-test assumes the mean of the sample follows a normal distribution. Following the same procedure we will evaluate the Type 1 error rate for the t-test under the same conditions. 


```python
stats.ttest_1samp(sample_exp1, popmean = 30)
```




    Ttest_1sampResult(statistic=0.007574231749496885, pvalue=0.9940356584152826)



We have calculated our p-value from the above hypothesis test. Using a 5% threshold we fail to reject the null hypothesis as there is not enough evidence to suggest people in Vancouver have a shorter or longer commute than 30 minutes.

It is nice to see our hypothesis test reached the correct conclusion!  However, we broke the normality assumption. Is it possible we got lucky with our random sample? After all, can we really measure the performance of the t-test through this single sample? This is where we can take full advantage of our theoretical world and simulate several hundred samples drawn from our population to conduct our t-test on. 

To measure the performance of our t-test we will look at the Type 1 error rate. We created a world where we should fail to reject the null. If we set the $\alpha$ to 0.05 we consequently expect a Type 1 error rate of 0.05. Through a simulation experiment we can see if our desired Type 1 error rate is the same as our actual Type 1 error rate.


```python
np.random.seed(3)

# Conduct t-test on thousands of samples
t_pvalues = []

for i in range(10000):
    sample = np.random.choice(pop_exp1, size = 20)
    t_stat, t_pvalue = stats.ttest_1samp(sample, popmean = 30)
    t_pvalues.append(t_pvalue) 

# Calculated Actual Type 1 Error
np.mean(np.array(t_pvalues) < 0.05)
```




    0.0604



The actual Type 1 error rate is closer to 6% with the t-test, not 5% as we assume when setting the significance level! 

#### Bootstrap-t Technique

A powerful alternative to parametric statistics is Bootstrapping. This is a type of simulation based inference that takes advantage of using computers to generate the sampling distribution. What makes this approach so powerful is that it does not need to make any assumptions about the population.

The procedure for conducting a one sample hypothesis test for the mean via bootstrap-t is as follows:
1. Take a random sample of size n of the population
2. Shift the sample to have a mean consistent with the Null Hypothesis and calculate many bootstrap samples from it
3. For each bootstrap sample, calcualte a t-test statistic and construct the bootstrap-t distribution
4. Calculate the p-value using the original sample t-test statistic and the bootstrap-t distribution. 
6. Reject or Fail to Reject the Null Hypothesis given the p-value and desired Type 1 error threshold. 




```python
# Step 1: Original Sample
ax = sns.distplot(sample_exp1, kde = False, bins = 30)
ax.axvline(x = np.mean(sample_exp1), color = "black")
ax.text(26.25,3, " "+str(round(np.mean(sample_exp1),2)),
        horizontalalignment='left')
ax.set_title("Sample Distribution")
ax.set_xlabel("Commute Time (min)");
```

    /Users/borisshabash/opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)



    
![png](output_67_1.png)
    



```python
# Step 2: Shift the sample distribution to the Null Hypothesis
#         And calculate many bootstraps

sample_exp1_null = sample_exp1 - (np.mean(sample_exp1) - 30)
BT_mean = []

ax = sns.distplot(sample_exp1_null, kde = False, bins = 30)
ax.axvline(x = np.mean(sample_exp1_null), color = "black")
ax.text(30,3, " "+str(round(np.mean(sample_exp1_null),2)),
        horizontalalignment='left')
ax.set_title("Sample Distribution - Shifted to Null H")
ax.set_xlabel("Commute Time (min)")

# Visualize the first 5 bootstraps
fig, axs = plt.subplots(1,5, figsize = (10,4))
for i in range(5):
    BT_sample = np.random.choice(sample_exp1_null, 
                                 size = len(sample_exp1_null), 
                                 replace = True)
    
    mean = np.mean(BT_sample)
    BT_mean.append(mean)

    ax = axs[i]
    sns.distplot(BT_sample, kde = False, bins = 30, ax = ax)
    ax.axvline(x = mean, color = "black")
    ax.text(0.75,0.75, " "+str(round(mean,2)),
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
    ax.set_title("Bootstrap"+str(i+1))

```


    
![png](output_68_0.png)
    



    
![png](output_68_1.png)
    



```python
# Step 3: Calculate t statistic for each bootstrap
#         and create new t distribution

np.random.seed(123)

Tstats = []
for i in range(5000):
    BT_sample = np.random.choice(sample_exp1_null, 
                                 size = len(sample_exp1_null),
                                 replace = True)

    Tstat, pvalue = stats.ttest_1samp(BT_sample, popmean = 30)
    Tstats.append(Tstat)

# Visualize bootstrap t distribution compared to theoretical
fig, ax = plt.subplots(1, 1)

df = len(sample_exp1_null) - 1
x = np.linspace(stats.t.ppf(0.001, df),
                stats.t.ppf(0.999, df), 1000)
ax.plot(x, stats.t.pdf(x, df),
        'r-', lw=1, label='Theoretical t')

sns.distplot(Tstats, hist = False, label = "Boostrap t", ax = ax)
ax.legend(loc='best', frameon=False);

```

    /Users/borisshabash/opt/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
      warnings.warn(msg, FutureWarning)



    
![png](output_69_1.png)
    



```python
# Step 4: Calculate p value and Perform decision rule

tstat, pvalue = stats.ttest_1samp(sample_exp1, popmean=30)

if tstat <= 0:
    BT_pvalue = 2*np.mean(Tstats <= tstat)
else:
    BT_pvalue = 2*np.mean(Tstats > tstat)

BT_pvalue = min(BT_pvalue, 1)
BT_pvalue
```




    0.94



Lastly, we can conduct our decision rule. From the above p-value we Fail to Reject the Null Hypothesis. This is the conclusion we were hoping to get! Similar to the t-test, let's see how this Bootstrap-t hypothesis test holds up on many samples.


```python
np.random.seed(123)

## LONG RUN TIME
# Calculate actual Type 1 error rate using Boostrap T approach
BT_pvalues = []

# Simulate thousands of samples from population
for j in range(1000):

    sample = np.random.choice(pop_exp1, size = 20)
    sample_null = sample - (np.mean(sample) - 30)

    # Calculate Bootstrap t statistic 
    Tstats = []
    
    for i in range(5000):
        BT_sample = np.random.choice(sample_null, 
                                     size = len(sample_null),
                                     replace = True)

        Tstat, pvalue = stats.ttest_1samp(BT_sample, popmean = 30)
        Tstats.append(Tstat)

    tstat, pvalue = stats.ttest_1samp(sample, popmean=30)

    # Calculate p value for each sample
    if tstat <= 0:
        BT_pvalue = 2*np.mean(Tstats <= tstat)
    else:
        BT_pvalue = 2*np.mean(Tstats > tstat)

    BT_pvalue = min(BT_pvalue, 1)
    BT_pvalues.append(BT_pvalue)
    print(f'{j+1} trials finished', end="\r")
    
```

    1000 trials finished


```python
np.mean(np.array(BT_pvalues) < 0.05)
```




    0.044



Here we see that with a Type 1 error rate of 4.4%, the Bootstrap-t method is the one that gets us closest to our expected Type 1 error rate of 5%. And it is on the more conservative end, that is we will make less type I errors than we budget for in our experiment. 

<div id="container" style="position:relative;">
<div style="position:relative; float:right"><img style="height:25px""width: 50px" src ="https://drive.google.com/uc?export=view&id=14VoXUJftgptWtdNhtNYVm6cjVmEWpki1" />
</div>
</div>
