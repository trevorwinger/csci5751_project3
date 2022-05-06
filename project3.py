# Databricks notebook source
# MAGIC %md #General Info

# COMMAND ----------

# MAGIC %md 
# MAGIC Trevor Winger <br/>
# MAGIC University of Minnesota<br/>
# MAGIC CSCI 5751 - Big Data Engineering<br/>
# MAGIC Chad Dvoracek

# COMMAND ----------

# MAGIC %md ##Data Details

# COMMAND ----------

# MAGIC %md Data came from one source: 
# MAGIC <ul>
# MAGIC   <li>New York Times; the data is availble from their GitHub [here](https://github.com/nytimes/covid-19-data).</li>
# MAGIC </ul>

# COMMAND ----------

# MAGIC %md ##Description of Notebook

# COMMAND ----------

# MAGIC %md 
# MAGIC <ul>
# MAGIC   <li>Python library imports</li>
# MAGIC   <li>Download data from s3 to pyspark dfs</li>
# MAGIC   <li>Clean and validate data</li>
# MAGIC   <li>Data wrangling </li>
# MAGIC   <li>Statistical Analysis</li>
# MAGIC   <li>Visualization</li>
# MAGIC   <li>Prep for Machine Learning</li>
# MAGIC   <li>Machine Learning Model Development</li>
# MAGIC   <li>Machine Learning Results</li>
# MAGIC   <li>Learned Feature Visualization</li>
# MAGIC </ul>

# COMMAND ----------

# MAGIC %md #Python Libraries

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import approx_count_distinct, avg, stddev, sum, max, min, lit
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)
import seaborn as sns

#sklearn specific
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# COMMAND ----------

# MAGIC %md #Data Importing

# COMMAND ----------

#set this to true if you want to see the .show(n=10) for each df, else set to False. 
#Useful to see interediate transactions for testing, in production it will significantly slow down notebook
logging = False

# COMMAND ----------

# MAGIC %md ##Helper Functions

# COMMAND ----------

def log_df(df):
    if logging:
        #use the pandas because it looks better
        df.toPandas().head(10)

# COMMAND ----------

def s3_to_df(file_name):
    '''
    Helper function for taking s3 file, and making it a pyspark dataframe
    Infer schema so I can demonstrate some cleaning functionality in other cells
    '''
    df = spark.read.format('csv').options(header='true', inferSchema='true').load(file_name)
    log_df(df)
    return df

# COMMAND ----------

# MAGIC %md ##New York Times Data

# COMMAND ----------

#state level data
state_df = s3_to_df('s3://winge159project3/us-states.csv')

#append county data on fps
counties_2020 = s3_to_df('s3://winge159project3/us-counties-2020.csv')
counties_2021 = s3_to_df('s3://winge159project3/us-counties-2021.csv')
counties_2022 = s3_to_df('s3://winge159project3/us-counties-2022.csv')
county_df = counties_2020.union(counties_2021)
county_df = county_df.union(counties_2022)

#mask usage
mask_df = s3_to_df('s3://winge159project3/mask-use/mask-use-by-county.csv')

#college data
col_df = s3_to_df('s3://winge159project3/colleges/colleges.csv')

#correctional facilities
facilities_df = s3_to_df('s3://winge159project3/prisons/facilities.csv')

# COMMAND ----------

# MAGIC %md #Clean & Validate Data

# COMMAND ----------

# MAGIC %md ##Data Prep

# COMMAND ----------

#rename columns in ny times data to not be duplicates
state_df = state_df.withColumnRenamed('cases', 'state_cases')
state_df = state_df.withColumnRenamed('deaths', 'state_deaths')

county_df = county_df.withColumnRenamed('cases', 'county_cases')
county_df = county_df.withColumnRenamed('deaths', 'county_deaths')

col_df = col_df.withColumnRenamed('cases', 'college_cases')


# COMMAND ----------

# MAGIC %md #Data Wrangling & Conditioning

# COMMAND ----------

# MAGIC %md ##Joins

# COMMAND ----------

#mask need to inner join with a set of mn_county_df to get masks 
distinct_counties = county_df.select('fips').distinct()
masks_df = mask_df.join(distinct_counties, distinct_counties.fips == mask_df.COUNTYFP, 'inner')
masks_df = masks_df.drop('COUNTYFP')


#create a state by x dfs of new york times data for each date with each longitudinal dataset
county_df = county_df.join(masks_df, ['fips'])

state_x_county_df = state_df.join(county_df, ['state', 'date'], 'inner')
state_x_college_df = state_df.join(col_df, ['state', 'date'], 'inner')


#join mask sentiment with our state by county df
#state_x_county_df = state_x_county_df.join(masks_df, ['fips'])
log_df(state_x_county_df)
log_df(masks_df)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ##Aggregation & Validation

# COMMAND ----------

# MAGIC %md ###College Aggregation Analysis

# COMMAND ----------

#state x college level analysis
def agg_stats_college(df):
    print('*'*100)
    #have to select distinct colleges 
    print('total # of cases in college', df.select(sum(df.college_cases)).collect()[0][0])
    print('percent of cases in college', round(df.select(sum(df.college_cases)).collect()[0][0]/ df.select(max(df.state_cases)).collect()[0][0], 4)* 100, '%')
    print('total colleges measured', df.select(approx_count_distinct(df.college)).collect()[0][0])
    print('average cases each day reported for colleges', df.select(avg(df.college_cases)).collect()[0][0])
    print('range in cases each day for colleges', df.select(max(df.college_cases)-min(df.college_cases)).collect()[0][0])
    print('*'*100)

# COMMAND ----------

if logging:
    #run on entire country dataset
    agg_stats_college(state_x_college_df)

    #run on just MN
    agg_stats_college(state_x_college_df.filter(state_x_college_df.state == 'Minnesota'))

# COMMAND ----------

if logging:
    #run this computation for every state in system
    for s in state_x_college_df.select(state_x_college_df.state).distinct().collect():
        print(s.state)
        agg_stats_college(state_x_college_df.filter(state_x_college_df.state == s.state))



# COMMAND ----------

# MAGIC %md ###County Aggregation Analysis

# COMMAND ----------

def agg_county(df):
    print('*'*100)
    print('total # of cases in county', df.select(max(df.county_cases)).collect()[0][0])
    print('total recorded # cases in state(s)', df.select(max(df.state_cases)).collect()[0][0])
    print('percent of cases in county', round(df.select(max(df.county_cases)).collect()[0][0]/ df.select(max(df.state_cases)).collect()[0][0], 4)* 100, '%')
    print('average cases each day reported for county', df.select(avg(df.county_cases)).collect()[0][0])
    print('average cases each day reported for state', df.select(avg(df.state_cases)).collect()[0][0])
    print('standard deviation in cases each day for county', df.select(stddev(df.county_cases)).collect()[0][0])
    print('standard deviation in cases each day for state(s)', df.select(stddev(df.state_cases)).collect()[0][0])
    print('*'*100)

# COMMAND ----------

if logging:
    #this could be computer on a county x state level for each state; however for free tier limitations I will only demonstrate it on Minnesota counties
    for c in state_x_county_df.filter(state_x_county_df.state == 'Minnesota').select(state_x_county_df.county).distinct().collect():
        print(c.county)
        agg_county(state_x_county_df.filter(state_x_county_df.county == c.county))

# COMMAND ----------

# MAGIC %md ##Created Functions

# COMMAND ----------


def county_lambda(x):
    #perform some computations on our dataset; will be passes a single row and return new row for our transformed df
    date = x.date
    state = x.state
    county = x.county
    percent_at_day_c = x.county_cases / x.state_cases
    percent_at_day_d = 0
    if x.state_deaths != 0:
        percent_at_day_d = x.county_deaths / x.state_deaths
    n = x.NEVER
    r = x.RARELY
    s = x.SOMETIMES
    f = x.FREQUENTLY
    a = x.ALWAYS
    
    return (date, state, county, percent_at_day_c, percent_at_day_d, n, r, s, f, a)

# COMMAND ----------

def college_lambda(x):
    #perform some computations on our college df
    date = x.date
    state = x.state
    college = x.college
    percent_at_day = x.college_cases / x.state_cases        
    return (date, state, college, percent_at_day)

# COMMAND ----------

def facility_lambda(x):
    #perform statistic computation on our prision df and transform to new one row for row
    state = x.facility_state
    pop = x.latest_inmate_population
    
    percent_pop_cases = 0
    percent_pop_dead = 0
    
    if pop != 0:
        percent_pop_cases = x.total_inmate_cases / pop
        percent_pop_dead = x.total_inmate_deaths / pop
        
    percent_cases_2_dead_inmate = 0
    
    if x.total_inmate_cases != 0:
        percent_cases_2_dead_inmate = x.total_inmate_deaths / x.total_inmate_cases
   
    return (state, round(percent_pop_cases, 4), round(percent_pop_dead, 4), round(percent_cases_2_dead_inmate, 4))

# COMMAND ----------

# MAGIC %md ###Mapping Features Daily

# COMMAND ----------

#mapping for our county x state
clean_c_x_s_df = state_x_county_df.rdd.map(lambda x: county_lambda(x)).toDF(['date', 'state', 'county', 'percent_of_cases', 'percent_of_death', 'never', 'rarely', 'sometimes', 'frequently', 'always'])
log_df(clean_c_x_s_df)

# COMMAND ----------

#mapping for our college x state
clean_col_x_s_df = state_x_college_df.rdd.map(lambda x: college_lambda(x)).toDF(['date', 'state', 'college', 'percent_of_cases'])
log_df(clean_col_x_s_df)

# COMMAND ----------

#mapping four our facility x state
clean_facil_df = facilities_df.rdd.map(lambda x: facility_lambda(x)).toDF(['state', 'percent_of_population_cases', 'percent_of_population_death', 'percent_death_covid_inmate']) 
log_df(clean_facil_df)
clean_facil_df.show(n=10)

# COMMAND ----------

# MAGIC %md ###Computing Summary Statistics

# COMMAND ----------

#compute some summary statistics to show volatility of covid cases aggregate averages and stddevs of all the counties at the state level
#the alias are for neateness, not necessary for analysis
temp_state_df = clean_c_x_s_df.groupBy(clean_c_x_s_df.state)\
.agg(
    avg(clean_c_x_s_df.percent_of_cases).alias('county_avg_cases'), 
    stddev(clean_c_x_s_df.percent_of_cases).alias('county_stddev_cases'),
    avg(clean_c_x_s_df.percent_of_death).alias('county_avg_death'), 
    stddev(clean_c_x_s_df.percent_of_death).alias('county_stddev_death'), 
    avg(clean_c_x_s_df.never).alias('never_avg'), 
    stddev(clean_c_x_s_df.never).alias('never_stddev'), 
    avg(clean_c_x_s_df.rarely).alias('rarely_avg'), 
    stddev(clean_c_x_s_df.rarely).alias('rarely_stddev'), 
    avg(clean_c_x_s_df.sometimes).alias('sometimes_avg'), 
    stddev(clean_c_x_s_df.sometimes).alias('sometimes_stddev'), 
    avg(clean_c_x_s_df.frequently).alias('freq_avg'), 
    stddev(clean_c_x_s_df.frequently).alias('freq_stddev'), 
    avg(clean_c_x_s_df.always).alias('always_avg'), 
    stddev(clean_c_x_s_df.always).alias('always_stddev'))

log_df(temp_state_df)

# COMMAND ----------

#compute some summary statistics to show volatility of covid cases aggregate averages and stddevs of all the colleges at the state level
#the alias are for neateness, not necessary for analysis
temp_col_df = clean_col_x_s_df.groupBy(clean_col_x_s_df.state)\
.agg(
    avg(clean_col_x_s_df.percent_of_cases).alias('avg_cases_college'), 
    stddev(clean_col_x_s_df.percent_of_cases).alias('stddev_cases_college'))
log_df(temp_col_df)

# COMMAND ----------

#compute some summary statistics to show volatility of covid cases aggregate averages and stddevs of all the facility at the state level
#the alias are for neateness, not necessary for analysis
clean_facil_df = clean_facil_df.na.drop('all')


temp_facil_df = clean_facil_df.groupBy(clean_facil_df.state)\
.agg(
    avg(clean_facil_df.percent_of_population_cases).alias('avg_perc_pop_cases').cast('double'),
    stddev(clean_facil_df.percent_of_population_cases).alias('stddev_perc_pop_cases').cast('double'),
    avg(clean_facil_df.percent_of_population_death).alias('avg_perc_pop_deaths').cast('double'),
    stddev(clean_facil_df.percent_of_population_death).alias('stddev_perc_pop_deaths').cast('double'),
    avg(clean_facil_df.percent_death_covid_inmate).alias('avg_perc_covid_deaths').cast('double'),
    stddev(clean_facil_df.percent_death_covid_inmate).alias('stddev_perc_covid_deaths').cast('double'))

log_df(temp_facil_df)

# COMMAND ----------

#join our dfs
final_df = temp_col_df.join(temp_state_df, ['state'])
#final_df = final_df.join(temp_facil_df, ['state'])
log_df(final_df)

# COMMAND ----------

# MAGIC %md #Preparing Data for ML Consumption

# COMMAND ----------

#label data to binary classification based on volatility of death day-to-day at the state level calc diffs from day-to-day in deaths and take the avg of diffs

def state_death_avg(df):
    df = df.toPandas()
    t_df = df.groupby(['state'])
    
    res = []
        
    for name, group in t_df:
        r = []
        r.append(name)
        m = group['state_deaths'].diff().mean() / group['state_cases'].diff().mean()
        if m != None:
            r.append(group['state_deaths'].diff().mean())
            res.append(r)
    
    df = spark.createDataFrame(pd.DataFrame(res, columns=['state', 'deaths']))
    df = df.na.drop('all')
    log_df(df)
    return df



# COMMAND ----------

#compute our labeling feature
label_df = state_death_avg(state_df)

#join label and final_df 
final_df = final_df.join(label_df, ['state'], 'inner')
log_df(final_df)

# COMMAND ----------

def label_lambda(x, avg_deaths):
    label = 0
    if x.deaths > avg_deaths:
        label = 1
    return (
    x.state, x.avg_cases_college, x.stddev_cases_college,
    x.county_avg_cases, x.county_stddev_cases, x.county_avg_death,
    x.county_stddev_death, x.never_avg, x.never_stddev,
    x.rarely_avg, x.rarely_stddev, x.sometimes_avg,
    x.sometimes_stddev, x.freq_avg, x.freq_stddev,
    x.always_avg, x.always_stddev, label)

# COMMAND ----------

cols = [
    'state', 'avg_cases_college', 'stddev_cases_college',
    'county_avg_cases', 'county_stddev_cases', 'county_avg_death', 
    'county_stddev_death', 'never_avg', 'never_stddev', 'rarely_avg',
    'rarely_stddev', 'sometimes_avg', 'sometimes_stddev',
    'freq_avg', 'freq_stddev', 'always_avg', 
    'always_stddev', 'label']

#get the mean deaths for our label splitting point
mean_deaths = final_df.agg(avg(final_df.deaths)).collect()[0][0]

#label 
f = final_df.rdd.map(lambda x: label_lambda(x, mean_deaths)).toDF(cols)
log_df(f)

# COMMAND ----------

# MAGIC %md #Statistical Analysis

# COMMAND ----------

# MAGIC %md ##General Data Frame Description

# COMMAND ----------

f.toPandas().describe()

# COMMAND ----------

# MAGIC %md ##Pearson Correlations

# COMMAND ----------

f.toPandas().corr(method='pearson').style.background_gradient(cmap='coolwarm').set_precision(2)
corr = f.toPandas().corr(method='pearson')
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

# COMMAND ----------

f.toPandas().corr().style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '0pt'})

# COMMAND ----------

# MAGIC %md #Visualization

# COMMAND ----------

# MAGIC %md ##States by Death Metric

# COMMAND ----------

t = state_death_avg(state_df).toPandas()
best = t.sort_values('deaths', ascending=True).head(10)
worst =t.sort_values('deaths', ascending=False).head(10)

# COMMAND ----------

best.plot.scatter(x='state', y='deaths')

# COMMAND ----------

worst.plot.scatter(x='state', y='deaths')

# COMMAND ----------

# MAGIC %md ##Sentiment Plots

# COMMAND ----------

#Select positive labeled data for this and the first 7
ff = f.filter(f.label == 1).limit(10)

# COMMAND ----------

#avg for never masking sentiment across counties in states we have deemed to have handled covid poorly
ff.toPandas().plot.scatter(x='state', y='never_avg')

# COMMAND ----------

#avg for rarely masking sentiment across counties in states we have deemed to have handled covid poorly
ff.toPandas().plot.scatter(x='state', y='rarely_avg')

# COMMAND ----------

#avg for sometimes masking sentiment across counties in states we have deemed to have handled covid poorly
ff.toPandas().plot.scatter(x='state', y='sometimes_avg')

# COMMAND ----------

#avg for always masking sentiment across counties in states we have deemed to have handled covid poorly
ff.toPandas().plot.scatter(x='state', y='always_avg')

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ##Case Plots

# COMMAND ----------

ff.toPandas().plot.scatter(x='state', y='avg_cases_college')

# COMMAND ----------

# MAGIC %md #Machine Learning Model Development

# COMMAND ----------

'''
Abstract class for our machine learning model
- Call show_models with your prepared df from above, will run all the models and prepare the best one
'''

class ModelComparer:
    def __init__(self):
        self.names = [ "Nearest Neighbors", "Linear SVM", "RBF SVM", 
                      "Gaussian Process", "Decision Tree", "Random Forest",
                      "Neural Net", "AdaBoost", "Naive Bayes", "QDA" ]
        
        self.clfs = [ KNeighborsClassifier(3), SVC(kernel="linear", C=0.025), SVC(gamma=2, C=1),
                     GaussianProcessClassifier(1.0 * RBF(1.0)), DecisionTreeClassifier(max_depth=5), RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                     MLPClassifier(alpha=1, max_iter=1000), AdaBoostClassifier(), GaussianNB(), QuadraticDiscriminantAnalysis()]
        
        
    def stats(self, name, preds, y_true):
        '''
        print & return accuracy, precsion, recall, and f1 scores; 
        given true labels and predicted labels from our trained models
        '''
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds)
        rec = recall_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        print('*'*100)
        print('results for model:', name)
        print('accuracy', acc)
        print('precision', prec)
        print('recall', rec)
        print('f1', f1)
        print('*'*100)
        
        return (acc, prec, rec, f1)
        
        
    def show_models(self, df):
        '''
        pass in pandas df with label feature as 'label'
        '''
        y = df['label']
        df = df.drop(['label', 'state'], axis=1)
        x = df
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
        res = []
        
        for name, clf in zip(self.names, self.clfs):
            clf.fit(x_train, y_train)
            p = clf.predict(x_test)
            s = self.stats(name, p, y_test)
            r = [name]
            r.append(s[0])
            r.append(s[1])
            r.append(s[2])
            r.append(s[3])
            res.append(r)
            
        
        return pd.DataFrame(res, columns=['clf', 'accuracy', 'precision', 'recall', 'f1'])
            
            
            

        

# COMMAND ----------

# MAGIC %md #Machine Learning Results

# COMMAND ----------

# MAGIC %md ##Test Model Set On Different Cut-Off Thresholds

# COMMAND ----------

#get mean of deaths for labeling 
mean_deaths = final_df.agg(avg(final_df.deaths)).collect()[0][0]
stddev_deaths = final_df.agg(stddev(final_df.deaths)).collect()[0][0]

# COMMAND ----------

#cols used for our mapped and labeled df 
cols = [
    'state', 'avg_cases_college', 'stddev_cases_college',
    'county_avg_cases', 'county_stddev_cases', 'county_avg_death', 
    'county_stddev_death', 'never_avg', 'never_stddev', 'rarely_avg',
    'rarely_stddev', 'sometimes_avg', 'sometimes_stddev',
    'freq_avg', 'freq_stddev', 'always_avg', 
    'always_stddev', 'label']

# COMMAND ----------

#mean split for label split
f = final_df.rdd.map(lambda x: label_lambda(x, mean_deaths)).toDF(cols)
mean_c = ModelComparer()
mean_df = mean_c.show_models(f.toPandas())

# COMMAND ----------

#mean - stddev for label split 
f = final_df.rdd.map(lambda x: label_lambda(x, mean_deaths-(stddev_deaths/2))).toDF(cols)
minus_c = ModelComparer()
mean_minus_df = minus_c.show_models(f.toPandas())

# COMMAND ----------

#mean + stddev for label split 
f = final_df.rdd.map(lambda x: label_lambda(x, mean_deaths+(stddev_deaths/2))).toDF(cols)
plus_c = ModelComparer()
mean_plus_df = plus_c.show_models(f.toPandas())

# COMMAND ----------

# MAGIC %md # Visualization of Model Performance

# COMMAND ----------

# MAGIC %md Plot accuracy over recall plots for each of our classifiers

# COMMAND ----------

mean_df.plot.scatter(x='recall', y='accuracy')

# COMMAND ----------

mean_minus_df.plot.scatter(x='recall', y='accuracy')

# COMMAND ----------

mean_plus_df.plot.scatter(x='recall', y='accuracy')

# COMMAND ----------

# MAGIC %md ##Learned Feature Reporting

# COMMAND ----------

# MAGIC %md ##Decision Tree

# COMMAND ----------

# MAGIC %md Plot the decision tree for each of our dataset to validate there is no overfitting going on

# COMMAND ----------

t = f.drop('state')

# COMMAND ----------

plot_tree(mean_c.clfs[4], feature_names=t.columns)

# COMMAND ----------

plot_tree(minus_c.clfs[4],feature_names=t.columns)

# COMMAND ----------

plot_tree(plus_c.clfs[4], feature_names=t.columns)
