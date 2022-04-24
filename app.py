# Library Imports
from doctest import DocFileCase
import tempfile
from joblib import load
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
# import streamlit as st
import _pickle as pickle
from random import random, sample, randint
from PIL import Image
from scipy.stats import halfnorm
from flask import Flask, request, abort, jsonify
from statistics import mean
import os
from mongo_db_integration import get_stylist_from_id

app = Flask(__name__)


# Loading the Profiles
path = os.getcwd() + "/Pickles/refined_profiles.pkl"
with open(path,'rb') as fp:
    df = pickle.load(fp)

path = os.getcwd() + "/Pickles/refined_cluster.pkl"    
with open(path, 'rb') as fp:
    cluster_df = pickle.load(fp)

path = os.getcwd() + "/Pickles/vectorized_refined.pkl"      
with open(path, 'rb') as fp:
    vect_df = pickle.load(fp)
    
path = os.getcwd() + "/Pickles/mapping.pkl"
with open(path, 'rb') as fp:
    mapping = pickle.load(fp)
    
# Loading the Classification Model
model = load("refined_model.joblib")

## Helper Functions

def string_convert(x):
    """
    First converts the lists in the DF into strings
    """
    if isinstance(x, list):
        return ' '.join(x)
    else:
        return x
 
count = 0
def vectorization(df, columns, input_df):
    """
    Using recursion, iterate through the df until all the categories have been vectorized
    """
    column_name = columns[0]
    
    # Checking if the column name has been removed already
    if column_name not in ['Profiles', 'Style']:
        return df, input_df
    
    else:
        # Instantiating the Vectorizer
        vectorizer = CountVectorizer()
        # print(count)
        

        # Creating a new DF that contains the vectorized words
    
        # X = stylist 
        x = vectorizer.fit_transform(df[column_name])
        df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out(), index= df.index)
        # Concating the words DF with the original DF
        df = df.loc[~df.index.duplicated(keep='first')]
        df_wrds = df_wrds.loc[~df_wrds.index.duplicated(keep='first')]
        new_df = pd.concat([df, df_wrds], axis=1)

        # Dropping the column because it is no longer needed in place of vectorization
        new_df = new_df.drop(column_name, axis=1)
        new_df = new_df.dropna()
        new_df = new_df.reset_index(drop=True)
        
        # Y = Seeker
        y = vectorizer.transform(input_df[column_name])
        y_wrds = pd.DataFrame(y.toarray(), columns=vectorizer.get_feature_names_out(), index=input_df.index)
        y_df = pd.concat([input_df, y_wrds], 1)
        y_df = y_df.drop(column_name, 1)
        
        print(y_wrds.shape)
        
        return vectorization(new_df, new_df.columns, y_df)

    
def scaling(df, input_df):
    """
    Scales the new data with the scaler fitted from the previous data
    """
    scaler = MinMaxScaler()
    
    scaler.fit(df)
    # print(input_df.shape)
    input_vect = pd.DataFrame(scaler.transform(input_df), index=input_df.index, columns=input_df.columns)
        
    return input_vect
    


def top_ten(cluster, vect_df, input_vect):
    """
    Returns the DataFrame containing the top 10 similar profiles to the new data
    """
    # Filtering out the clustered DF
    des_cluster = vect_df[vect_df['Cluster #']==cluster[0]].drop('Cluster #', 1)
    
    # Appending the new profile data
    des_cluster = des_cluster.append(input_vect, sort=False)
        
    # Finding the Top 10 similar or correlated users to the new user
    user_n = input_vect.index[0]
    
    # Trasnposing the DF so that we are correlating with the index(users) and finding the correlation
    corr = des_cluster.T.corrwith(des_cluster.loc[user_n])

    # Creating a DF with the Top 10 most similar profiles
    top_10_sim = corr.sort_values(ascending=False)[1:]
        
    # The Top Profiles
    top_10 = df.loc[top_10_sim.index]
        
    # Converting the floats to ints
    top_10[top_10.columns[1:]] = top_10[top_10.columns[1:]]
    
    return top_10.astype('object')


## Creating a List for each Category

# Probability dictionary
p = {}
style_types = ["Classic", "Elegant", "Dramatic", "Feminine", "Sexy", "Masculine", 
               "Romantic", "Casual", "Streetwear", "Glam", "Minimalist", "Vintage", 
               "Boho", "Editorial", "Androgynous", "Edgy", "Preppy", "Maximalist"]

p['style'] = [0.11, 0.10, 0.07, 0.13, 0.04, 0.06,
              0.06, 0.04, 0.01, 0.01, 0.10, 0.03,
              0.03, 0.07, 0.02, 0.04, 0.01, 0.07]

# Age (generating random numbers based on half normal distribution)
age = halfnorm.rvs(loc=18,scale=8, size=df.shape[0]).astype(int)
# gender = pd.DataFrame(gender, columns=["Gender"])
rate = halfnorm.rvs(loc=20,scale=50, size=df.shape[0]).astype(int)

final_categories = [style_types, age, rate]
# names = ["Style", "Age", "Gender", "Rate"]
names = ["Style", "Age", "Rate"]
combined = dict(zip(names, final_categories))
    
@app.route('/connections', methods = ['GET'])
def connections():
    # need interests
    # cost 
    interests = request.get_json()["interests"]
    # find the stylists that match to them 
    # add column in style seeker that is a list of stylists that they have connected to
    # THis is a many to many relationship.
    # maybe store it as a list of stylists that belong to a styelseekers
    # list of connections from most to least
    path = os.getcwd() + "/Pickles/refined_profiles.pkl"
    with open(path,'rb') as fp:
        df = pickle.load(fp)
    connections = []
    new_profile = pd.DataFrame(columns=df.columns, index=[df.index[-1]+1])

    # Asking for new profile data
    if not interests:
        abort(400, description="Please answer the questions displayed")

    ages = {"less than 18": 0, "18 to 34": 1, "35 to 50": 2, "51 to 69": 3, "greater than 69": 4}
    clothing = {"Casual/Everyday": "Everyday", "Work": "Work", "Dressy": "Dressy", "Seasonal Update": "Seasonal", "Special Event": "Special"}

    age = ages[interests[0]]
    rate = interests[1].replace("$", "")
    rate = rate.replace("+", "")
    rate = rate.replace(" ", "")
    rate = rate.split("-")
    rate = list(map(lambda x: int(x), rate))
    rate = round(mean(rate)) if len(rate) > 1 else rate
    final_interests = []
    final_interests.append(clothing[interests[2]])
    final_interests.append(interests[3])
    temp = ', '.join([str(interest) for interest in final_interests])
    new_profile['Profiles'] = temp
    new_profile['Rate'] = rate
    new_profile['Age'] = age
    style_temp = temp.replace(',', '')
    new_profile['Style'] = style_temp
        
    for col in df.columns:
        df[col] = df[col].apply(string_convert)
        new_profile[col] = new_profile[col].apply(string_convert)
    # top 10 matches using the the newest data in the dataframe
    # Vectorizing the New Data
    # print(df.shape)
    df = df[['Profiles', 'Style', 'Age', 'Rate']]
    df_v, input_df = vectorization(df, df.columns, new_profile)
    df_v = df_v.loc[:,~df_v.columns.duplicated()]  
    input_df = input_df.loc[:,~input_df.columns.duplicated()]  
    
    # Scaling the New Data
    new_df = scaling(df_v, input_df)
                
    # Predicting/Classifying the new data
    cluster = model.predict(new_df)
    
        
    # Finding the top 10 related profiles 
    matches = top_ten(cluster, vect_df, new_df) # everything in cluster
    
    # matches everything in cluster, mapping is all the mongodb stuff
    matches = matches.loc[matches.index.isin(mapping['index'])] # everything in cluster & is in mapping
    # print(matches)
    leftovers = vect_df[vect_df.index.isin(mapping['index'])]
    leftovers = leftovers.loc[leftovers['Cluster #'] != cluster[0]]

    indeces = np.concatenate((np.array(matches.index), np.array(leftovers.index)))
    ids = [mapping[mapping['index'] == ind].iloc[0]['id'] for ind in indeces]
    profiles = []
    for id in ids:
        profiles.append(get_stylist_from_id(id))
    
    return jsonify(profiles)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
