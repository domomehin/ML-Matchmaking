# Library Imports
from joblib import load
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import _pickle as pickle
from random import sample, randint
from PIL import Image
from scipy.stats import halfnorm
from flask import Flask, request

app = Flask(__name__)


# Loading the Profiles
with open("refined_profiles.pkl",'rb') as fp:
    df = pickle.load(fp)
    
with open("refined_cluster.pkl", 'rb') as fp:
    cluster_df = pickle.load(fp)
    
with open("vectorized_refined.pkl", 'rb') as fp:
    vect_df = pickle.load(fp)
    
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
 
    
def vectorization(df, columns):
    """
    Using recursion, iterate through the df until all the categories have been vectorized
    """
    column_name = columns[0]
    
    # Checking if the column name has been removed already
    if column_name not in ['Profiles', 'Style', 'Gender']:
        return df
    
    else:
        # Instantiating the Vectorizer
        vectorizer = CountVectorizer()
        
        # Fitting the vectorizer to the Bios
        x = vectorizer.fit_transform(df[column_name])

        # Creating a new DF that contains the vectorized words
        df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())

        # Concating the words DF with the original DF
        new_df = pd.concat([df, df_wrds], axis=1)

        # Dropping the column because it is no longer needed in place of vectorization
        new_df = new_df.drop(column_name, axis=1)
        
        return vectorization(new_df, new_df.columns) 

    
def scaling(df, input_df):
    """
    Scales the new data with the scaler fitted from the previous data
    """
    scaler = MinMaxScaler()
    
    scaler.fit(df)
    
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
    top_10_sim = corr.sort_values(ascending=False)[1:11]
        
    # The Top Profiles
    top_10 = df.loc[top_10_sim.index]
        
    # Converting the floats to ints
    top_10[top_10.columns[1:]] = top_10[top_10.columns[1:]]
    
    return top_10.astype('object')


def example_bios():
    """
    Creates a list of random example bios from the original dataset
    """
    # Example Bios for the user
    st.write("-"*100)
    st.text("Some example Bios:\n(Try to follow the same format)")
    for i in sample(list(df.index), 3):
        st.text(df['Bios'].loc[i])
    st.write("-"*100)

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
# gender = pd.DataFrame(columns=["Gender"])
gender = []
for i in range(df.shape[0]):
    # Range of numbers to represent different labels in each category
    number = randint(0, 2)
    if number == 1:
      gender.append("Female")
    elif number == 2:
      gender.append("Non-Binary")
    else:
      gender.append("Male")
gender = pd.DataFrame(gender, columns=["Gender"])

final_categories = [style_types, age, gender]
names = ["Style", "Age", "Gender"]
combined = dict(zip(names, final_categories))
    
@app.route('/connections', methods = 'GET')
def connections():
    # need interests
    # cost 
    # gender
    interests = request.args.get("interests")
    rate = request.args.get("rate")
    # find the stylists that match to them 
    # add column in style seeker that is a list of stylists that they have connected to
    # THis is a many to many relationship.
    # maybe store it as a list of stylists that belong to a styelseekers
    # list of connections from most to least
    connections = []
    new_profile = pd.DataFrame(columns=df.columns, index=[df.index[-1]+1])

    # Asking for new profile data
    new_profile['Profiles'] = interests
    new_profile['Rate'] = rate
    new_profile['Gender'] = gender
    new_profile['Age'] = age
    for col in df.colums:
        df[col] = df[col].apply(string_convert)
        new_profile[col] = new_profile[col].apply(string_convert)
        -[[[[]]]]
    # top 10 matches using the the newest data in the dataframe
    # Vectorizing the New Data
    df_v, input_df = vectorization(df, df.columns, new_profile)
                
    # Scaling the New Data
    new_df = scaling(df_v, input_df)
                
    # Predicting/Classifying the new data
    cluster = model.predict(new_df)
        
    # Finding the top 10 related profiles 
    #### TO DO: make this top till the dataframe ends
    connections = top_ten(cluster, vect_df, new_df)
    return connections 
    

## Interactive Section

if __name__ == '__main__':
    app.run()       

    

