#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

from surprise import SVD, SVDpp, KNNBasic
from surprise import Dataset
from surprise.model_selection import cross_validate,train_test_split, GridSearchCV
from surprise import NormalPredictor
from surprise import Reader

import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk


# In[2]:


nltk.download('stopwords')


# In[3]:


movies = pd.read_csv(r'C:\Users\shrey\OneDrive\Desktop\ML\ml-latest-small\movies.csv')
ratings = pd.read_csv(r'C:\Users\shrey\OneDrive\Desktop\ML\ml-latest-small\ratings.csv')


# In[4]:


ratings_array= ratings['rating'].unique()
max_rating = np.amax(ratings_array)
min_rating = np.amin(ratings_array)
print(ratings_array)


# In[5]:


movie_map = pd.Series(movies.movieId.values,index=movies.title).to_dict()
reverse_movie_map = {v: k for k, v in movie_map.items()}
movieId_to_index_map = pd.Series(movies.index.values,index=movies.movieId).to_dict()
movieId_all_array = movies['movieId'].unique()


# To get movie id that corresponds to the movie name

# In[6]:



def get_movieId (movie_name):
    if (movie_name in movie_map):
        return movie_map[movie_name]
    else:
        similar = []
        for title, movie_id in movie_map.items():
            ratio = fuzz.ratio(title.lower(), movie_name.lower())
            if ( ratio >= 60):
                similar.append( (title, movie_id, ratio ) )
        if (len(similar) == 0):
            print("Movie does not exist")
        else:
            match_item = sorted( similar , key=lambda x: x[2] )[::-1]
            print( "Matched item might be:", match_item[0][0], ", ratio=",match_item[0][2] )
            return match_item[0][1]


# # 3. Content Based Filtering 

# Content-based filtering uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback.

# Here i will be using the TF-IDF pairwise approach in vector space

# In[7]:


def tokenizer(text):
    torkenized = [PorterStemmer().stem(word).lower() for word in text.split('|') if word not in stopwords.words('english')]
    return torkenized


# In[8]:


tfid=TfidfVectorizer(analyzer='word', tokenizer=tokenizer)


# In[9]:


tfidf_matrix = tfid.fit_transform(movies['genres'])


# In[10]:

cos_sim = cosine_similarity(tfidf_matrix,tfidf_matrix)


# In[11]:


tfidf_matrix.shape


# In[12]:


cos_sim.shape


# In[13]:


movies.shape


# # 4. Collaborative Filtering - using svd model

# In[14]:


features = ['userId','movieId', 'rating']
reader = Reader(rating_scale=(min_rating, max_rating))
data = Dataset.load_from_df(ratings[features], reader)
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)


# In[15]:


gs.fit(data)


# In[16]:


gs.best_score['rmse']


# In[17]:


gs.best_params['rmse']


# In[18]:


best_params = gs.best_params['rmse']
model_svd = gs.best_estimator['rmse']
model_svd.fit(data.build_full_trainset())


# In[19]:


def get_rating_from_prediction(prediction, ratings_array):
    rating = ratings_array[np.argmin([np.abs(item-prediction)for item in ratings_array])]
    return rating


# In[20]:


prediction = model_svd.predict(1,1)


# In[21]:


print('rating',ratings[(ratings.userId==1)&(ratings.movieId==1)]['rating'])


# In[22]:


print("prediction",prediction.est)


# # Build hyrbid! 

# # First i'll be building an item based fn

# It will return the top n (10) movie recommendation based on the input movie
# 
# The parameters of the function are:
# 
#  similarity_matrix: pairwise similarity matrix [ 2D ]
# 
#  movieId_all_array:array of all movie Ids [1D]
# 
#  ratings_data: ratings data
# 
#  id_to_movie_map: the map from movieId to movie title
# 
#  movieId_to_index_map: the map from movieId to the index of the movie dataframe
# 
#  inp_movie_list: input list of movies
# 
#  n_recommendations: top n recommendations
# 
#  userId: int optional (default=-99), the user Id
#             if userId = -99, the new user will be created
#             if userId = -1, the latest inserted user is chosen
# 
#     Return:
#     list of top n movie recommendations
# 
#   

# In[23]:


def make_recommendation_item_based( similarity_matrix ,movieId_all_array, ratings_data, id_to_movie_map, movieId_to_index_map, inp_movie_list, n_recommendations, userId=-99):

    if (userId == -99):
        userId = np.amax( ratings_data['userId'].unique() ) + 1
    elif (userId == -1):
        userId = np.amax( ratings_data['userId'].unique() )

    movieId_list = []
    for movie_name in inp_movie_list:
        movieId_list.append( get_movieId(movie_name) )    

    # Get the movie id which corresponding to the movie the user didn't watch before
    movieId_user_exist = list( ratings_data[ ratings_data.userId==userId ]['movieId'].unique() )
    movieId_user_exist = movieId_user_exist + movieId_list
    movieId_input = []
    for movieId in movieId_all_array:
        if (movieId not in movieId_user_exist):
            movieId_input.append( movieId )


    index = movieId_to_index_map[movieId_list[0]]
    cos_sim_scores=list(enumerate(similarity_matrix[index]))
    cos_sim_scores=sorted(cos_sim_scores,key=lambda x:x[1],reverse=True)
    
    topn_movieIndex = []
    icount = 0
    for i in range(len(cos_sim_scores)):
        if( cos_sim_scores[i][0] in [movieId_to_index_map[ids] for ids in movieId_input ]  ):
            icount += 1
            topn_movieIndex.append( cos_sim_scores[i][0] )
        if( icount == n_recommendations ):
            break
    
    topn_movie = [ movies.loc[index].title for index in topn_movieIndex ]
    return topn_movie
    


# # User based fn is next

# It will return top n (10) movie recommendation based on input movie
# The parameters are:
#     
# best_model_params: dict, {'iterations': iter, 'rank': rank, 'lambda_': reg}
# 
#  movieId_all_array: the array of all movie Id
# 
# ratings_data: ratings data
# 
#  id_to_movie_map: the map from movieId to movie title
# 
#  inp_movie_list: list, user's list of favorite movies
# 
# n_recommendations: int, top n recommendations
# 
# userId: int optional (default=-99), the user Id
#             if userId = -99, the new user will be created
#             if userId = -1, the latest inserted user is chosen
# 
#     Return:
#     list of top n movie recommendations
# 

# In[24]:


def make_recommendation_user_based(best_model_params, movieId_all_array, ratings_data, id_to_movie_map,inp_movie_list, n_recommendations, userId=-99 ):


    movieId_list = []
    for movie_name in inp_movie_list:
        movieId_list.append( get_movieId(movie_name) )

    if (userId == -99):
        userId = np.amax( ratings_data['userId'].unique() ) + 1
    elif (userId == -1):
        userId = np.amax( ratings_data['userId'].unique() )

    ratings_array = ratings['rating'].unique()
    max_rating = np.amax( ratings_array )
    min_rating = np.amin( ratings_array )
    
    # create the new row which corresponds to the input data
    user_rows = [[userId, movieId, max_rating] for movieId in movieId_list]
    df = pd.DataFrame(user_rows, columns =['userId', 'movieId', 'rating']) 
    train_data = pd.concat([ratings_data, df], ignore_index=True, sort=False)

    # Get the movie id which corresponding to the movie the user didn't watch before
    movieId_user_exist = train_data[ train_data.userId==userId ]['movieId'].unique()
    movieId_input = []
    for movieId in movieId_all_array:
        if (movieId not in movieId_user_exist):
            movieId_input.append( movieId )

    reader = Reader(rating_scale=(min_rating, max_rating))

    data = Dataset.load_from_df(train_data, reader)

    model = SVD(**best_model_params)
    model.fit(data.build_full_trainset())

    predictions = []
    for movieId in movieId_input:
        predictions.append( model.predict(userId,movieId) )

    
    sort_index = sorted(range(len(predictions)), key=lambda k: predictions[k].est, reverse=True)
    topn_predictions = [ predictions[i].est for i in sort_index[0:min(n_recommendations,len(predictions))] ]
    topn_movieIds = [ movieId_input[i] for i in sort_index[0:min(n_recommendations,len(predictions))] ]
    topn_rating = [ get_rating_from_prediction( pre, ratings_array ) for pre in topn_predictions ]

    topn_movie = [ id_to_movie_map[ ids ] for ids in topn_movieIds ]
    return topn_movie


# In[31]:


def recommendation(inp):
    # get recommendations
    n_recommendations = 5

    recommends_item_based = make_recommendation_item_based( 
        similarity_matrix = cos_sim,
        movieId_all_array = movieId_all_array,
        ratings_data = ratings[features], 
        id_to_movie_map = reverse_movie_map, 
        movieId_to_index_map = movieId_to_index_map,
        inp_movie_list = inp, 
        n_recommendations = n_recommendations)

    recommends_user_based = make_recommendation_user_based(
        best_model_params = best_params, 
        movieId_all_array = movieId_all_array,
        ratings_data = ratings[features], 
        id_to_movie_map = reverse_movie_map, 
        inp_movie_list = inp, 
        n_recommendations = n_recommendations)

    print("Based on items content similarity")
    print('The movies similar to' , inp , ':' )
    for i, title in enumerate(recommends_item_based):
        print(i+1, title)  
    if( len(recommends_item_based) < n_recommendations ):
      print("Couldn't offer recommendations :(")    

    print("Based on similarity between users")
    print('The users like' , inp, 'also like:')
    for i, title in enumerate(recommends_user_based):
        print(i+1, title)
    if( len(recommends_user_based) < n_recommendations ):
      print("Couldn't offer recommendations :(")




# In[44]:


def predict(name:str):
    m_list=[]
    m_list.append(name)
    st=recommendation(m_list)
    return st
    


# In[45]:

# In[ ]:




