# -*- coding: utf-8 -*-
"""
Created on Thu May 26 23:14:57 2022

@author: shrey
"""

import streamlit as st
import finalmodel
import itembasedfn
import userbasedfn


def predict_movies(name: str):
    df = finalmodel.predict(name)
    return df

def predict_moviesitem(movie: str):
    itemdf = itembasedfn.predict(movie)
    return itemdf

def predict_moviesuser(movie: str):
    userdf = userbasedfn.predict(movie)
    return userdf

st.title('Shreyas Movie Recommender')

input_movie=st.text_input('Movie name', " ")

option = st.selectbox(
     'Choose approach',
     ('Returning user - personalise your recommendations', 'Other User also watched', 'More movies like this one'))

if(option=='Returning user - personalise your recommendations'):
    if st.button('Recommend'):
            pf= predict_movies(input_movie)
            for i in range(0, len(pf)):
                st.write(pf[i])
elif(option == 'Other User also watched'):
    if st.button('Recommend'):
        userf= predict_moviesuser(input_movie)
        for i in range(0, len(userf)):
                st.write(userf[i])
else:
    if st.button('Recommend'):
        itemf= predict_moviesitem(input_movie)
        for i in range(0, len(itemf)):
                st.write(itemf[i])
    
        
