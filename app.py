import pickle as pk
import streamlit as st
import pandas as pd

# Title
st.title('Sentiment analysis App')

# Load the trained model
with open('sentiment.pickle', 'rb') as dbfile:
    model = pk.load(dbfile)

# User Inputs
review = st.text_input("Enter your review = " )
review_data = {'predict_sentiment':[review]}
review_data_df = pd.DataFrame(review_data)


# On button click
if st.button("Predict"):

    # Create input DataFrame with correct column names
    df = pd.DataFrame({
        'review':[review]
              })
     
    st.dataframe(df)

    # Make prediction
    result = model.predict(review_data_df['predict_sentiment'])[0]
    if int(result) == 0:
        result = "Negative"
    else:
        result ="positive"  
    print(result)   
    st.write(result)  

    # Show result
    st.write("succes!")
    st.balloons()
    st.snow()
    st.success("Prediction successful!")
