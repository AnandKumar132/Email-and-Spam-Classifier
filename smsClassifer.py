import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()



page_bg_img = """
<style>
[data-testid= "stAppViewContainer"] {
background-image: url(https://images.unsplash.com/photo-1477346611705-65d1883cee1e?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwxfDB8MXxyYW5kb218MHx8fHx8fHx8MTY4MTg1MDI0NQ&ixlib=rb-4.0.3&q=80&utm_campaign=api-credit&utm_medium=referral&utm_source=unsplash_source&w=1080);
background-size: 100vw 100vh;
}

[data-testid="stHeader"]{
background-color: rgba(0, 0, 0, 0);
}

[data-testid="stToolbar"]{
right: 2rem;
}
</style> 
"""

st.markdown(page_bg_img, unsafe_allow_html=True)



def text_transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
         y.append(ps.stem(i))
       
    
    
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email and SMS Spam Classifier")

input_sms = st.text_area("Enter message here...")


if st.button('Check'):

    # Preprocess 
    transformed_sms = text_transform(input_sms)
    # Vectorize
    Vector_input = tfidf.transform([transformed_sms])
    # Predict
    result = model.predict(Vector_input)[0]
    # Display
    if result == 1:
        st.header("Spam Message")
        st.container()  # Create a container for spam messages
        spam_box = st.container()
        with spam_box:
            st.markdown(f"**Spam Message:** {input_sms}", unsafe_allow_html=True)
            st.markdown('<style>div.Widget.row-widget.stContainer{background-color: #FF0000;}</style>', unsafe_allow_html=True)
    else:
        st.header("Not Spam")
        st.container()  # Create a container for legitimate messages
        legit_box = st.container()
        with legit_box:
            st.markdown(f"**Legitimate Message:** {input_sms}", unsafe_allow_html=True)
            st.markdown('<style>div.Widget.row-widget.stContainer{background-color: #00FF00;}</style>', unsafe_allow_html=True)