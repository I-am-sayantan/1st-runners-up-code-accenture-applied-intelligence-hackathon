
import requests
import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import scipy.spatial
import pandas as pd

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Question Answering Webapp")
st.text("What would you like to know about Amazon today?")

@st.cache(allow_output_mutation=True)
def load_model():
  pickle_in = open("corpus_embeddings.pkl","rb")
  model=pickle.load(pickle_in)
  
  return model

with st.spinner('Loading Model Into Memory....'):
  embedder = SentenceTransformer('bert-base-nli-mean-tokens')
  corpus_embeddings = load_model()
  df_sentences = pd.read_csv("covid_sentences.csv")
  df_sentences = df_sentences.set_index("Unnamed: 0")
  df_sentences = df_sentences["paper_id"].to_dict()
  df_sentences_list = list(df_sentences.keys())
  df_sentences_list = [str(d) for d in df_sentences_list]
  df = pd.read_csv("covid_sentences_Full.csv", index_col=0)
  corpus = df_sentences_list

texts = st.text_input('Enter your questions here..')
texts=[texts]
if texts[0]:
    st.write("Response :")
    query_embeddings = embedder.encode(texts,show_progress_bar=True)
    closest_n = 5
    with st.spinner('Searching for answers.....'):
      for query, query_embedding in zip(texts, query_embeddings):
          distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
          results = zip(range(len(distances)), distances)
          results = sorted(results, key=lambda x: x[1])
          for idx, distance in results[0:closest_n]:
              #st.write("Score:   ", "(Score: %.4f)" % (1-distance) , "\n" )
              st.write("Paragraph:  {}".format(corpus[idx].strip()) )
              row_dict = df.loc[df.index== corpus[idx]].to_dict()
              st.write("paper_id:  {}".format(row_dict["paper_id"][corpus[idx]]))
              st.write("Title:  {}".format(row_dict["title"][corpus[idx]]))
              st.write("Abstract:  {}".format(row_dict["abstract"][corpus[idx]]))
              st.write("Abstract_Summary:  {}".format(row_dict["abstract_summary"][corpus[idx]]))
              st.write("-------------------------------------------")

      
      








    
        