import pandas as pd
import numpy as np
import streamlit as st

# load data
@st.cache_data
def load_products_info():
    return pd.read_csv('3_task_4_no_duplicates_reduced.zip', low_memory=False, encoding='utf-8')

df = load_products_info()

@st.cache_data
def load_similarity_matrix():
    return np.load("similarity_top_k.npy")

similarity = load_similarity_matrix()

# page styling
st.markdown(f"""
    <style>
        p {{
            margin-bottom: 0;
        }}
        div[data-testid="column"] {{
            margin-bottom: 1rem;
        }}
        .block-container {{
            padding: 0
        }}
    </style>""",
    unsafe_allow_html=True,
)

# output
st.title('Berlin Groceries Product Similarities')

query_params = st.experimental_get_query_params()
if not query_params:
    # user input
    id = st.number_input("Select Product ID (0-"+str(similarity.shape[0])+"):", 0, similarity.shape[0], 45551)
else:
    # id passed via URL
    id = int(query_params["id"][0])

#output recommandations
n_cols = 4
j = 0

for i in similarity[id][:20]:
    if j%n_cols == 0:
        cols = st.columns(n_cols)
    
    with st.container():
        with cols[j%n_cols]:
            if pd.notna(df.iloc[i]['IMAGE_URL']):
                st.image(df.iloc[i]['IMAGE_URL'])
            if j==0:
                st.write(":red[**Reference product**]")
            st.markdown('['+df.iloc[i]['PRODUCT_NAME']+']('+df.iloc[i]['PRODUCT_LINK']+')', unsafe_allow_html=True)
            st.write("**"+str(df.iloc[i]['STORE_NAME'])+"**")
            st.write(":red[**"+str(df.iloc[i]['PRODUCT_PRICE'])+" €**]")
    j+=1