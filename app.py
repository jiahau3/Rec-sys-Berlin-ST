import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Any time something must be updated on the screen, Streamlit reruns your entire Python script from top to bottom.
# This can happen in two situations:
# - Whenever you modify your app's source code.
# - Whenever a user interacts with widgets in the app. For example, when dragging a slider, entering text in an input box, or clicking a button.

# For this reason, we use state variables to store the current state and info of the app,
#   and according to these variables, we run (display) or hide certain parts of the code.

# This way we are able to build a more complex app. For example mimicking a two page app.

# The main focus was building the architecture according to how Streamlit operated and creating, storing, and updating the state variables.


# PAGE STYLING
st.markdown(f"""
    <style>
        p {{
            margin-bottom: 0;
        }}
        div[data-testid="column"]:nth-of-type(n+3) {{
            margin-bottom: 1rem;
        }}
        .block-container {{
            padding: 0
        }}
        footer {{
            margin-top: 100px
            display: none
        }}
        .stButton {{
            margin-top: .5rem
        }}
    </style>""",
    unsafe_allow_html=True,
)


# DATA LOADING AND CACHING
@st.cache_data
def load_products_info():
    return pd.read_csv('3_task_4_no_duplicates_reduced.zip', low_memory=False, encoding='utf-8')

df = load_products_info()

@st.cache_data
def load_embeddings():
    return np.load("final_matrix_f64_noP.npy")

embeddings_matrix = load_embeddings()

@st.cache_data
def load_similarity_matrix():
    return np.load("similarity_top_k.npy")

similarity = load_similarity_matrix()

@st.cache_data
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

bert = load_model()


# FUNCTIONS THAT CHANGE SESSION VARIABLES
def sort_recommendations():
    if st.session_state.order=="Relevance":
        st.session_state.recommendation_ids_ordered = st.session_state.recommendation_ids_filtered
    
    elif st.session_state.order=="Price ascending":
        st.session_state.recommendation_ids_ordered = df.iloc[st.session_state.recommendation_ids_filtered].sort_values("PRODUCT_PRICE").index

    else:
        st.session_state.recommendation_ids_ordered = df.iloc[st.session_state.recommendation_ids_filtered].sort_values("PRODUCT_PRICE", ascending=False).index

def get_recommendation_ids_full():
    search_arr = np.array([st.session_state.search])

    # get embeddings
    search_embeddings = bert.encode(search_arr)

    # cosine similarity
    similarity_matrix = cosine_similarity(search_embeddings, embeddings_matrix)

    # retrieve and store top recommendation ids
    st.session_state.recommendation_ids_full = np.argsort(similarity_matrix[0])[::-1]
    st.session_state.recommendation_ids_filtered = st.session_state.recommendation_ids_full[:20]
    st.session_state.recommendation_ids_ordered = st.session_state.recommendation_ids_filtered

    # reset price filters
    if "price_min" in st.session_state:
        st.session_state.price_min = st.session_state.price_absolute_min
        st.session_state.price_max = st.session_state.price_absolute_max

def get_recommendation_ids_filtered():
    st.session_state.recommendation_ids_filtered = \
        df.iloc[st.session_state.recommendation_ids_full][(df["PRODUCT_PRICE"]>=st.session_state.price_min) & (df["PRODUCT_PRICE"]<=st.session_state.price_max)][:20].index
    sort_recommendations()

def change_product_id(new_product_id):
    if new_product_id:
        st.session_state.search_clone = st.session_state.search
        st.session_state.price_min_clone = st.session_state.price_min
        st.session_state.price_max_clone = st.session_state.price_max
        st.session_state.order_clone = st.session_state.order
    else:
        st.session_state.search = st.session_state.search_clone
        st.session_state.price_min = st.session_state.price_min_clone
        st.session_state.price_max = st.session_state.price_max_clone
        st.session_state.order = st.session_state.order_clone

    st.product_id = new_product_id


# OUTPUT FUNCTIONS
def output_recommendations(ids, display_similarities=False):
    n_cols = 4
    j = 0

    for i, row in df.iloc[ids].iterrows():
        if j%n_cols == 0:
            cols = st.columns(n_cols)
        
        with st.container():
            with cols[j%n_cols]:
                if pd.notna(row['IMAGE_URL']):
                    st.image(row['IMAGE_URL'])
                if display_similarities & (j==0):
                    st.write(":red[**Reference product**]")
                st.markdown('['+row['PRODUCT_NAME']+']('+row['PRODUCT_LINK']+')', unsafe_allow_html=True)
                st.write("**"+str(row['STORE_NAME'])+"**")
                st.write(":red[**"+str(row['PRODUCT_PRICE'])+" €**]")
                if not display_similarities:
                    st.button('similar products', key=""+str(i)+"", on_click=change_product_id, kwargs=({"new_product_id":i}))
        j+=1


# INITIALIZE SESSION VARIABLES
if "search" not in st.session_state:
    # initialize and associate widget (form) values with session variables
    st.session_state.search = "chocolate gift large"
    get_recommendation_ids_full()
    st.session_state.price_absolute_min = int(df["PRODUCT_PRICE"].min())
    st.session_state.price_absolute_max = int(df["PRODUCT_PRICE"].max())
    st.session_state.price_min = st.session_state.price_absolute_min
    st.session_state.price_max = st.session_state.price_absolute_max
    st.product_id = False


# OUTPUT
if st.product_id:

    st.title('Berlin Grocery Recommender: Product Similarities')

    st.button('< Back to the initial search', on_click=change_product_id, kwargs=({"new_product_id":False}))

    output_recommendations(similarity[st.product_id][:20], True)

else:

    st.title('Berlin Grocery Recommender')

    st.text_input('Search query', key="search", on_change=get_recommendation_ids_full)

    col1, col2 = st.columns(2)

    with col1:
        st.number_input(
            'Min price ('+str(st.session_state.price_absolute_min)+'-'+str(st.session_state.price_absolute_max)+')', 
            min_value=st.session_state.price_absolute_min,
            max_value=st.session_state.price_absolute_max,
            key="price_min",
            on_change=get_recommendation_ids_filtered
        )

    with col2:
        st.number_input(
            'Max price ('+str(st.session_state.price_absolute_min)+'-'+str(st.session_state.price_absolute_max)+')', 
            min_value=st.session_state.price_absolute_min,
            max_value=st.session_state.price_absolute_max,
            key="price_max",
            on_change=get_recommendation_ids_filtered
        )

    st.selectbox(
        "Sort results by:",
        ("Relevance", "Price ascending", "Price descending"),
        on_change = sort_recommendations,
        key = "order"
    )

    output_recommendations(st.session_state.recommendation_ids_ordered)