
import streamlit as st
import pandas as pd
import torch
import requests
import random
from io import BytesIO
from PIL import Image
from torch_geometric.nn import SAGEConv, to_hetero, Linear
from dotenv import load_dotenv
import os

import torch
from torch_geometric.nn import SAGEConv, to_hetero, Linear
from dotenv import load_dotenv


import viz_utils
import model_def

load_dotenv() #load environment variables from .env file


API_KEY = os.getenv("HUGGINGFACE_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

# --- LOAD MOVIES DATA AND MODEL ---
movies_df = pd.read_csv("./sampled_movie_dataset/movies_metadata.csv")  # Load your movie data
m_data = torch.load("./PyGdata.pt")






device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# m_model = model_def.Model(hidden_channels=32).to(device) 
# m_model.load_state_dict(torch.load("PyGTrainedModelState.pt"))
# m_model.eval()



# --- AMAZON MODEL DEFINITION, TODO: MOVE TO OWN FILE ----

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['products'][col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, a_data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
    


# --- AMAZON DATA &  MODEL --- 
amazon_df = pd.read_csv("./amazon_data_source/amazon_reviews.csv")
a_data = torch.load("./product_data_PYG.pt")


a_model = Model(hidden_channels=32).to(device) 
a_model.load_state_dict(torch.load("amazon_best_model.pt"))
a_model.eval()


# --- AMAZON DATA EXTRA TODO: Delete, fold a_data into sampled_ad
sampled_ad = a_data[['ProductId', 'UserId', 'ProfileName', 'Score', 'Summary', 'Text']]
sampled_ad.columns = ['product_id', 'user_id', 'username', 'rating', 'review_title', 'review_content']
sampled_ad = sampled_ad.sort_values('product_id')
sampled_ad.reset_index(drop=True, inplace=True)

sampled_ad['username'] = sampled_ad['username'].fillna('unknown')
sampled_ad['review_title'] = sampled_ad['review_title'].fillna('')
sampled_ad['rating'] = sampled_ad['rating'].astype(int) 

# --- STREAMLIT APP ---
st.title("Movie Recommendation App")
user_id = st.number_input("Enter the User ID:", min_value=0)
#TODO: allow search by username for amazon product

with torch.no_grad():
    a = m_model.encoder(m_data.x_dict,m_data.edge_index_dict)
    user = pd.DataFrame(a['user'].detach().cpu())
    movie = pd.DataFrame(a['movie'].detach().cpu())
    embedding_df = pd.concat([user, movie], axis=0)

st.subheader('UMAP Visualization')
umap_fig = viz_utils.visualize_embeddings_umap(embedding_df)
st.plotly_chart(umap_fig)

st.subheader('TSNE Visualization')
tsne_fig = viz_utils.visualize_embeddings_tsne(embedding_df)
st.plotly_chart(tsne_fig)

st.subheader('PCA Visualization')
pca_fig = viz_utils.visualize_embeddings_pca(embedding_df)
st.plotly_chart(pca_fig)






def get_product_recommendations(model, a_data, user_id, total_products):
    user_row = torch.tensor([user_id] * total_products).to(device)
    print("user row is", user_row)
    all_product_ids = torch.arange(total_products).to(device)
    edge_label_index = torch.stack([user_row, all_product_ids], dim=0)

    pred = model(a_data.x_dict, a_data.edge_index_dict, edge_label_index).to('cpu')
    top_five_indices = pred.topk(5).indices

    recommended_products = sampled_ad["product_id"].iloc[top_five_indices]
    return recommended_products

# def get_movie_recommendations(model, data, user_id, total_movies):
#     user_row = torch.tensor([user_id] * total_movies).to(device)
#     all_movie_ids = torch.arange(total_movies).to(device)
#     edge_label_index = torch.stack([user_row, all_movie_ids], dim=0)

#     pred = model(data.x_dict, data.edge_index_dict, edge_label_index).to('cpu')
#     top_five_indices = pred.topk(5).indices

#     recommended_movies = movies_df.iloc[top_five_indices]
#     return recommended_movies

# def generate_poster(movie_title):
#     headers = {"Authorization": f"Bearer {API_KEY}"}

#     #creates random seed so movie poster changes on refresh even if same title. 
#     seed = random.randint(0, 2**32 - 1)
#     payload = {
#         "inputs": movie_title,
#         # "parameters": {
#         #     "seed": seed
#         # }
#     }

#     try:
#         response = requests.post(API_URL, headers=headers, json=payload)
#         response.raise_for_status()  # Raise an error if the request fails

#         # Display the generated image
#         image = Image.open(BytesIO(response.content))
#         st.image(image, caption=movie_title)

#     except requests.exceptions.HTTPError as err:
#         st.error(f"Image generation failed: {err}")


# if st.button("Get Recommendations"):
#     st.write("Top 5 Recommendations:")
#     try:
#         total_movies = m_data['movie'].num_nodes  
#         recommended_movies = get_movie_recommendations(m_model, m_data, user_id, total_movies)
#         cols = st.columns(3)  

       
#         for i, row in recommended_movies.iterrows():
#             with cols[i % 3]: 
#                 #st.write(f"{i+1}. {row['title']}") 
#                 try:
#                     image = generate_poster(row['title'])
#                 except requests.exceptions.HTTPError as err:
#                     st.error(f"Image generation failed for {row['title']}: {err}")

#     except Exception as e:
#         st.error(f"An error occurred: {e}")

if st.button("Get Amazon Recommendations"):
    st.write("Top 5 Recommendations:")
    try:
        total_products = sampled_ad['product_id'].num_nodes  
        recommended_products = get_product_recommendations(a_model, a_data, user_id, total_products)
        cols = st.columns(3)  

       
        for i, row in recommended_products.iterrows():
            with cols[i % 3]: 
                st.write(f"{i+1}. {row['product_id']}") 
                # try:
                #     image = generate_poster(row['title'])
                # except requests.exceptions.HTTPError as err:
                #     st.error(f"Image generation failed for {row['title']}: {err}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
