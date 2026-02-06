import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


# Page config
st.set_page_config(page_title="Restaurant Recommendation", layout="wide")


# Title
st.title("🍽️ Restaurant Recommendation System")
st.write("Content-Based Filtering using User Preferences")


# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv("restaurant_data.csv", encoding="latin1")
    return data


data = load_data()


# Show dataset
st.subheader("📊 Dataset Preview")
st.dataframe(data)


# Handle Missing Values
data = data.fillna("Unknown")


# Encoding Categorical Data
le_city = LabelEncoder()
le_cuisine = LabelEncoder()
le_delivery = LabelEncoder()

data["City_enc"] = le_city.fit_transform(data["City"])
data["Cuisine_enc"] = le_cuisine.fit_transform(data["Cuisines"])
data["Delivery_enc"] = le_delivery.fit_transform(data["Online_Delivery"])


# Normalize Cost
scaler = MinMaxScaler()
data["Cost_scaled"] = scaler.fit_transform(data[["Average_cost"]])


# Features for Similarity
features = data[
    ["City_enc", "Cuisine_enc", "Delivery_enc", "Cost_scaled"]
]


# Cosine Similarity
similarity = cosine_similarity(features)


# Sidebar Inputs
st.sidebar.header("🔍 Select Your Preferences")

city = st.sidebar.selectbox("Select City", data["City"].unique())
cuisine = st.sidebar.selectbox("Select Cuisine", data["Cuisines"].unique())
delivery = st.sidebar.selectbox("Online Delivery", ["Yes", "No"])
max_price = st.sidebar.slider("Maximum Price", 100, 1000, 500)


# Recommendation Button
if st.sidebar.button("Get Recommendations"):

    # Filter by user preference
    filtered = data[
        (data["City"] == city) &
        (data["Cuisines"] == cuisine) &
        (data["Online_Delivery"] == delivery) &
        (data["Average_cost"] <= max_price)
    ]


    st.subheader("✅ Recommended Restaurants")


    if len(filtered) == 0:
        st.warning("No restaurant found for your preferences 😔")

    else:
        st.dataframe(
            filtered[
                ["Restaurant", "City", "Cuisines", "Average_cost", "Rating"]
            ]
        )


        # Visualization
        st.subheader("📈 Ratings Comparison")

        plt.figure()
        sns.barplot(
            x="Restaurant",
            y="Rating",
            data=filtered
        )
        plt.xticks(rotation=45)

        st.pyplot(plt)


# Similar Restaurants Section
st.subheader("🤝 Similar Restaurants Finder")

selected_restaurant = st.selectbox(
    "Select Restaurant",
    data["Restaurant"]
)


if st.button("Find Similar Restaurants"):

    idx = data[data["Restaurant"] == selected_restaurant].index[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_matches = scores[1:6]


    st.write("### Similar Restaurants:")

    for i in top_matches:
        st.write(
            data.iloc[i[0]]["Restaurant"],
            " (Similarity Score:",
            round(i[1], 2),
            ")"
        )


# Cost Distribution
st.subheader("💰 Price Distribution")

plt.figure()
sns.histplot(data["Average_cost"], bins=10)
st.pyplot(plt)
