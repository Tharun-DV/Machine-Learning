import streamlit as st
import pandas as pd
from io import StringIO


# Movie Recommendation App class
class MovieRecommendationApp:
    def __init__(self):
        self.data = None  # Initialize data to None

    def upload_file(self):
        uploaded_file = st.file_uploader("Upload a CSV file", type=["txt"])
        if uploaded_file is not None:
            try:
                # Read the uploaded file (Text file)
                s = uploaded_file.read().decode("utf-8")
                # Read the text content into a DataFrame, assuming CSV format with commas
                self.data = pd.read_csv(
                    StringIO(s),
                    header=None,
                    names=["title", "genre", "rating"],
                    sep=",",
                )
                self.data["rating"] = self.data["rating"].astype(
                    float
                )  # Convert ratings to float
                st.success("File uploaded and read successfully!")
                st.dataframe(self.data)
            except Exception as e:
                st.error(f"Error reading file: {e}")

    def get_recommendations(self):
        if self.data is None or self.data.empty:
            st.warning("Please upload a movie file first.")
            return

        genre = st.text_input("Genre:")
        min_rating = st.number_input("Min Rating:", min_value=0.0, format="%.1f")

        if st.button("Get Recommendations"):
            if not genre:
                st.warning("Please enter a genre.")
            else:
                recommendations = self.recommend_movies(genre, min_rating)
                if recommendations.empty:
                    st.warning("No movies found for the given criteria.")
                else:
                    st.success("Recommended Movies:")
                    st.dataframe(recommendations)

    def recommend_movies(self, genre, min_rating):
        # Filter the movies based on genre and minimum rating
        return self.data[
            (self.data["genre"].str.contains(genre, case=False, na=False))
            & (self.data["rating"] >= min_rating)
        ]


# Initialize the app
app = MovieRecommendationApp()
app.upload_file()
app.get_recommendations()
