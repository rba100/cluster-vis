# Semantic Clustering

This is a Streamlit app that performs semantic clustering of text items and presents them as a visual plot. The app uses various techniques such as word embeddings, clustering algorithms, and t-SNE dimensionality reduction to group similar text items together.

## Setup

1. Clone the repository.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Create `.streamlit/secrets.toml` with connection string and openai key.
4. Run the app by executing `streamlit run app.py` in the terminal.

## Usage

See in-app help. Quick start: paste a bunch of text items, each on a new line, and click 'Render'.

## Additional Information

- The app requires a PostgreSQL database to store the text items and perform database operations. Unfortunately it also needs a tonne of static reference data to be useful.
