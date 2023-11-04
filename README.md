# Semantic Clustering

This is a Streamlit app that performs semantic clustering of text items and presents them as a visual plot. The app uses various techniques such as word embeddings, clustering algorithms, and t-SNE dimensionality reduction to group similar text items together.

## Setup

1. Clone the repository.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Set up a PostgreSQL database and update the connection string in the `ui.py` file.
4. Run the app by executing `streamlit run ui.py` in the terminal.

## Usage

1. Enter your text items in the input box, with each item separated by a new line.
2. Click the "Generate Scatter Plot" button to generate the visual plot of the text items.
3. Optionally, specify the number of clusters to identify and choose whether to use OpenAI to name the clusters.
4. Use the filtering options to show or hide items that are similar to a given text.
5. Explore the plot by hovering over the points to see the corresponding text and manually identify common themes.

## Additional Information

- The app requires a PostgreSQL database to store the text items and perform database operations. Unfortunately it also needs anout 6GB of static reference data to be useful.