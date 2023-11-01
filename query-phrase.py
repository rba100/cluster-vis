import sys
import psycopg2
import openai

input_string = sys.argv[1]

# Get the OpenAI embedding
result = openai.Embedding.create(model='text-embedding-ada-002', input=input_string)
input_embedding = result['data'][0]['embedding']

# Convert the embedding to a string and format it correctly
input_embedding_str = str(input_embedding).replace(' ', '').replace('\'', '')
#print(input_embedding_str)

# Connect to the Postgres database
conn = psycopg2.connect(host='localhost', database='postgres', user='postgres', password='postgres')
cursor = conn.cursor()

# Query to find the ten closest words
query = """
SELECT word, embedding, iscommon
FROM words WHERE isCommon = true
ORDER BY embedding <#> %s
LIMIT 5
"""

# Execute the query
cursor.execute(query, (input_embedding_str,))

# Fetch the results and print them
results = cursor.fetchall()
for result in results:
    print(result[0])

# Close the connection
conn.close()