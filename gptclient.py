from openai import OpenAI
import concurrent.futures
import streamlit as st

client = OpenAI()

model = "gpt-3.5-turbo"

@st.cache_data(max_entries=40)
def generate_cluster_names_many(tasks):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks and collect futures
        future_to_task = {executor.submit(generate_cluster_name, task["labels"], task["samples"]): i for i, task in enumerate(tasks)}
        results = [None] * len(tasks)  # Pre-allocate the result list
        for future in concurrent.futures.as_completed(future_to_task):
            task_index = future_to_task[future]  # Get the original index of the task
            results[task_index] = future.result()  # Place result in the correct order
    return results

def generate_cluster_name(labels, samples):
    if isinstance(labels, list):
        labels = ', '.join(labels)
    
    nl = '\n'

    prompt = f"""
Examine these samples of text:
```
{nl.join(samples)}
```
These have been matched against the words: {labels}.
Give a name for a master label that encompasses the words are in the context of the samples (or a description of the common theme if the words are unhelpful).
"""

    completion = client.chat.completions.create(model=model,  temperature=0, messages=[
        {"role":"system","content":"en-GB. You name categories. Reply only with one word (or a short noun phrase if one word doesn't cut it)."},
        {"role": "user", "content": prompt}])
    content =  completion.choices[0].message.content.rstrip('.')
    content = content[0].upper() + content[1:]
    return content

def name_clusters_array(text_array):

    text = "\n".join(text_array)
    prompt = f"""
User
This is a text analysis. Common words in a corpus have been identified by vector similarity search.
Give a list of themes that a market researching could look for in responses. Ignore terms that are clearly not helpful like 'none' or 'thanks'.
```
{text}
```
reply with a list of themes, one for each cluster. Do not write anything else, just write one theme per line of text returned. Write only the headline of the theme, do not write any description or explanation. If you are not sure what to write for a given input, return the original string for that theme.
"""

    completion = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    lines = completion.choices[0].message.content.split("\n")
    if(len(lines) != len(text_array)):
        raise Exception("Number of lines returned does not match number of clusters")
    return lines
