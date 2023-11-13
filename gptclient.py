import openai
import concurrent.futures
import streamlit as st

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

    completion = openai.ChatCompletion.create(model=model,  temperature=0, messages=[
        {"role":"system","content":"en-GB. You name categories. Reply only with one word (or a short noun phrase if one word doesn't cut it)."},
        {"role": "user", "content": prompt}])
    content =  completion.choices[0].message.content.rstrip('.')
    content = content[0].upper() + content[1:]
    return content

def name_clusters_array(text_array):

    text = "\n".join(text_array)
    prompt = f"""
This is a text analysis. Common words in a corpus have been identified by vector similarity search. The words may seem fairly random as they are selected by vector similarity, not by frequency. The context of this request is the coding of market research survey free text responses.
```
{text}
```
Reply with exactly {len(text_array)} lines. On each line, name the cluster based on the general theme of the words in the list. If they are wildly different or opposite, you can use `/` to seperate. E.g. "yes,no,not,none,yeah" => "yes/no", but try to avoid this and just describe the theme in one or two words.
"""

    completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}])
    lines = completion.choices[0].message.content.split("\n")
    if(len(lines) != len(text_array)):
        raise Exception("Number of lines returned does not match number of clusters")
    return lines
