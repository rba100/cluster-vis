import openai
import concurrent.futures

model = "gpt-3.5-turbo"

def name_clusters_summary(text):

    prompt = f"""
This is a text analysis. Common words in a corpus have been identified by vector similarity search.
Give a list of themes that a market researcher could look for in responses. Ignore terms that are clearly not helpful like 'none' or 'thanks'.
```
{text}
```
First reply with a list of themes, one for each cluster, then reply with a shorter list compiled from that first list, combining or removing as needed to obtain a concise list of themes, but do not combine things that are still interesting separately. The concise list doesn't have to be smaller than the clusters, it just needs to be the interesting labels to classify responses by. In brackets for each theme give a database column name for a bit column that marks whether a given body of text mentions that theme, in camelCase).
"""

    completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}])
    return completion.choices[0].message.content

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
These have been matched against the labels: {labels}.
Give a name for a master label that encompasses the labels are in the context of the samples (or a description of the common theme if the labels are unhelpful).
"""

    completion = openai.ChatCompletion.create(model=model, messages=[
        {"role":"system","content":"en-GB. You name categories. Reply only with one word (or a two word noun phrases if one word doesn't cut it)."},
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
