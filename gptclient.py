from openai import OpenAI
import concurrent.futures
import streamlit as st

client = OpenAI()

model = "gpt-3.5-turbo"


@st.cache_data(max_entries=40, show_spinner="Renaming clusters with GPT...")
def generate_cluster_names_many_st(tasks):
    return generate_cluster_names_many(tasks)


def generate_cluster_names_many(tasks):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks and collect futures
        future_to_task = {
            executor.submit(
                generate_cluster_name,
                task["labels"],
                task["samples"],
                task["additionalInstructions"],
            ): i
            for i, task in enumerate(tasks)
        }
        results = [None] * len(tasks)  # Pre-allocate the result list
        for future in concurrent.futures.as_completed(future_to_task):
            task_index = future_to_task[future]  # Get the original index of the task
            results[task_index] = future.result()  # Place result in the correct order
    return results


def generate_cluster_name(words, samples, additionalInstructions=None):
    # First, we apply some hardcoded rules
    poorQualityText = ["n/a", "none", "nothing", "not sure"]
    for text in poorQualityText:
        if sum(1 for sample in samples if sample.lower().strip().startswith(text)) >= 2:
            return "Poor quality: " + text

    if isinstance(words, list):
        words = ", ".join(words)

    nl = "\n"

    prompt = f"""
Examine these seperate lines of text:
```
{nl.join(samples)}
```
These have been matched against the words: {words}.
{additionalInstructions} Reply with one line only, 1-3 words.
"""

    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "en-GB. You name themes detected in survey responses. You will be given some examples of the responses and some words that may or may not help guide you. Reply only with 1-3 words to give a name for a theme label that can be found in each of the samples. If the samples are less than 3 words typically, just give an example.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = completion.choices[0].message.content.rstrip(".")
    content = content[0].upper() + content[1:]
    return content.strip('"')


def getCommonTheme(themes):
    text = "\n".join(themes)
    prompt = f"""
Here is a list of themes observed in a corpus of text. The themes are:
```
{text}
```
If there are concepts that appear to be dominating the themes, reply with the name of the common theme. The common theme should be a short noun phrase or a single word and can be influenced by the words in brackets.
If there are no common themes, reply with 'none', otherwise reply with the name of the common theme ONLY.
"""

    completion = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content.strip()
