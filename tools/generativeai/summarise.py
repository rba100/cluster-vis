from openai import OpenAI

def summariseStats(report: str):
    client = OpenAI()
    systemMessage = "You are an expert market researcher."
    userMessage = f"```\n{report}\n```\nWrite a summary of insights from this data. The percentage shows the proportion of that class label who have the given flag. Chi2 and p being secondary indicators. Do not mention chi2, but it can influence what you write."
    messages = [{"role": "system", "content": systemMessage}, {"role": "user", "content": userMessage}]

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages)
    
    content = response.choices[0].message.content
    return content
