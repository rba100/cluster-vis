from openai import OpenAI

def summariseStats(report: str):
    client = OpenAI()
    systemMessage = "You are an expert market researcher."
    userMessage = f"```\n{report}\n```\nWrite a summary of insights from this data. Pay close attention to the percentage, with chi2 and p being secondary indicators."
    messages = [{"role": "system", "content": systemMessage}, {"role": "user", "content": userMessage}]

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages)
    
    return response.choices[0].message.content
