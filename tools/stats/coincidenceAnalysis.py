import sys
import json
import pandas as pd
from scipy.stats import chi2_contingency
from openai import OpenAI

def printError(message, code):
    print(json.dumps({"error": message, "code": code}), file=sys.stderr)
    sys.exit(code)

def coincidence_analysis_report(df, metadata):

    # Enum columns and Bit flag columns

    enum_columnNames = [key for key in metadata.keys() if metadata[key]['type'] == 'classification']
    flag_columnNames = [key for key in metadata.keys() if metadata[key]['type'] == 'boolean']

    enum_columns = df.columns.intersection(enum_columnNames)
    bit_columns = df.columns.intersection(flag_columnNames)

    # For all bit columns, replace True,Yes,yes etc with 1 and anything else with 0 (including empty string = 0)
    df[bit_columns] = df[bit_columns].replace(r'(?i)^(yes|true|1)$', 1, regex=True)
    df[bit_columns] = df[bit_columns].replace(r'(?i)^(no|false|0)$', 0, regex=True)
    df[bit_columns] = df[bit_columns].replace(r'^\s*$', 0, regex=True)

    reportLines = []

    # Iterate over each enum and bit flag column to check for correlations
    for enum_col in enum_columns:
        unique_values = df[enum_col].unique()
        for unique_value in unique_values:
            for bit_col in bit_columns:
                # Create a contingency table for each unique value
                table = pd.crosstab(df[enum_col] == unique_value, df[bit_col])

                # Perform Chi-square test
                chi2, p, dof, _ = chi2_contingency(table)

                # Calculate the percentage
                total = sum(df[enum_col] == unique_value)
                if total > 0:
                    percentage = sum((df[enum_col] == unique_value) & (df[bit_col] == 1)) / total * 100
                else:
                    percentage = 0

                # Print out correlations with p-value < 0.05 and the percentage
                if p < 0.05:
                    reportLines.append(f"{enum_col} '{unique_value}' correlates with '{bit_col}'. Percentage: {percentage:.2f}%. Chi-square {chi2:.3f}, p-value: {p:.3f}.")

    return "\n".join(reportLines)

def summarise_analysis_report(report: str):
    client = OpenAI()
    systemMessage = "You are an expert market researcher."
    userMessage = f"```\n{report}\n```\nWrite a summary of insights from this data. Pay close attention to the percentage, with chi2 and p being secondary indicators."
    messages = [{"role": "system", "content": systemMessage}, {"role": "user", "content": userMessage}]

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages)
    
    return response.choices[0].message.content

def generate_charts_for_summary(summary: str):
    client = OpenAI()
    systemMessage = "You are an expert market researcher and python programmer."
    userMessage = f"```\n{summary}\n```\nWrite python code to go with insights from this summary." + \
         " The code should use plotly to generate professional looking charts. The charts should not have built-it titles." + \
         " Reply in a code block with nothing before or after the code block: only write code as your output will be copied to a file. The code should save the images in a subfolder 'images' and not show them."
    messages = [{"role": "system", "content": systemMessage}, {"role": "user", "content": userMessage}]

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages)
    
    pythonCode = response.choices[0].message.content.strip("`\n\t ").strip()
    if pythonCode.startswith("python"):
        pythonCode = pythonCode[6:].strip()
    return pythonCode

def generate_summary_with_charts(summary:str, chartNames: set[str]):
    client = OpenAI()
    summaryLines = summary.split("\n")
    padding = len(str(len(summaryLines)))
    numberedLines = [f"{str(i).rjust(padding)}: {line}" for i, line in enumerate(summaryLines, 1)]
    numberedSummary = "\n".join(numberedLines)

    tool = {
        "type" : "function",
        "function": {
            "name" : "insert_images",
            "description" : "specifies the line numbers in the original document to insert an images at.",
            "parameters" : {
                "type" : "object",
                "properties" : {
                    "indices" : {
                        "type": "array",
                        "items" : {
                            "type" : "object",
                            "properties" : {
                                "lineNumber" : {"type" : "integer"},
                                "imageName" : {"type" : "string"}
                            },
                            "required" : ["lineNumber", "imageName"]
                        }
                    }
                }
            }
        }
    }

    userMessage = f"```\n{numberedSummary}\n```\nInsert images into the summary based on the image name, at the end of the relevant section (to place an image at the end of the document use max index + 1). Images: `{'`, `'.join(chartNames)}`."
    messages = [{"role": "system", "content": "You are an expert market researcher"}, {"role": "user", "content": userMessage}]

    response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            tools=[tool]
        )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    if(not tool_calls):
        printError("No tool calls were returned", 1)
    
    tool_call = tool_calls[0]
    if(tool_call.function.name != "insert_images"):
        printError("Tool call was not insert_images", 1)
    function_args = json.loads(tool_call.function.arguments)
    indicies = function_args.get("indices", [])
    # sort by lineNumber descending so we can insert without affecting the line numbers
    indicies.sort(key=lambda x: x["lineNumber"], reverse=True)
    for index in indicies:
        lineNumber = index["lineNumber"]
        imageName = index["imageName"]
        friendlyImageName = imageName.replace("_", " ").replace(".png", "")
        friendlyImageName = " ".join([word.capitalize() for word in friendlyImageName.split(" ")])
        summaryLines.insert(lineNumber, f"![{friendlyImageName}](images/{imageName})")

    return "\n".join(summaryLines)