import sys
import json
from openai import OpenAI

def printError(message, code):
    print(json.dumps({"error": message, "code": code}), file=sys.stderr)
    sys.exit(code)

def getPythonForCharts(summary: str):
    client = OpenAI()
    systemMessage = "You are an expert market researcher and python programmer."
    userMessage = f"```\n{summary}\n```\nWrite python code to generate charts to go with insights from this summary. Charts should use colour to distinguish categories when relevant." + \
         " The code should use plotly to generate professional looking charts. The charts should not have built-it titles." + \
         " Reply in a code block with nothing before or after the code block: only write code as your output will be copied to a file. The code should save the images as png in a subfolder 'images' and not show them."
    messages = [{"role": "system", "content": systemMessage}, {"role": "user", "content": userMessage}]

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages)
    
    pythonCode = response.choices[0].message.content.strip("`\n\t ").strip()
    if pythonCode.startswith("python"):
        pythonCode = pythonCode[6:].strip()

    if "\n```\n" in pythonCode:
        pythonCode = pythonCode.split("\n```\n")[0]

    return pythonCode

def insertChartsIntoSummary(summary:str, chartNames: set[str]):
    client = OpenAI()
    summaryLines = summary.split("\n")
    if(summaryLines[-1] != ""):
        summaryLines.append("")
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

    userMessage = f"```\n{numberedSummary}\n```\nInsert images into the summary based on the image name, ideally at the end of the relevant subsection. Don't insert images in bullet lists, perfer right before the end of a subsection. Images: `{'`, `'.join(chartNames)}`."
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