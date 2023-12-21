import sys
import json
from openai import OpenAI

def printError(message, code):
    print(json.dumps({"error": message, "code": code}), file=sys.stderr)
    sys.exit(code)

def getPythonForCharts(summary: str, stats: str):
    client = OpenAI()
    systemMessage = "You are an expert market researcher and python programmer."
    userMessage = f"```\nSTATS\n{stats}\nSUMAMRY:\n{summary}\n```\nWrite python code to generate charts to go with insights from the text SUMMARY (stats provided for just for reference, percentages are independent and cannot be summed). Charts should use colour to distinguish categories when relevant, always prefering to use primary colour if there's only one or two classes. For very volumuous data you can use a heatmap. " + \
         " The code should use plotly where possible to generate professional looking charts. The charts should not have built-it titles. When listing categories, ensure they are in an order than makes sense." + \
         " Reply in a code block with nothing before or after the code block: only write code as your output will be copied to a file. The code should save the images as png in a subfolder 'images' and not show them." + \
         " Style guide colours: Primaries: Pink #F5AEB3 and Black #000000 (and grey tints 10%/25%/50%/75%) for most charts. Secondary colours: #5DAED4 #5EA19B #7F6BA8 #F6AA57 #0E5066 #E2585C if many colours needed. Chart background 10%/grey"
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
        imageFileNameWithoutPath = imageName.replace("\\","/").split("/")[-1]
        friendlyImageName = imageName.replace("_", " ").replace(".png", "")
        friendlyImageName = " ".join([word.capitalize() for word in friendlyImageName.split(" ")])
        # Need an extra \n because the mdpdf tool renders images too small if they are right after certain featuyres.
        summaryLines.insert(lineNumber, f"\n![{friendlyImageName}](images/{imageFileNameWithoutPath})")

    return "\n".join(summaryLines)