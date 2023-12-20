import sys
import pandas as pd
from joblib import Memory
from openai import OpenAI
import json

def printError(message, code):
    print(json.dumps({"error": message, "code": code}), file=sys.stderr)
    sys.exit(code)

memory = Memory(location='.', verbose=0, bytes_limit=1*1024*1024)

client = OpenAI()

classifyTool = {
    "type" : "function",
    "function": {
        "name" : "classify",
        "description" : "Classify the datatype of a column",
        "parameters" : {
            "type" : "object",
            "properties" : {
                "name" : {"type" : "string"},
                "type" : {"type" : "string", "enum" : [ "boolean", "classification", "range", "freeText", "id", "other" ]}
            },
            "required" : ["name", "type"]
        }
    }
}

tools = [classifyTool]

def hardCodedColumnTypes(samples):
    # Booleans
    allBooleans = [set(["0","1"]), set(["","1"]), set(["1"]), set(["no", "yes"]), set(["", "yes"]), set("yes"), set("no")]
    if set(samples) in allBooleans:
        return "boolean"
    if "male" in [s.lower() for s in samples]:
        return "classification"
    return None

@memory.cache
def classifyColumn(header: str, rowSamples: set[str]):
    systemPrompt = "You will classify a table column. The goal is to identify classification data and boolean data. " + \
                   " Classification is like gender, age-range, location or anything that is an enum. Booleans are yes/no 1/0 etc." + \
                   " All other columns should be classed as 'other', including IDs." + \
                   " Remember that numeric data is 'other', E.g. 'Age' is other, unless the rows are expressed as a range.\n"
    
    userMessage = f"ColumnName: {header}\nSample data for this column:\n" + "\n".join(rowSamples) + "\n"
    messages = [{"role": "system", "content": systemPrompt}, {"role": "user", "content": userMessage}]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            tools=tools
        )
    except Exception as e:
        printError(str(e), 1)

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    if(not tool_calls):
        printError("No tool calls were returned", 1)
    
    tool_call = tool_calls[0]
    if(tool_call.function.name != "classify"):
        printError("Tool call was not classify", 1)
    function_args = json.loads(tool_call.function.arguments)
    type = function_args.get("type", "other")

    # Handle decoy types
    if(type == "range"):
        return "classification"
    if(type == "freeText"):
        return "other"
    if(type == "id"):
        return "other"
    
    return type

def getColumnMetadata(df):
    columnMetadata = {}
    for column in df.columns:
        if(column == ""):
            continue
        columnMetadata[column] = {}
        unqiueValues = df[column].unique()[:3]

        hardCodedClass = hardCodedColumnTypes(unqiueValues)
        if(hardCodedClass):
            columnMetadata[column]["type"] = hardCodedClass
            columnMetadata[column]["classifiedBy"] = "hardcoded"
            continue

        columnMetadata[column]["type"] = classifyColumn(column, unqiueValues)
        columnMetadata[column]["classifiedBy"] = "openai"
    return columnMetadata

def processStdIn():
    table = pd.read_csv(sys.stdin, sep=",", na_filter=None, keep_default_na=False, dtype=str)
    columnMetadata = getColumnMetadata(table)
    print(json.dumps(columnMetadata, indent=4))

def getColumnMetadataCsv(path):
    table = pd.read_csv(path, sep=",", na_filter=None, keep_default_na=False, dtype=str)
    return getColumnMetadata(table)

def getColumnMetadataExcel(path, sheetName=0):
    table = pd.read_excel(path, na_filter=None, keep_default_na=False, dtype=str, sheet_name=sheetName)
    return getColumnMetadata(table)

if __name__ == "__main__":
    processStdIn()