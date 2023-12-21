import os
import sys
import subprocess
from joblib import Memory
import pandas as pd
from pandas import ExcelFile
import json
import shutil

from metadata.columns import getColumnMetadataExcel, getColumnMetadata
from stats.coincidenceAnalysis import getCoincidenceStats
from generativeai.summarise import summariseStats
from generativeai.charts import getPythonForCharts, insertChartsIntoSummary

if(len(sys.argv) < 2):
    print("Missing file name argument")
    sys.exit(1)

fileName = sys.argv[1]
if not os.path.isfile(fileName):
    print("File does not exist")
    sys.exit(2)

sheetName = 0 if(len(sys.argv) < 3) else sys.argv[2]

memory = Memory(location='.', verbose=0, bytes_limit=100*1024*1024)

def getMetadata(filePath, sheetName):
    print("Getting metadata")
    if isinstance(filePath, str):
        return getColumnMetadataExcel(filePath, sheetName)
    else:
        raise Exception("getMetadata requires a file path")

@memory.cache
def getSummary(report):
    print("Getting summary")
    return summariseStats(report)

@memory.cache
def getReport(fileName, metadata, sheetName):
    print("Getting stats report")
    table = pd.read_excel(fileName, na_filter=None, keep_default_na=False, dtype=str, sheet_name=sheetName)
    return getCoincidenceStats(table, metadata)

@memory.cache
def getCharts(summary):
    print("Getting chart python code")
    return getPythonForCharts(summary)

@memory.cache
def getSummaryWithCharts(summary, charts):
    print("Getting summary with charts")
    return insertChartsIntoSummary(summary, charts)

metadata = getMetadata(fileName, sheetName)
report, insignificant = getReport(fileName, metadata, sheetName)
summary = getSummary(report) if report.strip() != "" else ""
charts = getCharts(summary, report) if summary.strip() != "" else ""

if os.path.exists("out"):
    shutil.rmtree("out")
os.makedirs("out/images")

# write metadata, report, summary, charts to report.txt summary.md and charts.py
with open("out/metadata.json", "w") as f:
    f.write(json.dumps(metadata, indent=4))
with open("out/stats.txt", "w") as f:
    f.write(report)
with open("out/insignificant.txt", "w") as f:
    f.write(insignificant)
with open("out/summary.md", "w", encoding="utf-8") as f:
    f.write(summary)
with open("out/charts.py", "w") as f:
    f.write(charts)

if charts != "":
    subprocess.run(["python", "charts.py"], cwd="out")

# get image names without subfolder using os to enumerate dir contents
imageNames = [f for f in os.listdir("out/images") if f.endswith(".png")]
imageNames = [f.replace("\\", "/") for f in imageNames]
imageNames = set([f.split("/")[-1] for f in imageNames])
print(imageNames)

if(len(imageNames) > 0):
    summaryWithImages = getSummaryWithCharts(summary, imageNames)
    with open("out/summary.md", "w", encoding="utf-8") as f:
        f.write(summaryWithImages)

subprocess.run(["mdpdf", "summary.md", "-o", "report.pdf"], cwd="out")