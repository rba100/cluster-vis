import os
import sys
import subprocess
from joblib import Memory
import pandas as pd

from metadata.columns import getColumnMetadataExcel
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

memory = Memory(location='.', verbose=0)

@memory.cache
def getMetadata(fileName):
    print("Getting metadata")
    return getColumnMetadataExcel(fileName)

@memory.cache
def getSummary(report):
    print("Getting summary")
    return summariseStats(report)

@memory.cache
def getReport(fileName, metadata):
    print("Getting stats report")
    table = pd.read_excel(fileName, na_filter=None, keep_default_na=False, dtype=str, sheet_name=0)
    return getCoincidenceStats(table, metadata)

@memory.cache
def getCharts(summary):
    print("Getting chart python code")
    return getPythonForCharts(summary)

@memory.cache
def getSummaryWithCharts(summary, charts):
    print("Getting summary with charts")
    return insertChartsIntoSummary(summary, charts)

metadata = getMetadata(fileName)
report = getReport(fileName, metadata)
summary = getSummary(report)
charts = getCharts(summary)

if os.path.exists("out"):
    os.removedirs("out")
os.makedirs("out/images")

# write report, summary, charts to report.txt summary.md and charts.py
with open("out/stats.txt", "w") as f:
    f.write(report)
with open("out/summary.md", "w") as f:
    f.write(summary)
with open("out/charts.py", "w") as f:
    f.write(charts)

subprocess.run(["python", "charts.py"], cwd="out")

# get image names without subfolder using os to enumerate dir contents
imageNames = [f for f in os.listdir("out/images") if f.endswith(".png")]
imageNames = [f.replace("\\", "/") for f in imageNames]
imageNames = set([f.split("/")[-1] for f in imageNames])
print(imageNames)

if(len(imageNames) > 0):
    summaryWithImages = getSummaryWithCharts(summary, imageNames)
    with open("out/summary.md", "w") as f:
        f.write(summaryWithImages)

subprocess.run(["mdpdf", "summary.md", "-o", "report.pdf"], cwd="out")