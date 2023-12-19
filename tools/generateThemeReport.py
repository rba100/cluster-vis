import os
import sys
import subprocess
from joblib import Memory
import pandas as pd
from metadata.columns import getColumnMetadataExcel
from stats.coincidenceAnalysis import coincidence_analysis_report, summarise_analysis_report, generate_charts_for_summary, generate_summary_with_charts

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
    return getColumnMetadataExcel(fileName)

@memory.cache
def getSummary(report):
    return summarise_analysis_report(report)

@memory.cache
def getReport(fileName, metadata):
    table = pd.read_excel(fileName, na_filter=None, keep_default_na=False, dtype=str, sheet_name=0)
    return coincidence_analysis_report(table, metadata)

@memory.cache
def getCharts(summary):
    return generate_charts_for_summary(summary)

@memory.cache
def getSummaryWithCharts(summary, charts):
    return generate_summary_with_charts(summary, charts)

metadata = getMetadata(fileName)
report = getReport(fileName, metadata)
summary = getSummary(report)
charts = getCharts(summary)

os.makedirs("out", exist_ok=True)
os.makedirs("out/images", exist_ok=True)

# write report, summary, charts to report.txt summary.md and charts.py
with open("out/stats.txt", "w") as f:
    f.write(report)
with open("out/summary.md", "w") as f:
    f.write(summary)
with open("out/charts.py", "w") as f:
    f.write(charts)

subprocess.run(["python", "charts.py"], cwd="out")

# get image names without subfolder using os to enumerate dir contents
imageNames = [f for f in os.listdir("out/images") if os.path.isfile(os.path.join("out/images", f))]
imageNames = [f.replace("\\", "/") for f in imageNames]
imageNames = [f.split("/")[-1] for f in imageNames]

if(len(imageNames) > 0):
    summaryWithImages = getSummaryWithCharts(summary, set(imageNames))
    with open("out/summary.md", "w") as f:
        f.write(summaryWithImages)

subprocess.run(["mdpdf", "summary.md", "-o", "report.pdf"], cwd="out")