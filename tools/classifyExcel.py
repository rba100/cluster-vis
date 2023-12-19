from joblib import Memory
import pandas as pd

from metadata.columns import getColumnMetadataExcel
from stats.coincidenceAnalysis import coincidence_analysis_report, summarise_analysis_report

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

fileName = "F:\\Temp\\excel\\NFL VoF Synthetic Data.xlsx"

metadata = getMetadata(fileName)
report = getReport(fileName, metadata)
summary = getSummary(report)

print(summary)
