import pandas as pd
from scipy.stats import chi2_contingency
from openai import OpenAI

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
                    reportLines.append(f"{enum_col} '{unique_value}' correlates with '{bit_col}' with Chi-square statistic {chi2:.3f}. Percentage: {percentage:.2f}% p-value: {p:.3f}")

    return "\n".join(reportLines)

def summarise_analysis_report(report: str):
    client = OpenAI()
    systemMessage = ""
    userMessage = f"```\n{report}\n```\nWrite a summary of insights from this data. Pay close attention to the percentage, with chi2 and p being secondary indicators."
    messages = [{"role": "system", "content": systemMessage}, {"role": "user", "content": userMessage}]

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages)
    
    return response.choices[0].message.content
