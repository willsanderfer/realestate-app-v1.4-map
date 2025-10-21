# Comparable Adjustment Explorer (v1.4 map)

This Streamlit application helps real‑estate professionals analyze comparable sales, visualize adjustments, and generate data‑backed summaries.  Upload your sales data, choose a feature to analyze (such as square footage or garage spaces), and explore scatter plots with regression lines, summary statistics, and narrative explanations.

## Features

* **Flexible Uploads** – Accepts CSV or Excel files and automatically detects a price column from common names like **Sold Price** or **Close Price**.
* **Feature Picker** – Select from square footage, basement area, garage spaces, and other supported fields using fuzzy name matching.
* **Data Cleaning** – Strips currency symbols and commas, converts yes/no values to binary, and drops incomplete rows.
* **Interactive Charts** – Displays scatter plots with regression lines and, for discrete features, jittered dots or optional bar charts.  Shows slope and R² directly on the chart.
* **Summary Statistics** – Computes slope (price impact per unit), R², median price per square foot (where applicable), and sample size.  Outliers can be flagged and hidden.
* **Downloads** – Provides filtered comp lists, removed outliers, and summary tables as CSV files.  Generates an Excel report with charts embedded when `xlsxwriter` is installed.
* **Optional AI Narrative** – If an `OPENAI_API_KEY` is available in the environment, an AI generated narrative summarises the analysis in plain English.

## How to use

1. **Upload data** – Click **“Upload data file”** and select your CSV or Excel file of comparable sales.
2. **Choose a feature** – Pick a numeric feature from the dropdown (e.g. _SqFt Finished_, _Basement SqFt Finished_, _Garage Spaces_).  The app cleans and preprocesses the data automatically.
3. **Explore and export** – Review the scatter plot, adjust filters and outlier settings, and download the cleaned data, removed rows, summaries, and an optional Excel report with charts.

## Development

This repository contains the Streamlit app (`app.py`) along with its dependencies.  The original application logic resides in `app v1.4 (Map).py`.  When deploying on [Streamlit Community Cloud](https://share.streamlit.io/), set the main file to `app.py` and ensure the runtime Python version is 3.11 (via `runtime.txt`).

## License

This project is provided for educational purposes; no explicit license is granted.
