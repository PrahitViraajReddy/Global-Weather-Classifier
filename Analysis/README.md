# Global Weather Analysis

This directory contains the Power BI analysis layer for the **Global Weather Classifier** project.

## Objective

The analysis layer converts the global weather data used by the project into an interactive business-intelligence view for comparing weather conditions across countries and studying trends over time.

## Current Analysis Scope

- Executive weather KPI dashboard
- Average temperature analysis
- Average AQI analysis
- Average humidity analysis
- Average visibility analysis
- Average wind-speed analysis
- Average precipitation analysis
- Average cloud-cover analysis
- Average UV-index analysis
- Top-20 country comparisons
- Geographic analysis using maps
- Time-based trend analysis

## Power BI Report Structure

The report contains dedicated analysis pages for the major weather indicators, an executive dashboard, and a trends section.

### Executive Dashboard

Provides a consolidated view of the major weather KPIs:

- Total records
- Average temperature
- Average AQI
- Average UV index
- Average humidity
- Average visibility
- Average wind speed
- Average precipitation
- Average cloud cover

### Country-Level Analysis

The indicator pages use Top-20 country comparisons to keep the visual analysis focused and readable while still supporting global-scale data exploration.

### Trend Analysis

The trends section uses date-based visualizations to examine how the major weather indicators change over time.

## Architecture

```text
Global Weather Data
        |
        +--------------------+
        |                    |
        v                    v
   ML Classification     Power BI Analysis
        |                    |
        v                    v
   Model Output       Semantic Model + DAX
                             |
                             v
                    Interactive Dashboard
```

The current Power BI report focuses on weather analytics. Integration of model prediction outputs into the BI layer can be added as a subsequent phase so that actual-vs-predicted performance and model confidence can be analyzed alongside weather conditions.

## Tooling

- Power BI Desktop
- Power Query
- DAX
- Interactive maps and charts
- Date-based trend analysis

## Repository Role

The parent repository contains the machine-learning implementation. This `Analysis` directory keeps the BI/analytics layer separated from the model-development files while keeping both parts under the same project.

## Usage

Open the `.pbix` report with **Microsoft Power BI Desktop** to interact with the dashboard and inspect the report model, visuals, filters, and measures.

> Note: The PBIX file is a Power BI Desktop report and is intended to be used as an analysis artifact rather than executed as a Python application.
