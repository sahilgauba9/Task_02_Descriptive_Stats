# Task_02_Descriptive_Stats

A comprehensive statistical analysis of Facebook political advertising data using three different approaches: Pure Python, Pandas, and Polars.

## 📋 Overview

This repository contains three Python scripts that perform identical statistical analyses on a Facebook political advertising dataset using different approaches:

1. **`pure_python_stats.py`** - Uses only Python standard library
2. **`pandas_stats.py`** - Uses pandas for data manipulation and analysis  
3. **`polar_stats.py`** - Uses Polars as a fast alternative to pandas

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Dataset file: `2024_fb_ads_president_scored_anon.csv` (not included in repository)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Task_02_Descriptive_Stats.git
cd Task_02_Descriptive_Stats
```

2. Install required dependencies:

For pandas version:
```bash
pip install pandas numpy
```

For Polars version:
```bash
pip install polars
```

For pure Python version:
```bash
# No additional dependencies required - uses only standard library
```
### Usage

1. Place your dataset file `2024_fb_ads_president_scored_anon.csv` in the repository directory

2. Run any of the analysis scripts:

```bash
# Pure Python approach (no dependencies)
python pure_python_stats.py

# Pandas approach (fastest for most users)
python pandas_stats.py

# Polars approach (memory efficient)
python polar_stats.py
```

## 📊 Analysis Features
Each script performs the same comprehensive statistical analysis:

### Dataset-Level Analysis
- **Row/column counts** and data shape
- **Memory usage** estimation
- **Null value** detection and counting
- **Duplicate row** identification
- **Data type** distribution

### Column-Level Analysis
For **Numeric Columns**:
- Count, mean, standard deviation
- Min, max values
- Quartiles (25%, 50%, 75%)
- Sum and zero/negative value counts
- **Unique value** counts

For **Categorical Columns**:
- **Unique value** counts using `nunique()`
- **Most frequent values** using `value_counts()`
- **Mode** (most frequent value)
- **Frequency distributions**

### Aggregation Analysis
- **By `page_id`**: Aggregates ads by advertiser page
- **By `(page_id, ad_id)`**: Individual ad analysis
- **Total preservation** verification across aggregations

## 🔍 Key Findings

### Dataset Overview
- **Size**: 246,745 rows × 41 columns
- **Time Range**: July 2021 - November 2024
- **Total Spend**: $261,868,355
- **Total Impressions**: 11,251,948,521
- **Unique Advertisers**: 4,475 pages
- **Unique Ads**: 246,745 individual advertisements

### Top Political Advertisers (by ad count)
1. **HARRIS FOR PRESIDENT**: 49,788 ads (20.2%)
2. **HARRIS VICTORY FUND**: 32,612 ads (13.2%)
3. **BIDEN VICTORY FUND**: 15,539 ads (6.3%)
4. **DONALD J. TRUMP FOR PRESIDENT 2024**: 15,112 ads (6.1%)
5. **Trump National Committee JFC**: 7,279 ads (3.0%)

### Platform Distribution
- **Facebook + Instagram**: 87% of ads (dominant combination)
- **Facebook Only**: 9.4% of ads
- **Instagram Only**: 3.4% of ads

### Data Quality Insights
- **Missing Values**: Only 1,009 missing values (0.01% of dataset)
- **Data Integrity**: Perfect preservation of totals across aggregations
- **Currency**: 99.9% of ads use USD
- **No Duplicate Rows**: High data quality

### Interesting Patterns
- **Binary Flags**: Most topic/message type columns are binary (0/1)
- **Spend Distribution**: Highly skewed with median spend of $49
- **Geographic Targeting**: Complex JSON structures for regional targeting
- **Demographic Targeting**: Detailed age/gender breakdowns available

## 🛠 Technical Comparison

| Aspect | Pure Python | Pandas | Polars |
|--------|-------------|--------|--------|
| **Performance** | Slowest | Fast | Fastest |
| **Memory Usage** | Most Efficient | 646.55 MB | 512.43 MB |
| **Dependencies** | None | pandas, numpy | polars |
| **Code Complexity** | Highest | Lowest | Medium |
| **Learning Curve** | Steepest | Gentlest | Moderate |

### When to Use Each Approach

- **Pure Python**: Educational purposes, no-dependency environments, custom logic
- **Pandas**: Most common choice, extensive documentation, rich ecosystem
- **Polars**: Performance-critical applications, large datasets, modern syntax

## 📁 Repository Structure

```
Task_02_Descriptive_Stats/
│
├── README.md                 # This file
├── pure_python_stats.py      # Standard library implementation
├── pandas_stats.py           # Pandas implementation  
├── polar_stats.py            # Polars implementation              
```

## 🔬 Statistical Methods Used

### Built-in Functions Utilized
- **`DataFrame.describe()`**: Comprehensive numeric statistics
- **`value_counts()`**: Frequency analysis for categorical data
- **`nunique()`**: Unique value counting
- **`groupby().agg()`**: Data aggregation operations

### Custom Implementations (Pure Python)
- Manual mean and standard deviation calculations
- Custom frequency counting with `collections.Counter`
- Row-by-row data processing
- Memory-efficient CSV parsing
