
import csv
import math
from collections import defaultdict, Counter
from typing import Dict, List, Any, Union, Tuple


def is_numeric(value: str) -> bool:
    """Check if a string value can be converted to a number."""
    if not value or value.strip() == '':
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def safe_float(value: str) -> float:
    """Safely convert string to float, return 0.0 if conversion fails."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def calculate_mean(values: List[float]) -> float:
    """Calculate mean of numeric values."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_std_dev(values: List[float]) -> float:
    """Calculate standard deviation of numeric values."""
    if len(values) < 2:
        return 0.0
    
    mean = calculate_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def analyze_column(column_name: str, values: List[str]) -> Dict[str, Any]:
    """Analyze a single column and return statistics."""
    stats = {
        'column_name': column_name,
        'count': len(values),
        'non_empty_count': len([v for v in values if v and v.strip()]),
        'empty_count': len([v for v in values if not v or not v.strip()])
    }
    
    # Check if column contains numeric data
    numeric_values = [safe_float(v) for v in values if is_numeric(v)]
    
    if len(numeric_values) > 0:
        stats['is_numeric'] = True
        stats['numeric_count'] = len(numeric_values)
        stats['mean'] = calculate_mean(numeric_values)
        stats['min'] = min(numeric_values)
        stats['max'] = max(numeric_values)
        stats['std_dev'] = calculate_std_dev(numeric_values)
        stats['sum'] = sum(numeric_values)
    else:
        stats['is_numeric'] = False
        # For non-numeric fields, count unique values and find most frequent
        non_empty_values = [v for v in values if v and v.strip()]
        if non_empty_values:
            value_counts = Counter(non_empty_values)
            stats['unique_count'] = len(value_counts)
            stats['most_frequent'] = value_counts.most_common(5)  # Top 5 most frequent
        else:
            stats['unique_count'] = 0
            stats['most_frequent'] = []
    
    return stats


def load_dataset(filename: str) -> Tuple[List[str], List[List[str]]]:
    """Load CSV dataset using standard library."""
    print(f"Loading dataset from {filename}...")
    
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)  # Read header row
        
        data = []
        row_count = 0
        for row in csv_reader:
            # Ensure row has same number of columns as headers
            while len(row) < len(headers):
                row.append('')
            data.append(row[:len(headers)])  # Trim extra columns if any
            row_count += 1
            
            if row_count % 50000 == 0:
                print(f"  Loaded {row_count} rows...")
    
    print(f"Dataset loaded: {len(headers)} columns, {len(data)} rows")
    return headers, data


def analyze_dataset(headers: List[str], data: List[List[str]], title: str = "Dataset Analysis") -> Dict[str, Any]:
    """Analyze entire dataset and return comprehensive statistics."""
    print(f"\n{title}")
    print("=" * len(title))
    
    # Overall dataset stats
    overall_stats = {
        'title': title,
        'total_rows': len(data),
        'total_columns': len(headers),
        'column_analyses': {}
    }
    
    # Analyze each column
    for i, column_name in enumerate(headers):
        column_values = [row[i] if i < len(row) else '' for row in data]
        column_stats = analyze_column(column_name, column_values)
        overall_stats['column_analyses'][column_name] = column_stats
        
        # Print column statistics
        print(f"\nColumn: {column_name}")
        print(f"  Count: {column_stats['count']}")
        print(f"  Non-empty: {column_stats['non_empty_count']}")
        print(f"  Empty: {column_stats['empty_count']}")
        
        if column_stats['is_numeric']:
            print(f"  Type: Numeric")
            print(f"  Numeric values: {column_stats['numeric_count']}")
            print(f"  Mean: {column_stats['mean']:.2f}")
            print(f"  Min: {column_stats['min']}")
            print(f"  Max: {column_stats['max']}")
            print(f"  Std Dev: {column_stats['std_dev']:.2f}")
            print(f"  Sum: {column_stats['sum']:.2f}")
        else:
            print(f"  Type: Non-numeric")
            print(f"  Unique values: {column_stats['unique_count']}")
            if column_stats['most_frequent']:
                print(f"  Most frequent values:")
                for value, count in column_stats['most_frequent']:
                    display_value = value[:50] + "..." if len(value) > 50 else value
                    print(f"    '{display_value}': {count}")
    
    return overall_stats


def aggregate_by_columns(headers: List[str], data: List[List[str]], group_columns: List[str]) -> Tuple[List[str], List[List[str]]]:
    """Aggregate data by specified columns."""
    print(f"\nAggregating data by: {', '.join(group_columns)}")
    
    # Find indices of grouping columns
    group_indices = []
    for col in group_columns:
        if col in headers:
            group_indices.append(headers.index(col))
        else:
            print(f"Warning: Column '{col}' not found in headers")
            return headers, data
    
    # Group data by the specified columns
    groups = defaultdict(list)
    for row in data:
        # Create group key from the specified columns
        group_key = tuple(row[i] if i < len(row) else '' for i in group_indices)
        groups[group_key].append(row)
    
    print(f"Found {len(groups)} unique groups")
    
    # Create aggregated dataset
    # For numeric columns, we'll sum them up
    # For non-numeric columns, we'll take the first non-empty value
    aggregated_data = []
    
    for group_key, group_rows in groups.items():
        if not group_rows:
            continue
            
        # Initialize aggregated row with group key values
        agg_row = [''] * len(headers)
        for i, idx in enumerate(group_indices):
            agg_row[idx] = group_key[i]
        
        # Aggregate other columns
        for col_idx, header in enumerate(headers):
            if col_idx in group_indices:
                continue  # Already set
                
            column_values = [row[col_idx] if col_idx < len(row) else '' for row in group_rows]
            
            # Check if column is numeric
            numeric_values = [safe_float(v) for v in column_values if is_numeric(v)]
            
            if len(numeric_values) > 0:
                # For numeric columns, sum the values
                agg_row[col_idx] = str(sum(numeric_values))
            else:
                # For non-numeric columns, take first non-empty value
                non_empty_values = [v for v in column_values if v and v.strip()]
                agg_row[col_idx] = non_empty_values[0] if non_empty_values else ''
        
        aggregated_data.append(agg_row)
    
    return headers, aggregated_data


def main():
    """Main function to run the analysis."""
    filename = "2024_fb_ads_president_scored_anon.csv"
    
    try:
        # Load the dataset
        headers, data = load_dataset(filename)
        
        # 1. Analyze the original dataset
        original_stats = analyze_dataset(headers, data, "Original Dataset Analysis")
        
        # 2. Aggregate by page_id and analyze
        headers_page, data_page = aggregate_by_columns(headers, data, ['page_id'])
        page_stats = analyze_dataset(headers_page, data_page, "Analysis After Aggregation by page_id")
        
        # 3. Aggregate by (page_id, ad_id) and analyze
        headers_page_ad, data_page_ad = aggregate_by_columns(headers, data, ['page_id', 'ad_id'])
        page_ad_stats = analyze_dataset(headers_page_ad, data_page_ad, "Analysis After Aggregation by (page_id, ad_id)")
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Original dataset: {original_stats['total_rows']} rows, {original_stats['total_columns']} columns")
        print(f"After page_id aggregation: {page_stats['total_rows']} rows")
        print(f"After (page_id, ad_id) aggregation: {page_ad_stats['total_rows']} rows")
        
        # Find numeric columns and show their totals across all three analyses
        print(f"\nNumeric Column Totals Comparison:")
        print(f"{'Column':<30} {'Original':<15} {'page_id':<15} {'page_id+ad_id':<15}")
        print("-" * 75)
        
        for col_name in headers:
            orig_stats = original_stats['column_analyses'][col_name]
            page_stats_col = page_stats['column_analyses'][col_name]
            page_ad_stats_col = page_ad_stats['column_analyses'][col_name]
            
            if orig_stats['is_numeric']:
                orig_sum = orig_stats.get('sum', 0)
                page_sum = page_stats_col.get('sum', 0)
                page_ad_sum = page_ad_stats_col.get('sum', 0)
                
                print(f"{col_name:<30} {orig_sum:<15.2f} {page_sum:<15.2f} {page_ad_sum:<15.2f}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{filename}'")
        print("Please ensure the dataset file is in the current directory.")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main() 