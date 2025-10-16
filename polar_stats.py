import polars as pl
from typing import Dict, Any
import sys


def analyze_dataframe(df: pl.DataFrame, title: str = "Dataset Analysis") -> Dict[str, Any]:
    """Analyze entire dataframe and return comprehensive statistics."""
    print(f"\n{title}")
    print("=" * len(title))
    
    # Overall dataset stats
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Calculate memory usage (approximation)
    memory_mb = df.estimated_size('mb')
    print(f"Estimated memory usage: {memory_mb:.2f} MB")
    
    # Basic info about the dataset
    print(f"\nDataset Info:")
    print(f"Total cells: {df.shape[0] * df.shape[1]:,}")
    
    # Count null values across all columns
    null_counts = df.null_count()
    total_nulls = sum(null_counts.row(0))
    total_non_nulls = (df.shape[0] * df.shape[1]) - total_nulls
    
    print(f"Non-null values: {total_non_nulls:,}")
    print(f"Null values: {total_nulls:,}")
    
    # Check for duplicate rows
    duplicate_count = df.shape[0] - df.n_unique()
    print(f"Duplicate rows: {duplicate_count:,}")
    
    # Data types summary
    print(f"\nData Types Summary:")
    dtype_counts = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Analyze each column
    stats_summary = {
        'title': title,
        'shape': df.shape,
        'column_analyses': {}
    }
    
    print(f"\nColumn-by-Column Analysis:")
    print("-" * 50)
    
    for column in df.columns:
        print(f"\nColumn: {column}")
        col_series = df[column]
        col_dtype = col_series.dtype
        
        print(f"  Data type: {col_dtype}")
        print(f"  Count: {df.shape[0]}")
        
        # Count non-null values
        non_null_count = col_series.drop_nulls().len()
        null_count = df.shape[0] - non_null_count
        print(f"  Non-null: {non_null_count}")
        print(f"  Null: {null_count}")
        
        # Column statistics
        col_stats = {
            'dtype': str(col_dtype),
            'count': df.shape[0],
            'non_null': non_null_count,
            'null_count': null_count,
            'unique_count': col_series.n_unique()
        }
        
        # Check if column is numeric
        if col_dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]:
            print(f"  Type: Numeric")
            
            # Use polars describe for numeric columns
            desc_stats = df.select(pl.col(column).drop_nulls()).describe()
            
            print(f"  Unique values: {col_series.n_unique()}")
            
            # Extract statistics from describe
            stats_dict = {}
            for row in desc_stats.iter_rows(named=True):
                stats_dict[row['statistic']] = row[column]
            
            print(f"  Mean: {stats_dict.get('mean', 0):.2f}")
            print(f"  Std: {stats_dict.get('std', 0):.2f}")
            print(f"  Min: {stats_dict.get('min', 0)}")
            print(f"  25%: {stats_dict.get('25%', 0)}")
            print(f"  50% (Median): {stats_dict.get('50%', 0)}")
            print(f"  75%: {stats_dict.get('75%', 0)}")
            print(f"  Max: {stats_dict.get('max', 0)}")
            
            # Calculate sum
            col_sum = col_series.sum()
            print(f"  Sum: {col_sum:.2f}")
            
            # Additional numeric stats
            zeros = (col_series == 0).sum()
            negatives = (col_series < 0).sum()
            print(f"  Zeros: {zeros}")
            print(f"  Negative values: {negatives}")
            
            col_stats.update({
                'is_numeric': True,
                'mean': stats_dict.get('mean', 0),
                'std': stats_dict.get('std', 0),
                'min': stats_dict.get('min', 0),
                'max': stats_dict.get('max', 0),
                'median': stats_dict.get('50%', 0),
                'q25': stats_dict.get('25%', 0),
                'q75': stats_dict.get('75%', 0),
                'sum': col_sum,
                'zeros': zeros,
                'negative': negatives
            })
            
        else:
            print(f"  Type: Non-numeric")
            print(f"  Unique values: {col_series.n_unique()}")
            
            # For non-numeric columns, show most frequent values
            if non_null_count > 0:
                # Fixed value_counts usage for Polars
                value_counts = df.filter(pl.col(column).is_not_null()).select(pl.col(column).value_counts().alias("value_counts")).unnest("value_counts").sort("count", descending=True).head(5)
                print(f"  Most frequent values:")
                
                most_frequent_list = []
                for row in value_counts.iter_rows(named=True):
                    value = row[column]
                    count = row['count']
                    display_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    print(f"    '{display_value}': {count}")
                    most_frequent_list.append((value, count))
                
                # Get mode (most frequent value)
                mode_value = value_counts.row(0)[0] if value_counts.shape[0] > 0 else None
                
                col_stats.update({
                    'is_numeric': False,
                    'most_frequent': most_frequent_list,
                    'mode': mode_value
                })
            else:
                col_stats.update({
                    'is_numeric': False,
                    'most_frequent': [],
                    'mode': None
                })
        
        stats_summary['column_analyses'][column] = col_stats
    
    return stats_summary


def show_numeric_summary(df: pl.DataFrame, title: str):
    """Show summary statistics for all numeric columns."""
    # Get numeric columns
    numeric_cols = []
    for col in df.columns:
        if df[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]:
            numeric_cols.append(col)
    
    if len(numeric_cols) > 0:
        print(f"\n{title} - Numeric Columns Summary:")
        print("=" * (len(title) + 25))
        
        # Use polars describe for all numeric columns at once
        desc = df.select(numeric_cols).describe()
        print(desc)
        
        # Additional summary
        total_sum = 0
        cols_with_zeros = 0
        cols_with_negatives = 0
        
        for col in numeric_cols:
            col_sum = df[col].sum()
            total_sum += col_sum
            
            if (df[col] == 0).any():
                cols_with_zeros += 1
            if (df[col] < 0).any():
                cols_with_negatives += 1
        
        print(f"\nAdditional Numeric Statistics:")
        print(f"Total sum across all numeric columns: {total_sum:.2f}")
        print(f"Columns with zeros: {cols_with_zeros}")
        print(f"Columns with negative values: {cols_with_negatives}")


def show_categorical_summary(df: pl.DataFrame, title: str):
    """Show summary statistics for all categorical columns."""
    # Get categorical columns (non-numeric)
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype not in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]:
            categorical_cols.append(col)
    
    if len(categorical_cols) > 0:
        print(f"\n{title} - Categorical Columns Summary:")
        print("=" * (len(title) + 30))
        
        for col in categorical_cols:
            print(f"\n{col}:")
            
            unique_count = df[col].n_unique()
            print(f"  Unique values: {unique_count}")
            
            # Get mode (most frequent value)
            non_null_values = df.filter(pl.col(col).is_not_null())
            if non_null_values.shape[0] > 0:
                try:
                    mode_result = non_null_values.select(pl.col(col).value_counts().alias("value_counts")).unnest("value_counts").sort("count", descending=True).head(1)
                    mode_value = mode_result.row(0)[0] if mode_result.shape[0] > 0 else 'N/A'
                    print(f"  Most frequent: {mode_value}")
                    
                    # Show top 3 most frequent values
                    top_values = non_null_values.select(pl.col(col).value_counts().alias("value_counts")).unnest("value_counts").sort("count", descending=True).head(3)
                    print(f"  Top 3 values:")
                    for row in top_values.iter_rows(named=True):
                        value = row[col]
                        count = row['count']
                        display_value = str(value)[:30] + "..." if len(str(value)) > 30 else str(value)
                        percentage = (count / df.shape[0]) * 100
                        print(f"    '{display_value}': {count} ({percentage:.1f}%)")
                except Exception as e:
                    print(f"  Most frequent: N/A (Error: {e})")
            else:
                print(f"  Most frequent: N/A")


def aggregate_and_analyze(df: pl.DataFrame, group_cols: list, title: str) -> pl.DataFrame:
    """Aggregate dataframe by specified columns and analyze."""
    print(f"\n{title}")
    print("=" * len(title))
    
    # Separate numeric and non-numeric columns
    numeric_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if col not in group_cols:
            if df[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
    
    # Create aggregation expressions
    agg_exprs = []
    
    # For numeric columns, sum them up
    for col in numeric_cols:
        agg_exprs.append(pl.col(col).sum().alias(col))
    
    # For categorical columns, take the first non-null value
    for col in categorical_cols:
        agg_exprs.append(pl.col(col).drop_nulls().first().alias(col))
    
    # Perform aggregation
    if agg_exprs:
        aggregated_df = df.group_by(group_cols).agg(agg_exprs)
    else:
        # If no columns to aggregate, just get unique combinations
        aggregated_df = df.select(group_cols).unique()
    
    print(f"Original dataset: {df.shape[0]} rows")
    print(f"After aggregation by {group_cols}: {aggregated_df.shape[0]} rows")
    reduction = df.shape[0] - aggregated_df.shape[0]
    percentage = (reduction / df.shape[0]) * 100
    print(f"Reduction: {reduction} rows ({percentage:.1f}%)")
    
    return aggregated_df


def compare_numeric_totals(original_df: pl.DataFrame, agg_dfs: dict):
    """Compare numeric column totals across different aggregations."""
    # Get numeric columns
    numeric_cols = []
    for col in original_df.columns:
        if original_df[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]:
            numeric_cols.append(col)
    
    if len(numeric_cols) == 0:
        return
    
    print(f"\nNumeric Column Totals Comparison:")
    print("=" * 50)
    
    # Create comparison data
    comparison_data = {}
    
    # Original totals
    original_totals = {}
    for col in numeric_cols:
        original_totals[col] = original_df[col].sum()
    comparison_data['Original'] = original_totals
    
    # Aggregated totals
    for name, agg_df in agg_dfs.items():
        agg_totals = {}
        for col in numeric_cols:
            if col in agg_df.columns:
                agg_totals[col] = agg_df[col].sum()
            else:
                agg_totals[col] = 0
        comparison_data[name] = agg_totals
    
    # Print comparison table
    print(f"{'Column':<40} {'Original':<15} {'page_id_agg':<15} {'page_ad_agg':<15}")
    print("-" * 85)
    
    for col in numeric_cols:
        original_val = comparison_data['Original'][col]
        page_val = comparison_data.get('page_id_agg', {}).get(col, 0)
        page_ad_val = comparison_data.get('page_ad_agg', {}).get(col, 0)
        print(f"{col:<40} {original_val:<15.2f} {page_val:<15.2f} {page_ad_val:<15.2f}")
    
    # Check if totals are preserved
    print(f"\nTotal Preservation Check:")
    original_total = sum(comparison_data['Original'].values())
    for name, totals in comparison_data.items():
        if name != 'Original':
            agg_total = sum(totals.values())
            difference = abs(original_total - agg_total)
            if difference < 0.01:  # Account for floating point precision
                print(f"  {name}: ✓ Totals preserved")
            else:
                print(f"  {name}: ✗ Difference: {difference:.2f}")


def main():
    """Main function to run the Polars analysis."""
    filename = "2024_fb_ads_president_scored_anon.csv"
    
    try:
        print("Loading dataset with Polars...")
        
        # Load dataset with polars
        df = pl.read_csv(filename, infer_schema_length=10000)
        print(f"Dataset loaded successfully!")
        
        # 1. Analyze the original dataset
        print("\n" + "="*80)
        original_stats = analyze_dataframe(df, "Original Dataset Analysis")
        show_numeric_summary(df, "Original Dataset")
        show_categorical_summary(df, "Original Dataset")
        
        # 2. Aggregate by page_id and analyze
        print("\n" + "="*80)
        df_page = aggregate_and_analyze(df, ['page_id'], "Aggregation by page_id")
        page_stats = analyze_dataframe(df_page, "Analysis After Aggregation by page_id")
        show_numeric_summary(df_page, "page_id Aggregated")
        
        # 3. Aggregate by (page_id, ad_id) and analyze
        print("\n" + "="*80)
        df_page_ad = aggregate_and_analyze(df, ['page_id', 'ad_id'], "Aggregation by (page_id, ad_id)")
        page_ad_stats = analyze_dataframe(df_page_ad, "Analysis After Aggregation by (page_id, ad_id)")
        show_numeric_summary(df_page_ad, "(page_id, ad_id) Aggregated")
        
        # 4. Compare numeric totals
        print("\n" + "="*80)
        agg_dfs = {
            'page_id_agg': df_page,
            'page_ad_agg': df_page_ad
        }
        compare_numeric_totals(df, agg_dfs)
        
        # 5. Final Summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print(f"Original dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"After page_id aggregation: {df_page.shape[0]:,} rows")
        print(f"After (page_id, ad_id) aggregation: {df_page_ad.shape[0]:,} rows")
        
        print(f"\nDataset Insights:")
        print(f"- Total estimated spend: ${df['estimated_spend'].sum():,.2f}")
        print(f"- Total estimated impressions: {df['estimated_impressions'].sum():,}")
        print(f"- Unique pages: {df['page_id'].n_unique():,}")
        print(f"- Unique ads: {df['ad_id'].n_unique():,}")
        print(f"- Date range: {df['ad_creation_time'].min()} to {df['ad_creation_time'].max()}")
        
        # Most active advertisers
        if 'bylines' in df.columns:
            try:
                top_advertisers = df.filter(pl.col('bylines').is_not_null()).select(pl.col('bylines').value_counts().alias("value_counts")).unnest("value_counts").sort("count", descending=True).head(5)
                print(f"\nTop 5 Advertisers by Ad Count:")
                for row in top_advertisers.iter_rows(named=True):
                    advertiser = row['bylines']
                    count = row['count']
                    print(f"  {advertiser}: {count:,} ads")
            except Exception as e:
                print(f"\nTop advertisers analysis failed: {e}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{filename}'")
        print("Please ensure the dataset file is in the current directory.")
    except ImportError:
        print("Error: Polars is not installed. Please install it with: pip install polars")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 