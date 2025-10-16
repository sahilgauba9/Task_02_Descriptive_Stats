import pandas as pd
import numpy as np
from typing import Dict, Any


def analyze_dataframe(df: pd.DataFrame, title: str = "Dataset Analysis") -> Dict[str, Any]:
    """Analyze entire dataframe and return comprehensive statistics."""
    print(f"\n{title}")
    print("=" * len(title))
    
    # Overall dataset stats
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Basic info about the dataset
    print(f"\nDataset Info:")
    print(f"Total cells: {df.size:,}")
    print(f"Non-null values: {df.count().sum():,}")
    print(f"Null values: {df.isnull().sum().sum():,}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")
    
    # Data types summary
    print(f"\nData Types Summary:")
    dtype_counts = df.dtypes.value_counts()
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
        print(f"  Data type: {df[column].dtype}")
        print(f"  Count: {len(df[column])}")
        print(f"  Non-null: {df[column].count()}")
        print(f"  Null: {df[column].isnull().sum()}")
        
        # Column statistics
        col_stats = {
            'dtype': str(df[column].dtype),
            'count': len(df[column]),
            'non_null': df[column].count(),
            'null_count': df[column].isnull().sum(),
            'unique_count': df[column].nunique()
        }
        
        if pd.api.types.is_numeric_dtype(df[column]):
            print(f"  Type: Numeric")
            
            # Use pandas describe for numeric columns
            desc = df[column].describe()
            print(f"  Unique values: {df[column].nunique()}")
            print(f"  Mean: {desc['mean']:.2f}")
            print(f"  Std: {desc['std']:.2f}")
            print(f"  Min: {desc['min']}")
            print(f"  25%: {desc['25%']}")
            print(f"  50% (Median): {desc['50%']}")
            print(f"  75%: {desc['75%']}")
            print(f"  Max: {desc['max']}")
            print(f"  Sum: {df[column].sum():.2f}")
            
            # Additional numeric stats
            print(f"  Zeros: {(df[column] == 0).sum()}")
            print(f"  Negative values: {(df[column] < 0).sum()}")
            
            col_stats.update({
                'is_numeric': True,
                'mean': desc['mean'],
                'std': desc['std'],
                'min': desc['min'],
                'max': desc['max'],
                'median': desc['50%'],
                'q25': desc['25%'],
                'q75': desc['75%'],
                'sum': df[column].sum(),
                'zeros': (df[column] == 0).sum(),
                'negative': (df[column] < 0).sum()
            })
            
        else:
            print(f"  Type: Non-numeric")
            print(f"  Unique values: {df[column].nunique()}")
            
            # For non-numeric columns, show most frequent values
            if df[column].count() > 0:
                value_counts = df[column].value_counts().head(5)
                print(f"  Most frequent values:")
                for value, count in value_counts.items():
                    display_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    print(f"    '{display_value}': {count}")
                
                col_stats.update({
                    'is_numeric': False,
                    'most_frequent': list(value_counts.items())[:5],
                    'mode': df[column].mode().iloc[0] if len(df[column].mode()) > 0 else None
                })
            else:
                col_stats.update({
                    'is_numeric': False,
                    'most_frequent': [],
                    'mode': None
                })
        
        stats_summary['column_analyses'][column] = col_stats
    
    return stats_summary


def show_numeric_summary(df: pd.DataFrame, title: str):
    """Show summary statistics for all numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        print(f"\n{title} - Numeric Columns Summary:")
        print("=" * (len(title) + 25))
        
        # Use pandas describe for all numeric columns at once
        desc = df[numeric_cols].describe()
        print(desc)
        
        # Additional summary
        print(f"\nAdditional Numeric Statistics:")
        print(f"Total sum across all numeric columns: {df[numeric_cols].sum().sum():.2f}")
        print(f"Columns with zeros: {(df[numeric_cols] == 0).any().sum()}")
        print(f"Columns with negative values: {(df[numeric_cols] < 0).any().sum()}")


def show_categorical_summary(df: pd.DataFrame, title: str):
    """Show summary statistics for all categorical columns."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        print(f"\n{title} - Categorical Columns Summary:")
        print("=" * (len(title) + 30))
        
        for col in categorical_cols:
            print(f"\n{col}:")
            print(f"  Unique values: {df[col].nunique()}")
            print(f"  Most frequent: {df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'}")
            
            # Show top 3 most frequent values
            top_values = df[col].value_counts().head(3)
            print(f"  Top 3 values:")
            for value, count in top_values.items():
                display_value = str(value)[:30] + "..." if len(str(value)) > 30 else str(value)
                percentage = (count / len(df)) * 100
                print(f"    '{display_value}': {count} ({percentage:.1f}%)")


def aggregate_and_analyze(df: pd.DataFrame, group_cols: list, title: str) -> pd.DataFrame:
    """Aggregate dataframe by specified columns and analyze."""
    print(f"\n{title}")
    print("=" * len(title))
    
    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove grouping columns from aggregation columns
    numeric_cols = [col for col in numeric_cols if col not in group_cols]
    categorical_cols = [col for col in categorical_cols if col not in group_cols]
    
    # Create aggregation dictionary
    agg_dict = {}
    
    # For numeric columns, sum them up
    for col in numeric_cols:
        agg_dict[col] = 'sum'
    
    # For categorical columns, take the first non-null value
    for col in categorical_cols:
        agg_dict[col] = lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None
    
    # Perform aggregation
    if agg_dict:
        aggregated_df = df.groupby(group_cols).agg(agg_dict).reset_index()
    else:
        # If no columns to aggregate, just get unique combinations
        aggregated_df = df[group_cols].drop_duplicates().reset_index(drop=True)
    
    print(f"Original dataset: {len(df)} rows")
    print(f"After aggregation by {group_cols}: {len(aggregated_df)} rows")
    print(f"Reduction: {len(df) - len(aggregated_df)} rows ({((len(df) - len(aggregated_df)) / len(df) * 100):.1f}%)")
    
    return aggregated_df


def compare_numeric_totals(original_df: pd.DataFrame, agg_dfs: dict):
    """Compare numeric column totals across different aggregations."""
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return
    
    print(f"\nNumeric Column Totals Comparison:")
    print("=" * 50)
    
    # Create comparison dataframe
    comparison_data = {}
    
    # Original totals
    comparison_data['Original'] = original_df[numeric_cols].sum()
    
    # Aggregated totals
    for name, agg_df in agg_dfs.items():
        # Only include columns that exist in aggregated dataframe
        common_cols = [col for col in numeric_cols if col in agg_df.columns]
        if common_cols:
            comparison_data[name] = agg_df[common_cols].sum().reindex(numeric_cols, fill_value=0)
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df)
    
    # Check if totals are preserved
    print(f"\nTotal Preservation Check:")
    original_total = comparison_df['Original'].sum()
    for col in comparison_df.columns:
        if col != 'Original':
            col_total = comparison_df[col].sum()
            difference = abs(original_total - col_total)
            if difference < 0.01:  # Account for floating point precision
                print(f"  {col}: ✓ Totals preserved")
            else:
                print(f"  {col}: ✗ Difference: {difference:.2f}")


def main():
    """Main function to run the pandas analysis."""
    filename = "2024_fb_ads_president_scored_anon.csv"
    
    try:
        print("Loading dataset with pandas...")
        
        # Load dataset with pandas
        df = pd.read_csv(filename, low_memory=False)
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
        print(f"- Unique pages: {df['page_id'].nunique():,}")
        print(f"- Unique ads: {df['ad_id'].nunique():,}")
        print(f"- Date range: {df['ad_creation_time'].min()} to {df['ad_creation_time'].max()}")
        
        # Most active advertisers
        if 'bylines' in df.columns:
            top_advertisers = df['bylines'].value_counts().head(5)
            print(f"\nTop 5 Advertisers by Ad Count:")
            for advertiser, count in top_advertisers.items():
                print(f"  {advertiser}: {count:,} ads")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{filename}'")
        print("Please ensure the dataset file is in the current directory.")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 