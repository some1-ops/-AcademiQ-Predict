import pandas as pd
import io

def convert_df_to_arff(df: pd.DataFrame, relation_name="student_academic_data") -> str:
    """
    Converts a pandas DataFrame into a WEKA .arff string format.
    Dynamically maps numeric columns to @attribute <name> numeric
    and object/categorical columns to @attribute <name> {class1, class2}.
    """
    buffer = io.StringIO()
    buffer.write(f"@relation {relation_name}\n\n")

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            buffer.write(f"@attribute '{col}' numeric\n")
        else:
            # Categorical or string
            unique_vals = df[col].dropna().unique()
            # WEKA nominals are comma separated inside {}
            vals_str = ",".join([f"'{str(v)}'" for v in unique_vals])
            buffer.write(f"@attribute '{col}' {{{vals_str}}}\n")
            
    buffer.write("\n@data\n")
    
    # Write data rows
    for _, row in df.iterrows():
        row_vals = []
        for val in row:
            if pd.isna(val):
                row_vals.append("?")
            elif isinstance(val, (int, float)):
                row_vals.append(str(val))
            else:
                row_vals.append(f"'{str(val)}'")
        buffer.write(",".join(row_vals) + "\n")

    return buffer.getvalue()
