
import pandas as pd

def read_and_join_csv_files(file_paths, chunk_size=10000):
    """
    Read multiple CSV files efficiently using chunking with Pandas and join them into one DataFrame.
    
    Args:
    - file_paths (dict): Dictionary containing file paths for each CSV file.
    - chunk_size (int, optional): Number of rows to read at a time. Default is 10,000.
    
    Returns:
    - DataFrame: Joined DataFrame containing data from all CSV files.
    """
    # Initialize empty dictionary to store DataFrames for each CSV file
    dfs = {}
    
    # Iterate over each file path and read CSV files in chunks
    for key, file_path in file_paths.items():
        # Initialize empty list to store chunks for the current DataFrame
        chunks = []
        
        # Iterate over chunks for the current DataFrame
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Append the chunk to the list of chunks for the current DataFrame
            chunks.append(chunk)
        
        # Concatenate the chunks for the current DataFrame into a single DataFrame
        df = pd.concat(chunks)
        
        # Store the DataFrame in the dictionary with the corresponding key
        dfs[key] = df
    
    # Join DataFrames
    joined_df = dfs["train"].merge(dfs["members"], on='msno', how='left')
    joined_df = joined_df.merge(dfs["transactions"], on='msno', how='left')
    joined_df = joined_df.merge(dfs["logs"], on='msno', how='left')
    
    return joined_df