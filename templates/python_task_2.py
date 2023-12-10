import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here

    import pandas as pd
import networkx as nx

def calculate_distance_matrix(dataset_path):
    df = pd.read_csv(dataset_path)

    G = nx.from_pandas_edgelist(df, 'id_start', 'id_end', ['distance'])

    distance_matrix = nx.floyd_warshall_numpy(G, weight='distance', nodelist=sorted(G.nodes))

    distance_df = pd.DataFrame(distance_matrix, index=sorted(G.nodes), columns=sorted(G.nodes))

    distance_df.values[[range(len(distance_df))]*2] = 0

    return distance_df

    return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here

    def unroll_distance_matrix(df):
    unrolled_data = []

    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']

    
        for start in df['id_start'].unique():
            for end in df['id_end'].unique():
        
                if start != end:
                    unrolled_data.append({'id_start': start, 'id_end': end, 'distance': distance})


    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

    return df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here

    def find_ids_within_ten_percentage_threshold(df, reference_value):
    reference_rows = df[df['id_start'] == reference_value]
    average_distance = reference_rows['distance'].mean()
    
    lower_threshold = average_distance - (0.1 * average_distance)
    upper_threshold = average_distance + (0.1 * average_distance)

    within_threshold = df[(df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]


    result_ids = sorted(within_threshold['id_start'].unique())
    return result_ids
    

    return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    def calculate_toll_rate(df):
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df

result_df = calculate_toll_rate(df)
print(result_df)


    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    from datetime import time, timedelta

def calculate_time_based_toll_rates(df):
    time_ranges = [
        {'start': time(0, 0, 0), 'end': time(10, 0, 0), 'weekday_factor': 0.8, 'weekend_factor': 0.7},
        {'start': time(10, 0, 0), 'end': time(18, 0, 0), 'weekday_factor': 1.2, 'weekend_factor': 0.7},
        {'start': time(18, 0, 0), 'end': time(23, 59, 59), 'weekday_factor': 0.8, 'weekend_factor': 0.7}
    ]


    df['start_day'] = df['end_day'] = df['start_time'] = df['end_time'] = None

    
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']

        
        for time_range in time_ranges:
            
            df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'start_day'] = 'Monday'
            df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'end_day'] = 'Sunday'
            df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'start_time'] = time_range['start']
            df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), 'end_time'] = time_range['end']

            
            weekday_mask = (df['start_day'] != 'Saturday') & (df['start_day'] != 'Sunday')
            weekend_mask = ~weekday_mask

            df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end) & weekday_mask, 'moto':'truck'] *= time_range['weekday_factor']
            df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end) & weekend_mask, 'moto':'truck'] *= time_range['weekend_factor']

    
    df['start_time'] = pd.to_datetime(df['start_time'], format='%H:%M:%S').dt.time
    df['end_time'] = pd.to_datetime(df['end_time'], format='%H:%M:%S').dt.time

    return df

result_df = calculate_time_based_toll_rates(df)
print(result_df)

    return df
