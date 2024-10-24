import pandas as pd

df = pd.read_csv('C:/Users/nithi/Documents/GitHub/MapUp-DA-Assessment-2024/datasets/dataset-2.csv')


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    ids = sorted(set(df['id_start']).union(set(df['id_end'])))
    
    # Initialize a DataFrame filled with infinity (represented as float('inf')) for unknown distances
    distance_matrix = pd.DataFrame(float('inf'), index=ids, columns=ids)

    # Set diagonal to 0 (distance from a point to itself is 0)
    for id in ids:
        distance_matrix.at[id, id] = 0

    # Fill in the direct distances from the dataset
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']
        
        # Since the matrix is symmetric, fill both [id_start, id_end] and [id_end, id_start]
        distance_matrix.at[id_start, id_end] = distance
        distance_matrix.at[id_end, id_start] = distance

    # Apply the Floyd-Warshall algorithm to compute cumulative distances
    for k in ids:
        for i in ids:
            for j in ids:
                # Check if going through node 'k' provides a shorter path from 'i' to 'j'
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    return distance_matrix
distance_matrix = calculate_distance_matrix(df)
print(distance_matrix)



def unroll_distance_matrix(distance_matrix)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here

    unrolled_data = []
    # Iterate over the matrix to extract the id_start, id_end, and distance
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:  # Exclude diagonal (where id_start == id_end)
                distance = distance_matrix.at[id_start, id_end]
                unrolled_data.append([id_start, id_end, distance])

    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])
    return unrolled_df

unrolled_df = unroll_distance_matrix(distance_matrix)
print(unrolled_df)


def find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)->pd.DataFrame():
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
    reference_distances = unrolled_df[unrolled_df['id_start'] == reference_id]
    
    # Calculate the average distance for the reference_id
    avg_distance = reference_distances['distance'].mean()
    
    # Define the 10% threshold range
    lower_bound = avg_distance * 0.9
    upper_bound = avg_distance * 1.1
    
    # Find ids whose distances fall within the 10% threshold
    ids_within_threshold = reference_distances[
        (reference_distances['distance'] >= lower_bound) & 
        (reference_distances['distance'] <= upper_bound)
    ]['id_end'].unique()

    # Return the sorted list of IDs
    return sorted(ids_within_threshold)

reference_id = 1001400  # Example reference id
ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
print(ids_within_threshold)


def calculate_toll_rate(unrolled_df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    # Define rate coefficients
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Add columns for each vehicle type, calculating tolls by multiplying distance by rate
    unrolled_df['moto'] = unrolled_df['distance'] * rates['moto']
    unrolled_df['car'] = unrolled_df['distance'] * rates['car']
    unrolled_df['rv'] = unrolled_df['distance'] * rates['rv']
    unrolled_df['bus'] = unrolled_df['distance'] * rates['bus']
    unrolled_df['truck'] = unrolled_df['distance'] * rates['truck']

    return unrolled_df



toll_rates_df = calculate_toll_rate(unrolled_df)
print(toll_rates_df.head())


def calculate_time_based_toll_rates(toll_rates_df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    # Define days of the week
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekend = ['Saturday', 'Sunday']
    
    # Define time ranges and discount factors for weekdays
    weekday_time_ranges = [
        ("00:00:00", "10:00:00", 0.8),  # 00:00 to 10:00
        ("10:00:00", "18:00:00", 1.2),  # 10:00 to 18:00
        ("18:00:00", "23:59:59", 0.8)   # 18:00 to 23:59
    ]
    
    # Weekend constant discount factor
    weekend_discount = 0.7
    weekend_time_range = [("00:00:00", "23:59:59", weekend_discount)]
    
    # List to store new rows with time-based toll rates
    time_based_rows = []
    
    # Iterate over each unique id_start, id_end pair
    for _, row in toll_rates_df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        
        # Apply the weekday discount factors
        for day in weekdays:
            for start_time, end_time, discount in weekday_time_ranges:
                new_row = row.copy()
                new_row['start_day'] = day
                new_row['end_day'] = day
                new_row['start_time'] = start_time
                new_row['end_time'] = end_time
                
                # Apply discount factor to all vehicle columns
                new_row['moto'] *= discount
                new_row['car'] *= discount
                new_row['rv'] *= discount
                new_row['bus'] *= discount
                new_row['truck'] *= discount
                
                time_based_rows.append(new_row)
        
        # Apply the weekend discount factor
        for day in weekend:
            for start_time, end_time, discount in weekend_time_range:
                new_row = row.copy()
                new_row['start_day'] = day
                new_row['end_day'] = day
                new_row['start_time'] = start_time
                new_row['end_time'] = end_time
                
                # Apply discount factor to all vehicle columns
                new_row['moto'] *= discount
                new_row['car'] *= discount
                new_row['rv'] *= discount
                new_row['bus'] *= discount
                new_row['truck'] *= discount
                
                time_based_rows.append(new_row)
    
    # Create a new DataFrame with the time-based toll rates
    time_based_toll_rates_df = pd.DataFrame(time_based_rows)

    cols = ['id_start', 'id_end', 'distance', 'start_day', 'start_time', 'end_day', 'end_time', 
            'moto', 'car', 'rv', 'bus', 'truck']
    time_based_toll_rates_df = time_based_toll_rates_df[cols]

    return time_based_toll_rates_df

time_based_toll_rates_df = calculate_time_based_toll_rates(toll_rates_df)
print(time_based_toll_rates_df.head())
