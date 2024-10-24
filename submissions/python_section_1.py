from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    for i in range(0, len(lst), n):
        # Reverse the chunk of n elements in place
        left, right = i, min(i + n - 1, len(lst) - 1)
        while left < right:
            # Swap the elements at left and right
            lst[left], lst[right] = lst[right], lst[left]
            left += 1
            right -= 1

    return lst
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    length_dict = {}
    
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    
    # Sorting the dictionary by the key (length)
    sorted_length_dict = dict(sorted(length_dict.items()))
    return sorted_length_dict

print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
print(group_by_length(["one", "two", "three", "four"]))

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    def flatten(current_dict, parent_key=''):
        items = []
        
        for k, v in current_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k 
            
            if isinstance(v, dict):
                # If value is a dictionary, recurse into it
                items.extend(flatten(v, new_key).items())
            elif isinstance(v, list):
                # If value is a list, recurse into each element of the list
                for i, item in enumerate(v):
                    list_key = f"{new_key}[{i}]"
                    if isinstance(item, dict):
                        # If list element is a dictionary, recurse
                        items.extend(flatten(item, list_key).items())
                    else:
                        items.append((list_key, item))
            else:
                # Base case: add the key-value pair to the items
                items.append((new_key, v))
        
        return dict(items)

    return flatten(nested_dict)

nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

print(flatten_dict(nested_dict))


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    def backtrack(start):
        # If we have a complete permutation, add it to the result
        if start == len(nums):
            result.append(nums[:])  # Append a copy of the current nums
            return
        
        seen = set()
        for i in range(start, len(nums)):
            # Skip duplicates
            if nums[i] in seen:
                continue
            seen.add(nums[i])

            # Swap the current element with the start element
            nums[start], nums[i] = nums[i], nums[start]

            # Recurse to generate permutations for the next position
            backtrack(start + 1)

            # Swap back (backtrack)
            nums[start], nums[i] = nums[i], nums[start]

    result = []
    nums.sort()  
    backtrack(0)
    
    pass  

    return result

nums = [1, 1, 2]
print(unique_permutations(nums))    


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    def is_valid_dd_mm_yyyy(date_str: str) -> bool:
        """Checks if the date string is in dd-mm-yyyy format."""
        parts = date_str.split('-')
        if len(parts) != 3 or len(parts[0]) != 2 or len(parts[1]) != 2 or len(parts[2]) != 4:
            return False
        return parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit()

    def is_valid_mm_dd_yyyy(date_str: str) -> bool:
        """Checks if the date string is in mm/dd/yyyy format."""
        parts = date_str.split('/')
        if len(parts) != 3 or len(parts[0]) != 2 or len(parts[1]) != 2 or len(parts[2]) != 4:
            return False
        return parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit()

    def is_valid_yyyy_mm_dd(date_str: str) -> bool:
        """Checks if the date string is in yyyy.mm.dd format."""
        parts = date_str.split('.')
        if len(parts) != 3 or len(parts[0]) != 4 or len(parts[1]) != 2 or len(parts[2]) != 2:
            return False
        return parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit()

    words = [word.strip('.,') for word in text.split()] 
    dates = []

    for word in words:
        # Check if the word matches any of the date formats
        if '-' in word and is_valid_dd_mm_yyyy(word):
            dates.append(word)
        elif '/' in word and is_valid_mm_dd_yyyy(word):
            dates.append(word)
        elif '.' in word and is_valid_yyyy_mm_dd(word):
            dates.append(word)

    pass  

    return dates

text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(find_all_dates(text))


# Question 6: Decode Polyline, Convert to DataFrame with Distances
def decode_polyline(polyline_str):
    """Decodes a polyline string into a list of (latitude, longitude) tuples."""
    coordinates = []
    index = 0
    length = len(polyline_str)
    lat = 0
    lng = 0

    while index < length:
        # Decode latitude
        b = 0
        shift = 0
        while True:
            byte = ord(polyline_str[index]) - 63
            index += 1
            b |= (byte & 0x1F) << shift
            shift += 5
            if byte < 0x20:
                break
        lat += ((~b) >> 1) if (b & 1) else b >> 1

        # Decode longitude
        b = 0
        shift = 0
        while True:
            byte = ord(polyline_str[index]) - 63
            index += 1
            b |= (byte & 0x1F) << shift
            shift += 5
            if byte < 0x20:
                break
        lng += ((~b) >> 1) if (b & 1) else b >> 1

        # Append the coordinates to the list
        coordinates.append((lat * 1e-5, lng * 1e-5))

    return coordinates

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the Haversine distance between two points on the Earth."""
    # Radius of the Earth in meters
    R = 6371000

    # Convert latitude and longitude from degrees to radians
    phi1 = lat1 * (3.141592653589793 / 180)
    phi2 = lat2 * (3.141592653589793 / 180)
    delta_phi = (lat2 - lat1) * (3.141592653589793 / 180)
    delta_lambda = (lon2 - lon1) * (3.141592653589793 / 180)

    # Haversine formula
    a = (sin(delta_phi / 2) ** 2 +
         cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Distance in meters
    distance = R * c
    return distance

def sin(x):
    """Approximate sine function using Taylor series."""
    term = x
    result = 0
    n = 0
    while True:
        result += term
        n += 1
        term *= -x * x / (2 * n * (2 * n + 1))
        if abs(term) < 1e-10:
            break
    return result

def cos(x):
    """Approximate cosine function using Taylor series."""
    term = 1
    result = 0
    n = 0
    while True:
        result += term
        n += 1
        term *= -x * x / (2 * n * (2 * n - 1))
        if abs(term) < 1e-10:
            break
    return result

def atan2(y, x):
    """Approximate atan2 using a simple approximation for arctan."""
    if x > 0:
        return atan(y / x)
    elif x < 0 and y >= 0:
        return atan(y / x) + 3.141592653589793
    elif x < 0 and y < 0:
        return atan(y / x) - 3.141592653589793
    elif x == 0 and y > 0:
        return 3.141592653589793 / 2
    elif x == 0 and y < 0:
        return -3.141592653589793 / 2
    else:
        return 0

def atan(x):
    """Approximate arctangent using Taylor series."""
    result = 0
    term = x
    n = 0
    while True:
        result += term
        n += 1
        term *= -(x * x * (2 * n - 1)) / (2 * n + 1)
        if abs(term) < 1e-10:
            break
    return result

def sqrt(x):
    """Approximate square root using Newton's method."""
    if x < 0:
        return None
    guess = x
    while True:
        new_guess = (guess + x / guess) / 2
        if abs(new_guess - guess) < 1e-10:
            return new_guess
        guess = new_guess

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = decode_polyline(polyline_str)
    
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    distances = [0] 
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude']
        lat2, lon2 = df.loc[i, 'latitude'], df.loc[i, 'longitude']
        dist = haversine(lat1, lon1, lat2, lon2)
        distances.append(dist)
    
    df['distance'] = distances
    
    return df

polyline_str = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
df = polyline_to_dataframe(polyline_str)
print(df)    
    

# Question 7: Matrix Rotation and Transformation
def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
        
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    # Rotate by first reversing the rows and then transposing
    rotated = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    # Step 2: Precompute the row sums and column sums of the rotated matrix
    row_sums = [sum(rotated[i]) for i in range(n)]
    col_sums = [sum(rotated[i][j] for i in range(n)) for j in range(n)]
    
    # Step 3: Create a new matrix with the desired transformation
    transformed_matrix = [[row_sums[i] + col_sums[j] - 2 * rotated[i][j] for j in range(n)] for i in range(n)]
    
    return transformed_matrix

matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

result = rotate_and_multiply_matrix(matrix)
for row in result:
    print(row)


# Days of the week for comparison
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Convert day names to numeric indices for easier comparison
day_to_num = {day: i for i, day in enumerate(days_order)}

df = pd.read_csv('C:/Users/nithi/Documents/GitHub/MapUp-DA-Assessment-2024/datasets/dataset-1.csv')

def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    def covers_24_hours(times):
        return '00:00:00' in times and '23:59:59' in times

    # Helper function to check if all days of the week are covered
    def covers_all_days(days):
        return set(day_to_num[day] for day in days) == set(range(7))

    results = []
    
    # Group by `id` and `id_2` to check each unique pair
    for (id_val, id_2_val), group in df.groupby(['id', 'id_2']):
        # Collect all unique days and times from startDay, endDay, startTime, and endTime
        unique_days = set(group['startDay']).union(set(group['endDay']))
        start_times = set(group['startTime'])
        end_times = set(group['endTime'])
        
        # Check if the group covers all 7 days and spans a full 24-hour period
        full_day_coverage = covers_all_days(unique_days)
        full_time_coverage = covers_24_hours(start_times.union(end_times))
        
        # Append the result (False if timestamps are incorrect)
        results.append((id_val, id_2_val, full_day_coverage and full_time_coverage))
    
    # Convert results into a multi-indexed boolean Series
    index = pd.MultiIndex.from_tuples([(r[0], r[1]) for r in results], names=['id', 'id_2'])
    return pd.Series([r[2] for r in results], index=index)


# Assuming `df` is the DataFrame loaded from 'dataset-1.csv'
time_completeness = time_check(df)
print(time_completeness.head())