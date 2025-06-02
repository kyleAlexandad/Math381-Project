import csv
import os
import random

def create_circle_in_rectangular_csv(input_filepath, output_filepath, radius, fill_value,
                                     center_row=None, center_col=None,
                                     do_not_replace_values=None,
                                     random_center_target_value="0"):
    """
    Reads a rectangular CSV. If center_row/col are not provided, it randomly selects
    a center from cells matching random_center_target_value.
    Places a 'circle' of fill_value around the center index.
    Optionally, a specific value can be preserved and not replaced.

    Args:
        input_filepath (str): Path to the input CSV file.
        output_filepath (str): Path to save the modified CSV file.
        radius (int): The radius of the circle (Manhattan distance).
        fill_value (any): The value to place in the circle cells.
                          It will be converted to a string for CSV writing.
        center_row (int, optional): Row index of the circle's center. If None, random selection is attempted.
        center_col (int, optional): Column index of the circle's center. If None, random selection is attempted.
        do_not_replace_value (any, optional): If a cell's current value matches this,
                                             it will not be replaced. Defaults to None.
        random_center_target_value (str, optional): The value to look for when randomly selecting a center.
                                                    Defaults to "0".
    """
    data = []
    try:
        with open(input_filepath, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            for r in reader:
                data.append(list(r))
    except FileNotFoundError:
        print(f"Error: Input file '{input_filepath}' not found.")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if not data:
        print("Error: CSV file is empty.")
        return

    num_rows = len(data)
    num_cols = len(data[0]) if num_rows > 0 else 0

    if num_cols == 0 and num_rows > 0:
        print("Error: CSV rows are empty (no columns).")
        return

    # Determine center coordinates
    if center_row is None or center_col is None:
        print(f"Attempting to find a random center with value '{random_center_target_value}'...")
        possible_centers = []
        for r_idx in range(num_rows):
            for c_idx in range(num_cols):
                if data[r_idx][c_idx] == str(random_center_target_value): # Compare with string version
                    possible_centers.append((r_idx, c_idx))

        if not possible_centers:
            print(f"Error: No cells found with value '{random_center_target_value}' to pick a random center.")
            return
        
        center_row, center_col = random.choice(possible_centers)
        print(f"Randomly selected center: ({center_row}, {center_col})")
    else:
        # Validate provided center coordinates
        if not (0 <= center_row < num_rows):
            print(f"Error: Provided center row {center_row} is out of bounds (0-{num_rows-1}).")
            return
        if not (0 <= center_col < num_cols):
            print(f"Error: Provided center column {center_col} is out of bounds (0-{num_cols-1}).")
            return
        print(f"Using provided center: ({center_row}, {center_col})")


    fill_value_str = str(fill_value)

    do_not_replace_values_str_list = []
    if do_not_replace_values is not None:
        if isinstance(do_not_replace_values, list):
            do_not_replace_values_str_list = [str(val) for val in do_not_replace_values]
        else: # Assume it's a single value
            do_not_replace_values_str_list = [str(do_not_replace_values)]
    if do_not_replace_values_str_list: # Check if the list is not empty
        print(f"Values that will not be replaced: {do_not_replace_values_str_list}")

    modified_data = [list(row_content) for row_content in data]

    for r_idx in range(num_rows):
        for c_idx in range(num_cols):
            distance = abs(r_idx - center_row) + abs(c_idx - center_col)
            if distance <= radius:
                if do_not_replace_values and modified_data[r_idx][c_idx] in do_not_replace_values:
                        continue
                modified_data[r_idx][c_idx] = fill_value_str
    try:
        with open(output_filepath, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(modified_data)
        print(f"Successfully created circle and saved to '{output_filepath}'")
        return (center_col, center_row)
    except Exception as e:
        print(f"Error writing CSV: {e}")

if __name__ == "__main__":
    input_file = "data/sunset_heatmap.csv"
    output_file = "data/defender_sunset_heatmap.csv"

    # --- Example 1: Randomly chosen center (looking for "0") ---
    print(f"\n--- Running Example 1 (Random Center) ---")
    circle_radius_random = 20
    value_to_fill_random = 4
    value_to_preserve = ["3", "1"] 

    center_col, center_row = create_circle_in_rectangular_csv(
        input_filepath=input_file,
        output_filepath=output_file,
        radius=circle_radius_random,
        fill_value=value_to_fill_random,
        do_not_replace_values=value_to_preserve,
        random_center_target_value="0" 
    )

    create_circle_in_rectangular_csv(
        input_filepath=output_file,
        output_filepath=output_file,
        center_row=center_row,
        center_col=center_col,
        radius=10,
        fill_value=2,
        do_not_replace_values=value_to_preserve
    )
