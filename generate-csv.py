import csv
import random

def get_int_input(prompt, min_val=None, max_val=None):
    while True:
        try:
            val = int(input(prompt))
            if min_val is not None and val < min_val:
                continue
            if max_val is not None and val > max_val:
                continue
            return val
        except ValueError:
            continue

def get_float_input(prompt, min_val=None, max_val=None):
    while True:
        try:
            val = float(input(prompt))
            if min_val is not None and val < min_val:
                continue
            if max_val is not None and val > max_val:
                continue
            return val
        except ValueError:
            continue

# Step 1: Input headers
num_headers = get_int_input("How many headers? ", 1)
headers = []
for i in range(num_headers):
    name = input(f"Enter name for header #{i + 1}: ")
    headers.append(name)

# Step 2: Rating range
min_rating = get_int_input("Enter minimum rating (e.g., 1): ")
max_rating = get_int_input("Enter maximum rating (e.g., 5): ", min_val=min_rating)

# Step 3: Number of respondents
num_respondents = get_int_input("Enter number of respondents: ", 1)

# Step 4: Target averages
target_averages = []
print(f"Enter target average for each header (float from {min_rating} to {max_rating}):")
for header in headers:
    avg = get_float_input(f"{header}: ", min_rating, max_rating)
    target_averages.append(avg)

# Step 5: Generate respondent scores
respondents_data = []

for _ in range(num_respondents):
    row = []
    for idx, avg in enumerate(target_averages):
        # Weighted random selection with stronger bias towards target average
        choices = list(range(min_rating, max_rating + 1))
        # Use exponential weighting to create stronger bias
        weights = [1 / (abs(val - avg) ** 2 + 0.1) for val in choices]
        score = random.choices(choices, weights=weights, k=1)[0]
        row.append(score)
    respondents_data.append(row)

# Step 6: Write to CSV
with open("output.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(respondents_data)

print("✅ CSV file 'output.csv' has been generated.")
