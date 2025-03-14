import csv
import random
import os

def generate_survey_data():
    # Get total number of respondents
    while True:
        try:
            num_respondents = int(input("Enter the total number of respondents: "))
            if num_respondents <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    # Define rating categories
    rating_categories = [
        "Objectives_Met",
        "Venue_Rating",
        "Schedule_Rating",
        "Allowance_Rating",
        "Speaker_Rating",
        "Facilitator_Rating",
        "Participant_Rating"
    ]

    # Pre-defined departments
    department_names = ["SITE", "SBAHM", "SOM", "SNAHS", "BEU"]
    departments = {}

    # Get student counts for each department
    print("\nEnter student count for each department:")
    for dept_name in department_names:
        while True:
            try:
                student_count = int(input(f"Enter student count for {dept_name}: "))
                if student_count <= 0:
                    print("Please enter a positive number.")
                    continue
                break
            except ValueError:
                print("Please enter a valid number.")
        departments[dept_name] = student_count

    # Get specific rating counts for each category
    rating_counts = {}

    for category in rating_categories:
        print(f"\nEnter rating counts for {category}:")
        category_counts = {}
        total_rated = 0

        for rating in range(4):  # 0, 1, 2, 3
            while True:
                try:
                    count = int(input(f"Number of respondents who rated '{rating}': "))
                    if count < 0:
                        print("Please enter a non-negative number.")
                        continue

                    total_rated += count
                    category_counts[rating] = count

                    # Check if we've exceeded the total respondents
                    if total_rated > num_respondents:
                        over = total_rated - num_respondents
                        print(f"Error: Total exceeds number of respondents by {over}.")
                        total_rated -= count
                        continue

                    # If this is the last rating (3) and we haven't reached total respondents yet
                    if rating == 3 and total_rated < num_respondents:
                        remaining = num_respondents - total_rated
                        print(f"Warning: Total is {remaining} short of total respondents.")
                        print(f"Adjusting count for rating '3' to {count + remaining}.")
                        category_counts[rating] += remaining
                        total_rated += remaining

                    break
                except ValueError:
                    print("Please enter a valid number.")

        rating_counts[category] = category_counts

    # Generate data based on specific counts
    data = []
    data.append(rating_categories + ["department_name"])  # Header

    # Create a pool of ratings for each category based on counts
    rating_pools = {}
    for category, counts in rating_counts.items():
        pool = []
        for rating, count in counts.items():
            pool.extend([rating] * count)
        random.shuffle(pool)  # Randomize the order
        rating_pools[category] = pool

    # Assign departments based on their size
    department_pool = []
    for dept, count in departments.items():
        # Scale department counts to match number of respondents
        scaled_count = int((count / sum(departments.values())) * num_respondents)
        department_pool.extend([dept] * scaled_count)

    # Adjust if necessary to match exact number of respondents
    while len(department_pool) < num_respondents:
        department_pool.append(random.choice(department_names))
    while len(department_pool) > num_respondents:
        department_pool.pop()

    random.shuffle(department_pool)  # Randomize the order

    # Generate rows
    for i in range(num_respondents):
        row = []
        for category in rating_categories:
            row.append(str(rating_pools[category][i]))
        row.append(department_pool[i])
        data.append(row)

    return data

def save_to_csv(data, filename="survey_data.csv"):
    try:
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            for row in data:
                writer.writerow(row)
        print(f"Data saved to {os.path.abspath(filename)}")
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False

def main():
    print("Survey Data Generator")
    print("=====================")

    data = generate_survey_data()

    # Ask for filename
    filename = input("Enter filename for CSV (default: survey_data.csv): ")
    if not filename:
        filename = "survey_data.csv"
    if not filename.endswith('.csv'):
        filename += '.csv'

    save_to_csv(data, filename)

    # Display sample of generated data
    print("\nSample of generated data:")
    for i, row in enumerate(data):
        if i > 5:  # Show header + 5 rows
            break
        print(', '.join(row))

    print(f"Total: {len(data)-1} respondents")

if __name__ == "__main__":
    main()
