import pandas as pd
import random
import numpy as np

# Generate a dataset based on user-defined respondent counts for ratings 1-5
def generate_ratings_data(rating_counts, num_respondents):
    """
    Generate a dataset for respondents' ratings.

    Args:
        rating_counts (dict): A dictionary where keys are ratings (0-3) and values are the counts.
        num_respondents (int): Total number of respondents.

    Returns:
        list: A list of ratings matching the distribution.
    """
    ratings = []
    for rating in range(4):  # 0,1,2,3
        if rating in rating_counts:
            ratings.extend([rating] * rating_counts[rating])

    # Shuffle the ratings to randomize the order
    random.shuffle(ratings)

    # Ensure the list matches the number of respondents
    if len(ratings) != num_respondents:
        raise ValueError("The total count of ratings does not match the number of respondents.")

    return ratings

# Function to randomly distribute respondents across ratings
def generate_random_rating_counts(num_respondents, is_low_feature=False):
    """
    Randomly distribute respondents across ratings 0-3.

    Args:
        num_respondents (int): Total number of respondents.
        is_low_feature (bool): If True, bias towards lower ratings.

    Returns:
        dict: A dictionary with ratings as keys and counts as values.
    """
    # Initialize counts for each rating
    rating_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    # Randomly assign each respondent a rating
    for _ in range(num_respondents):
        if is_low_feature:
            # Bias towards lower ratings (0-1)
            rating = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1])[0]
        else:
            # Normal distribution (bias towards higher ratings)
            rating = random.choices([0, 1, 2, 3], weights=[0.1, 0.2, 0.3, 0.4])[0]
        rating_counts[rating] += 1

    return rating_counts

# New function to generate correlated ratings across categories
def generate_associated_ratings(num_respondents, columns, association_type='positive'):
    """
    Generate ratings with associations between different categories.

    Args:
        num_respondents (int): Total number of respondents.
        columns (list): List of column names to generate data for.
        association_type (str): Type of association - 'positive', 'negative', 'random_groups', or 'mixed'.

    Returns:
        dict: A dictionary with column names as keys and rating lists as values.
    """
    associated_data = {}

    if association_type == 'positive':
        # Generate base ratings that will influence all categories
        # People who rate one category highly tend to rate others highly too
        base_ratings = []
        for _ in range(num_respondents):
            # Generate a "satisfaction level" for each respondent
            satisfaction = random.random()  # 0 to 1

            # Convert to rating with some noise
            if satisfaction < 0.15:
                base = 0
            elif satisfaction < 0.35:
                base = 1
            elif satisfaction < 0.65:
                base = 2
            else:
                base = 3

            base_ratings.append(base)

        # Generate ratings for each column based on the base rating
        for column in columns:
            ratings = []
            for base in base_ratings:
                # Add some noise to the base rating
                noise = random.choices([-1, 0, 1], weights=[0.15, 0.7, 0.15])[0]
                rating = max(0, min(3, base + noise))  # Keep within 0-3 range
                ratings.append(rating)
            associated_data[column] = ratings

    elif association_type == 'negative':
        # Some categories are inversely related
        # Split columns into two groups
        mid_point = len(columns) // 2
        group1 = columns[:mid_point]
        group2 = columns[mid_point:]

        # Generate base ratings for group 1
        base_ratings = []
        for _ in range(num_respondents):
            satisfaction = random.random()
            if satisfaction < 0.15:
                base = 0
            elif satisfaction < 0.35:
                base = 1
            elif satisfaction < 0.65:
                base = 2
            else:
                base = 3
            base_ratings.append(base)

        # Generate ratings for each column
        for column in columns:
            ratings = []
            for base in base_ratings:
                if column in group1:
                    # Group 1 follows base rating with noise
                    noise = random.choices([-1, 0, 1], weights=[0.15, 0.7, 0.15])[0]
                    rating = max(0, min(3, base + noise))
                else:
                    # Group 2 is inversely related to base rating
                    inverse_base = 3 - base
                    noise = random.choices([-1, 0, 1], weights=[0.15, 0.7, 0.15])[0]
                    rating = max(0, min(3, inverse_base + noise))
                ratings.append(rating)
            associated_data[column] = ratings

    elif association_type == 'random_groups':
        # Create random respondent groups with different rating patterns
        # Define 3-4 different respondent personas
        personas = [
            {'name': 'Highly Satisfied', 'weights': [0.05, 0.1, 0.25, 0.6]},
            {'name': 'Neutral', 'weights': [0.1, 0.3, 0.4, 0.2]},
            {'name': 'Dissatisfied', 'weights': [0.4, 0.3, 0.2, 0.1]},
            {'name': 'Mixed', 'weights': [0.25, 0.25, 0.25, 0.25]}
        ]

        # Assign each respondent to a persona
        respondent_personas = random.choices(range(len(personas)), k=num_respondents)

        # Generate ratings for each column
        for column in columns:
            ratings = []
            for persona_idx in respondent_personas:
                persona = personas[persona_idx]
                rating = random.choices([0, 1, 2, 3], weights=persona['weights'])[0]
                ratings.append(rating)
            associated_data[column] = ratings

    elif association_type == 'mixed':
        # Some columns are strongly correlated, others independent
        # Randomly select columns to be correlated
        num_correlated = max(2, len(columns) // 2)
        correlated_columns = random.sample(columns, num_correlated)
        independent_columns = [col for col in columns if col not in correlated_columns]

        # Generate base ratings for correlated columns
        base_ratings = []
        for _ in range(num_respondents):
            satisfaction = random.random()
            if satisfaction < 0.15:
                base = 0
            elif satisfaction < 0.35:
                base = 1
            elif satisfaction < 0.65:
                base = 2
            else:
                base = 3
            base_ratings.append(base)

        # Generate ratings for each column
        for column in columns:
            if column in correlated_columns:
                # Correlated columns follow base rating with noise
                ratings = []
                for base in base_ratings:
                    noise = random.choices([-1, 0, 1], weights=[0.15, 0.7, 0.15])[0]
                    rating = max(0, min(3, base + noise))
                    ratings.append(rating)
            else:
                # Independent columns follow their own distribution
                ratings = []
                for _ in range(num_respondents):
                    rating = random.choices([0, 1, 2, 3], weights=[0.1, 0.2, 0.3, 0.4])[0]
                    ratings.append(rating)

            associated_data[column] = ratings

    return associated_data

# Map the long column names to the expected short names
columns_to_generate = {
    "Objectives": "Objectives_Met",
    "Venue": "Venue_Rating",
    "Schedule": "Schedule_Rating",
    "Invitational Allowance": "Allowance_Rating",
    "Resource Speakers": "Speaker_Rating",
    "Facilitators": "Facilitator_Rating",
    "Participants": "Participant_Rating"
}  # 7 columns based on the image

# List of departments to randomly assign
departments = ["SNAHS", "SITE", "SBAHM", "BEU", "SOM"]

num_respondents = int(input("Enter the total number of respondents: "))
use_random_distribution = input("Do you want to randomly distribute ratings? (yes/no): ").lower().strip() == 'yes'

# Add option for associated ratings
create_associations = False
association_type = 'positive'
if use_random_distribution:
    create_associations = input("Do you want to create associations between ratings? (yes/no): ").lower().strip() == 'yes'
    if create_associations:
        print("Association types:")
        print("1. Positive - Respondents who rate one category highly tend to rate others highly too")
        print("2. Negative - Some categories are inversely related to others")
        print("3. Random Groups - Respondents fall into groups with different rating patterns")
        print("4. Mixed - Some categories are correlated, others independent")

        association_choice = input("Choose association type (1-4): ").strip()
        association_map = {
            '1': 'positive',
            '2': 'negative',
            '3': 'random_groups',
            '4': 'mixed'
        }
        association_type = association_map.get(association_choice, 'positive')

low_feature_count = 0
if use_random_distribution and not create_associations:
    low_feature_question = input("Do you want some features to have low ratings? (yes/no): ").lower().strip()
    if low_feature_question == 'yes':
        max_features = len(columns_to_generate)
        while True:
            try:
                low_feature_count = int(input(f"How many features should have low ratings? (0-{max_features}): "))
                if 0 <= low_feature_count <= max_features:
                    break
                else:
                    print(f"Please enter a number between 0 and {max_features}.")
            except ValueError:
                print("Please enter a valid number.")

# Randomly select which features will have low ratings
low_feature_columns = []
if low_feature_count > 0:
    low_feature_columns = random.sample(list(columns_to_generate.values()), low_feature_count)
    print(f"The following features will have low ratings: {', '.join(low_feature_columns)}")

random_data = {}

# Generate associated ratings if requested
if create_associations:
    print(f"Generating associated ratings with '{association_type}' pattern...")
    column_names = list(columns_to_generate.values())
    random_data = generate_associated_ratings(num_respondents, column_names, association_type)

    # Print summary of generated data
    for column in column_names:
        ratings = random_data[column]
        rating_counts = {i: ratings.count(i) for i in range(4)}
        print(f"Distribution for '{column}':")
        for rating, count in rating_counts.items():
            print(f"  Rating {rating}: {count} respondents")
else:
    # Original code for independent ratings
    for long_name, short_name in columns_to_generate.items():
        if use_random_distribution:
            # Check if this feature should have low ratings
            is_low_feature = short_name in low_feature_columns

            # Randomly distribute respondents across ratings
            rating_counts = generate_random_rating_counts(num_respondents, is_low_feature)

            print(f"Random distribution for '{long_name}'{'(LOW RATINGS)' if is_low_feature else ''}:")
            for rating, count in rating_counts.items():
                print(f"  Rating {rating}: {count} respondents")
        else:
            print(f"Enter the number of respondents who rated 0-3 for: '{long_name}'")
            rating_counts = {}
            for rating in range(4):  # 0,1,2,3
                rating_counts[rating] = int(input(f"  Number of respondents who rated {rating}: "))

            if sum(rating_counts.values()) != num_respondents:
                raise ValueError(f"The total ratings for '{long_name}' do not match the number of respondents.")

        random_data[short_name] = generate_ratings_data(rating_counts, num_respondents)

# Generate random department assignments
random_data["department_name"] = random.choices(departments, k=num_respondents)

# Add this before creating the DataFrame
print("Column names:", list(random_data.keys()))
print("Number of columns:", len(random_data))

# Add these debug prints right before creating the DataFrame
print("\nDebug Information:")
print("1. Expected columns:", list(columns_to_generate.values()) + ["department_name"])
print("2. Data keys:", list(random_data.keys()))
for col, data in random_data.items():
    print(f"3. Length of {col} data:", len(data))
print("4. Number of respondents:", num_respondents)

# Create a new DataFrame with the generated data
generated_df = pd.DataFrame(random_data)

# Save the generated dataset to a CSV file
output_path = 'generated_ratings_data.csv'
generated_df.to_csv(output_path, index=False)

print(f"Random data generated and saved to {output_path}")
