import pandas as pd
import random

# Generate a dataset based on user-defined re spondent counts for ratings 1-5
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

# Map the long column names to the expected short names
columns_to_generate = {
    "Overall Event Evaluation": "Overall_Rating",
    "Objectives": "Objectives_Met",
    "Venue": "Venue_Rating",
    "Schedule": "Schedule_Rating",
    "Invitational Allowance": "Allowance_Rating",
    "Resource Speakers": "Speaker_Rating",
    "Facilitators": "Facilitator_Rating",
    "Participants": "Participant_Rating"
}  # 7 columns based on the image

num_respondents = int(input("Enter the total number of respondents: "))

random_data = {}
for long_name, short_name in columns_to_generate.items():
    print(f"Enter the number of respondents who rated 0-3 for: '{long_name}'")
    rating_counts = {}
    for rating in range(4):  # 0,1,2,3
        rating_counts[rating] = int(input(f"  Number of respondents who rated {rating}: "))

    if sum(rating_counts.values()) != num_respondents:
        raise ValueError(f"The total ratings for '{long_name}' do not match the number of respondents.")

    random_data[short_name] = generate_ratings_data(rating_counts, num_respondents)

# Add this before creating the DataFrame
print("Column names:", list(random_data.keys()))
print("Number of columns:", len(random_data))

# Add these debug prints right before creating the DataFrame
print("\nDebug Information:")
print("1. Expected columns:", list(columns_to_generate.values()))
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
