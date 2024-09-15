import pandas as pd


def find_s_algorithm(data, target):
    # Initialize the most specific hypothesis
    h = None
    # Iterate over the dataset
    for index, row in data.iterrows():
        if row[target] == 'Yes':  # Consider only positive instances
            if h is None:
                h = row.iloc[:-1].tolist()  # Initialize h with the first positive instance
            else:
                for i in range(len(h)):
                    if h[i] != row.iloc[i]:
                        h[i] = '?'  # Generalize h if attribute values differ
        print("Step:", index + 1, h)
    return h


if __name__ == "__main__":
    # Training Examples with labels
    training_examples = pd.DataFrame([
        ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
    ], columns=['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'EnjoySport'])
    target_concept = 'EnjoySport'
    # Apply Find-S Algorithm
    hypothesis = find_s_algorithm(training_examples, target_concept)
    # Print the final hypothesis
    print("Final Hypothesis:", hypothesis)
