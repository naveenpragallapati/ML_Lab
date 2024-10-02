# Define possible values for each attribute
attribute_values = {
    'Sky': ['Sunny', 'Cloudy', 'Rainy'],
    'AirTemp': ['Warm', 'Cold'],
    'Humidity': ['Normal', 'High'],
    'Wind': ['Strong', 'Weak'],
    'Water': ['Warm', 'Cool'],
    'Forecast': ['Same', 'Change']
}

attributes_map = {
    0: 'Sky',
    1: 'AirTemp',
    2: 'Humidity',
    3: 'Wind',
    4: 'Water',
    5: 'Forecast'
}

def candidate_elimination(examples):
    """
    Implements the candidate elimination algorithm.

    Args:
        examples: A list of tuples (instance, label), where instance is a list of attribute values and label is the corresponding class.

    Returns:
        A list of hypotheses that are consistent with all positive examples and inconsistent with all negative examples.
    """

    # Initialize the general and specific hypotheses
    general = [["?"] * len(examples[0][0])]
    specific = [["0"] * len(examples[0][0])]

    # Iterate over each example
    example_no = 0
    print(f"Initialization")
    print(f"S = {specific}")
    print(f"G = {general}")
    print()
    for instance, label in examples:
        example_no += 1
        if label == "Yes":
            # If the instance is positive, remove any general hypothesis that is inconsistent with it
            general = [h for h in general if is_consistent(h, instance, label)]

            # If the specific hypothesis is inconsistent with the instance, make it more general
            if not is_consistent(specific[0], instance, label):
                for i in range(len(specific[0])):
                    if specific[0][i] == "0":
                        specific[0][i] = instance[i]
                    elif specific[0][i] != instance[i]:
                        specific[0][i] = "?"
        else:
            # If the instance is negative, remove any specific hypothesis that is consistent with it
            specific = [h for h in specific if not is_consistent(h, instance, label)]

            # If the general hypothesis is consistent with the instance, make it more specific
            if general:
                bool_value = is_consistent(general[0], instance, label)
                if bool_value:
                    new_general = []
                    for g in general:
                        new_general += specialize_g(g, instance, attributes_map, attribute_values)
                    general = new_general

        # Check consistency of S and G with all previous examples
        for prev_instance, prev_label in examples[:example_no]:
            # specific = [h for h in specific if is_consistent(h, prev_instance, prev_label)]
            if prev_label == "Yes":
                general = [h for h in general if is_consistent(h, prev_instance, prev_label)]
            else:
                general = [h for h in general if not is_consistent(h, prev_instance, prev_label)]

        # Ensure S and G are not empty
        if not specific:
            specific = [["0"] * len(examples[0][0])]
        if not general:
            general = [["?"] * len(examples[0][0])]

        print(f"Instance {example_no}: {instance}, Label: {label}")
        print(f"S = {specific}")
        print(f"G = {general}")
        print()

    return general, specific

def is_consistent(hypothesis, instance, label):
    """
    Checks if a hypothesis is consistent with an instance and label.

    Args:
        hypothesis: A list of attribute values representing a hypothesis.
        instance: A list of attribute values representing an instance.
        label: The class label of the instance.

    Returns:
        True if the hypothesis is consistent with the instance and label, False otherwise.
    """

    # Check if the hypothesis is more general than the instance
    for i in range(len(hypothesis)):
        if hypothesis[i] != "?" and hypothesis[i] != instance[i]:
            return False

    # Check if the hypothesis implies the label
    if label == "Yes":
        return True
    else:
        return hypothesis != instance

def specialize_g(hypothesis, instance, attrib_map, attrib_values):
    """
    Specializes a general hypothesis to exclude a negative instance.

    Args:
        hypothesis: A list of attribute values representing a hypothesis.
        instance: A list of attribute values representing an instance.
        attrib_map: A dictionary mapping attribute indices to attribute names.
        attrib_values: A dictionary mapping attribute names to possible values.

    Returns:
        A list of specialized hypotheses.
    """
    specializations = []
    for i in range(len(hypothesis)):
        if hypothesis[i] == "?":
            for value in attrib_values[attrib_map[i]]:
                if value != instance[i]:
                    new_hypothesis = hypothesis[:]
                    new_hypothesis[i] = value
                    specializations.append(new_hypothesis)
    return specializations

if __name__ == "__main__":
    # Define the training examples
    training_examples = [
        (['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'], 'Yes'),
        (['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'], 'Yes'),
        (['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'], 'No'),
        (['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'], 'Yes')
    ]

    # Run the candidate elimination algorithm
    general_boundary, specific_boundary = candidate_elimination(training_examples)

    # Print the final version space
    print("Final Version Space:")
    print(f"S = {specific_boundary}")
    print(f"G = {general_boundary}")