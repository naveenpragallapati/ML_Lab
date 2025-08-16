# Candidate Elimination Algorithm for Concept Learning
# Define possible values for each attribute
attribute_values = {
    'Sky': ['Sunny', 'Cloudy', 'Rainy'],
    'AirTemp': ['Warm', 'Cold'],
    'Humidity': ['Normal', 'High'],
    'Wind': ['Strong', 'Weak'],
    'Water': ['Warm', 'Cool'],
    'Forecast': ['Same', 'Change']
}
# Create a mapping from attribute index to attribute name
attributes_map = {i: key for i, key in enumerate(attribute_values.keys())}

def consistent(hypothesis, instance, label):
    """
    Checks if a hypothesis is consistent with an instance and label.
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

def candidate_elimination(examples):
    """Returns a list of hypotheses that are consistent with all positive examples and inconsistent with all negative examples."""

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
            general = [h for h in general if (consistent(h, instance, label) is True)]

            # If the specific hypothesis is inconsistent with the instance, make it more general
            if not consistent(specific[0], instance, label):
                for i in range(len(specific[0])):
                    if specific[0][i] == "0":
                        specific[0][i] = instance[i]
                    elif specific[0][i] != instance[i]:
                        specific[0][i] = "?"
        else:
            # If the instance is negative, remove any specific hypothesis that is consistent with it
            specific = [h for h in specific if (consistent(h, instance, label) is False)]

            # If the general hypothesis is consistent with the instance, make it more specific
            if general:
                bool_value = consistent(general[0], instance, label)
                if bool_value:
                    new_general = []
                    for g in general:
                        new_general += specialize_g(g, instance, attributes_map, attribute_values)
                    general = new_general

        # Check consistency of S and G with all previous examples
        for prev_instance, prev_label in examples[:example_no]:
            if prev_label == "Yes":
                general = [h for h in general if consistent(h, prev_instance, prev_label)]
                specific = [h for h in specific if consistent(h, prev_instance, prev_label)]
            else:
                general = [h for h in general if not consistent(h, prev_instance, prev_label)]
                specific = [h for h in specific if not consistent(h, prev_instance, prev_label)]

        print(f"Instance {example_no}: {instance}, Label: {label}")
        print(f"S = {specific}")
        print(f"G = {general}")
        print()
    return general, specific

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