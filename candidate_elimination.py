import pandas as pd

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


def more_general(h1, h2):
    """ Check if hypothesis h1 is more general than h2 """
    more_general_parts = []
    for x, y in zip(h1, h2):
        mg = x == '?' or (x != '0' and (x == y or y == '0'))
        more_general_parts.append(mg)
    return all(more_general_parts)


def covers(example, hypothesis):
    """Check if hypothesis covers the example"""
    return more_general(hypothesis, example)


def consistent(h1, x1, y_label):
    """ Check if hypothesis h1 is consistent with example x1 """
    y_label = True if y_label == 'Yes' else False
    return True if (covers(x1, h1) == y_label) else False


def min_generalizations(s, d, y_label, gb):
    """Generate all minimal generalizations of h of s such that h is consistent with d"""
    min_genls = []
    for i in range(len(s)):
        if s[i] != d[i]:
            h = s
            if s[i] != '?':
                h[i] = '?'
            if consistent(h, d, y_label):
                # Check if some member of G is more general than h
                if any(more_general(g, h) for g in gb):
                    min_genls.append(h)
    return min_genls


def min_specializations(g, d, y_label, sb):
    """Generate all minimal specializations of h of g such that h is consistent with d"""
    min_spls = []
    for i in range(len(g)):
        if g[i] == d[i]:
            h = list(g)
            if consistent(h, d, y_label):
                if any(more_general(h, s) for s in sb):
                    min_spls.append(h)
        elif g[i] == '?':
            for val in attribute_values[attributes_map[i]]:
                if g[i] != val:
                    h = list(g)
                    h[i] = val
                    if consistent(h, d, y_label):
                        # Check if h is more general than some hypothesis in S
                        if any(more_general(h, s) for s in sb):
                            min_spls.append(tuple(h))
    return min_spls


def candidate_elimination_algorithm(data):
    # Get training examples
    instances = [set(data[col]) for col in data.columns[:-1]]
    # Initialize specific boundary and general boundary
    sb = [list('0' for _ in range(len(instances)))]
    gb = [list('?' for _ in range(len(instances)))]
    print("Step: 0", )
    print_sg_gb(sb, gb)

    for index, row in data.iterrows():
        d, y = row.iloc[:-1].tolist(), row.iloc[-1]
        if y == 'Yes':  # Positive instance
            # Remove from G any hypothesis inconsistent with d.
            gb = [g for g in gb if consistent(g, d, y)]
            sb_new = []
            for s in sb:
                if not consistent(s, d, y):
                    all_zeros = all(element == '0' for element in s)
                    if all_zeros:
                        sb_new.append(d)
                    else:
                        # Remove s from S
                        sb_new = [sh for sh in sb if sh != s]
                        # Add to S all minimal generalizations h of s such that h consistent with d
                        sb_new.extend(min_generalizations(s, d, y, gb))
                else:
                    sb_new.append(s)
            sb = sb_new
            # Remove from S any hypothesis that is more general than another hypothesis
            sb = [s for s in sb if not any(more_general(s, s1) for s1 in sb if s != s1)]
        else:  # Negative instance
            # Remove from S any hypothesis inconsistent with d.
            sb = [s for s in sb if consistent(s, d, y)]
            gb_new = []
            for g in gb:
                if not consistent(g, d, y):
                    # Remove g from G
                    gb_new = [gh for gh in gb if gh != g]
                    # Add to G all minimal specializations h of g such that h consistent with d
                    gb_new.extend(min_specializations(g, d, y, sb))
                else:
                    gb_new.append(g)
            gb = gb_new
            # Remove from G any hypothesis that is less general than another hypothesis
            gb = [g for g in gb if not any(more_general(g1, g) for g1 in gb if g != g1)]
        print("Step:", index+1)
        print_sg_gb(sb, gb)
    return sb, gb


def print_sg_gb(s, g):
    print("S:", s)
    print("G:", g)


if __name__ == "__main__":
    # Training Examples with labels
    training_examples = pd.DataFrame([
        ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
    ], columns=['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'EnjoySport'])
    target_concept = 'EnjoySport'

    # Run Candidate Elimination Algorithm
    specific_boundary, general_boundary = candidate_elimination_algorithm(training_examples)

    # Print Version Space
    print("Version Space:")
    print_sg_gb(specific_boundary, general_boundary)
