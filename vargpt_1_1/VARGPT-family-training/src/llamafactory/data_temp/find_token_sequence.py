
def find_token_sequence(batch_ids):
    result = []
    target_tokens = [151652, 151657]
    
    for sequence in batch_ids:
        found_positions = []
        for i, token in enumerate(sequence):
            if token in target_tokens:
                found_positions.append((i, 0 if token == 151652 else 1))
        
        result.extend([x[1] for x in sorted(found_positions, key=lambda x: x[0])])
    
    return result


batch_ids = [
    [21, 213, 32, 32, 32, 324, 1, 24, 151652, 323, 32, 151652],
    [151657, 32, 32, 3, 151652]
]
result = find_token_sequence(batch_ids)
print(result)  