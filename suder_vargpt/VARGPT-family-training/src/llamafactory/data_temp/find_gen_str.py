def find_placeholder_positions(text):
    normal_placeholder = "IMAGE_PLACEHOLDER"
    gen_placeholder = "IMAGE_GEN_PLACEHOLDER"
    
    all_positions = []
    gen_positions = []
    
    pos = 0
    while True:
        pos = text.find(normal_placeholder, pos)
        if pos == -1:
            break
        all_positions.append(pos)
        pos += 1
    
    pos = 0
    while True:
        pos = text.find(gen_placeholder, pos)
        if pos == -1:
            break
        all_positions.append(pos)
        gen_positions.append(pos)
        pos += 1
    
    all_positions.sort()
    
    result = []
    for gen_pos in gen_positions:
        result.append(all_positions.index(gen_pos))
    
    return result

