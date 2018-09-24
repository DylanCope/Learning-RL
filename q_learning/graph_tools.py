def dfs(graph, start, end):
    fringe = [(start, [])]
    while fringe:
        state, path = fringe.pop()
        if path and state == end:
            yield path
            continue
        for next_state in graph[state].keys():
            if next_state in path:
                continue
            fringe.append((next_state, path+[next_state]))
            
def get_cycles(graph):
    return [[node] + path  
            for node in graph 
            for path in dfs(graph, node, node)]
    
def get_unique_cycles(graph):
    all_cycles = get_cycles(graph)
    
    def cycles_match(c1, c2):
        return len(c1) == len(c2) and all(n in c2 for n in c1)
    
    unique_cycles = [all_cycles[0]]
    for c1 in all_cycles[1:]:
        if not any(cycles_match(c1, c2) 
                   for c2 in unique_cycles):
            unique_cycles.append(c1)
    return unique_cycles
         
def reward_per_turn(graph, cycle):
    a = cycle[0]
    b = cycle[1]
    total_reward = graph[a][b]
    len_cycle = len(cycle)
    if len_cycle > 2:
        a = b
        for b in cycle[2:]:
            total_reward += graph[a][b]
            a = b
    return total_reward / (len_cycle - 1)