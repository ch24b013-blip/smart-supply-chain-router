import heapq

def get_network_graph():
    # Quick-Commerce (Blinkit-style) Network
    # Warehouse -> City Hubs -> Dark Stores / Retail
    graph = {
        'Central Warehouse': [('City Hub North', 1.5), ('Express Dark Store A', 0.5), ('City Hub South', 1.2)],
        'City Hub North': [('Supermarket 1', 1.0), ('Supermarket 2', 1.2)],
        'City Hub South': [('Supermarket 3', 0.8), ('Express Dark Store B', 0.3)],
        'Express Dark Store A': [('Local Mart X', 0.2), ('Local Mart Y', 0.4)],
        'Express Dark Store B': [('Local Mart Z', 0.2)],
        'Supermarket 1': [], 'Supermarket 2': [], 'Supermarket 3': [],
        'Local Mart X': [], 'Local Mart Y': [], 'Local Mart Z': []
    }
    return graph

def compute_shortest_paths(graph, start_node):
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    priority_queue = [(0, start_node)]
    paths = {node: [] for node in graph}
    paths[start_node] = [start_node]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
            
        for neighbor, weight in graph.get(current_node, []):
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                paths[neighbor] = paths[current_node] + [neighbor]
                heapq.heappush(priority_queue, (distance, neighbor))
                
    return distances, paths