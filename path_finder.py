import networkx as nx
from collections import defaultdict
import heapq
from typing import List, Tuple, Dict, Optional

class PathFinder:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.node_types = nx.get_node_attributes(graph, 'type')
        self.node_positions = nx.get_node_attributes(graph, 'pos')
        
    def get_nodes_by_type(self, node_type: str) -> List[int]:
        """Get all nodes of a specific type"""
        return [node for node, ntype in self.node_types.items() if ntype == node_type]
    
    def dijkstra_shortest_path(self, source: int, target: int) -> Tuple[List[int], float]:
        """
        Find shortest path using Dijkstra's algorithm
        Returns: (path_nodes, total_distance)
        """
        try:
            path = nx.shortest_path(self.graph, source, target, weight='weight')
            distance = nx.shortest_path_length(self.graph, source, target, weight='weight')
            return path, distance
        except nx.NetworkXNoPath:
            return [], float('inf')
    
    def find_all_simple_paths(self, source: int, target: int, max_length: int = 20) -> List[Tuple[List[int], float]]:
        """
        Find all simple paths between source and target
        Returns: List of (path_nodes, total_distance) tuples
        """
        all_paths = []
        
        try:
            # Find all simple paths (no repeated nodes)
            paths = nx.all_simple_paths(self.graph, source, target, cutoff=max_length)
            
            for path in paths:
                # Calculate total distance for this path
                total_distance = 0.0
                for i in range(len(path) - 1):
                    edge_data = self.graph[path[i]][path[i+1]]
                    total_distance += edge_data.get('weight', 0.0)
                
                all_paths.append((path, total_distance))
                
        except nx.NetworkXNoPath:
            pass
        
        return all_paths
    
    def longest_path_between_nodes(self, source: int, target: int) -> Tuple[List[int], float]:
        """
        Find longest simple path between two nodes
        Returns: (path_nodes, total_distance)
        """
        all_paths = self.find_all_simple_paths(source, target)
        
        if not all_paths:
            return [], 0.0
        
        # Find the path with maximum distance
        longest_path, max_distance = max(all_paths, key=lambda x: x[1])
        return longest_path, max_distance
    
    def solve_yellow_shortest_path(self) -> Dict:
        """Find shortest path between yellow points"""
        try:
            yellow_nodes = self.get_nodes_by_type('yellow')
            
            if len(yellow_nodes) < 2:
                return {
                    'success': False,
                    'error': f'Found {len(yellow_nodes)} yellow points, need exactly 2',
                    'path': [],
                    'distance': 0.0
                }
            
            if len(yellow_nodes) > 2:
                print(f"Warning: Found {len(yellow_nodes)} yellow points, using first 2")
            
            source, target = yellow_nodes[0], yellow_nodes[1]
            print(f"Yellow path: Attempting {source} → {target}")
            
            path, distance = self.dijkstra_shortest_path(source, target)
            
            if not path:
                return {
                    'success': False,
                    'error': f'No path found between yellow nodes {source} and {target}',
                    'path': [],
                    'distance': 0.0
                }
            
            return {
                'success': bool(path),
                'algorithm': 'Dijkstra\'s Algorithm',
                'source_node': source,
                'target_node': target,
                'source_pos': self.node_positions[source],
                'target_pos': self.node_positions[target],
                'path': path,
                'distance': distance,
                'description': 'Shortest path between two yellow points using Dijkstra\'s algorithm'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error solving yellow shortest path: {str(e)}',
                'path': [],
                'distance': 0.0
            }
    
    def solve_orange_shortest_path(self) -> Dict:
        """Find shortest path between orange points"""
        try:
            orange_nodes = self.get_nodes_by_type('orange')
            
            if len(orange_nodes) < 2:
                return {
                    'success': False,
                    'error': f'Found {len(orange_nodes)} orange points, need exactly 2',
                    'path': [],
                    'distance': 0.0
                }
            
            if len(orange_nodes) > 2:
                print(f"Warning: Found {len(orange_nodes)} orange points, using first 2")
            
            source, target = orange_nodes[0], orange_nodes[1]
            print(f"Orange path: Attempting {source} → {target}")
            
            path, distance = self.dijkstra_shortest_path(source, target)
            
            if not path:
                return {
                    'success': False,
                    'error': f'No path found between orange nodes {source} and {target}',
                    'path': [],
                    'distance': 0.0
                }
            
            return {
                'success': bool(path),
                'algorithm': 'Dijkstra\'s Algorithm',
                'source_node': source,
                'target_node': target,
                'source_pos': self.node_positions[source],
                'target_pos': self.node_positions[target],
                'path': path,
                'distance': distance,
                'description': 'Shortest path between two orange points using Dijkstra\'s algorithm'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error solving orange shortest path: {str(e)}',
                'path': [],
                'distance': 0.0
            }
    
    def solve_yellow_to_orange_longest_path(self) -> Dict:
        """Find longest path from leftmost yellow to rightmost orange"""
        try:
            yellow_nodes = self.get_nodes_by_type('yellow')
            orange_nodes = self.get_nodes_by_type('orange')
            
            if len(yellow_nodes) == 0 or len(orange_nodes) == 0:
                return {
                    'success': False,
                    'error': f'Found {len(yellow_nodes)} yellow and {len(orange_nodes)} orange points',
                    'path': [],
                    'distance': 0.0
                }
            
            # Find leftmost yellow (minimum x coordinate)
            leftmost_yellow = min(yellow_nodes, key=lambda n: self.node_positions[n][0])
            
            # Find rightmost orange (maximum x coordinate)  
            rightmost_orange = max(orange_nodes, key=lambda n: self.node_positions[n][0])
            
            print(f"Longest path: Attempting {leftmost_yellow} → {rightmost_orange}")
            
            path, distance = self.longest_path_between_nodes(leftmost_yellow, rightmost_orange)
            
            if not path:
                return {
                    'success': False,
                    'error': f'No path found between yellow node {leftmost_yellow} and orange node {rightmost_orange}',
                    'path': [],
                    'distance': 0.0
                }
            
            return {
                'success': bool(path),
                'algorithm': 'All Simple Paths Enumeration + Maximum Selection',
                'source_node': leftmost_yellow,
                'target_node': rightmost_orange,
                'source_pos': self.node_positions[leftmost_yellow],
                'target_pos': self.node_positions[rightmost_orange],
                'path': path,
                'distance': distance,
                'description': 'Longest simple path from leftmost yellow to rightmost orange point'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error solving longest path: {str(e)}',
                'path': [],
                'distance': 0.0
            }
    
    def get_path_coordinates(self, path_nodes: List[int]) -> List[Tuple[int, int]]:
        """Get the actual coordinate path from graph edges"""
        if len(path_nodes) < 2:
            return []
        
        full_coordinate_path = []
        
        for i in range(len(path_nodes) - 1):
            node1, node2 = path_nodes[i], path_nodes[i + 1]
            
            # Get the stored path coordinates from the edge
            if self.graph.has_edge(node1, node2):
                edge_data = self.graph[node1][node2]
                edge_path = edge_data.get('path', [])
                
                if edge_path:
                    if i == 0:  # First segment, include start point
                        full_coordinate_path.extend(edge_path)
                    else:  # Subsequent segments, skip start point to avoid duplication
                        full_coordinate_path.extend(edge_path[1:])
                else:
                    # Fallback to node positions if no edge path
                    if i == 0:
                        full_coordinate_path.append(self.node_positions[node1])
                    full_coordinate_path.append(self.node_positions[node2])
        
        return full_coordinate_path
    
    def solve_all_problems(self) -> Dict:
        """Solve all three pathfinding problems"""
        print("🔍 Starting pathfinding analysis...")
        
        # Debug: Print graph structure
        self._debug_graph_structure()
        
        results = {
            'yellow_shortest': self.solve_yellow_shortest_path(),
            'orange_shortest': self.solve_orange_shortest_path(),
            'yellow_orange_longest': self.solve_yellow_to_orange_longest_path()
        }
        
        # Add coordinate paths for successful results
        for key, result in results.items():
            if result['success']:
                result['coordinate_path'] = self.get_path_coordinates(result['path'])
        
        self._print_results_summary(results)
        return results
    
    def _debug_graph_structure(self):
        """Print debug information about the graph structure"""
        print(f"\n🔍 Graph Debug Information:")
        print(f"   Total nodes: {self.graph.number_of_nodes()}")
        print(f"   Total edges: {self.graph.number_of_edges()}")
        
        # Count nodes by type
        node_counts = {}
        for node, node_type in self.node_types.items():
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
        
        print(f"   Node types: {dict(node_counts)}")
        
        # Check connectivity
        if self.graph.number_of_edges() == 0:
            print("   ⚠️ WARNING: Graph has no edges!")
        
        # Print colored nodes specifically
        yellow_nodes = self.get_nodes_by_type('yellow')
        orange_nodes = self.get_nodes_by_type('orange')
        print(f"   Yellow nodes: {yellow_nodes}")
        print(f"   Orange nodes: {orange_nodes}")
        
        # Check if colored nodes have any connections
        for node in yellow_nodes + orange_nodes:
            neighbors = list(self.graph.neighbors(node))
            print(f"   Node {node} ({self.node_types[node]}) has {len(neighbors)} neighbors: {neighbors}")
        
        print()
    
    def _print_results_summary(self, results: Dict):
        """Print a summary of all pathfinding results"""
        print("\n" + "="*60)
        print("🎯 PATHFINDING RESULTS SUMMARY")
        print("="*60)
        
        for i, (key, result) in enumerate(results.items(), 1):
            problem_name = {
                'yellow_shortest': 'Shortest path between Yellow points',
                'orange_shortest': 'Shortest path between Orange points', 
                'yellow_orange_longest': 'Longest path from Yellow (left) to Orange (right)'
            }[key]
            
            print(f"\n{i}. {problem_name}")
            print("-" * 50)
            
            if result['success']:
                print(f"✅ Algorithm: {result['algorithm']}")
                print(f"📏 Distance: {result['distance']:.2f} pixels")
                print(f"🛤️  Path length: {len(result['path'])} nodes")
                print(f"📍 Coordinate points: {len(result.get('coordinate_path', []))} points")
                print(f"🔗 Node path: {' → '.join(map(str, result['path']))}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        print("\n" + "="*60) 