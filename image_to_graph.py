import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from skimage.morphology import skeletonize
from skimage import measure
import math

class ImageToGraph:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.graph = nx.Graph()
        self.colored_points = {'yellow': [], 'orange': []}
        
    def load_and_preprocess_image(self):
        """Load image and preprocess for path extraction"""
        print("Loading and preprocessing image...")
        
        # Load image
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {self.image_path}")
            
        # Convert to RGB for matplotlib
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        print(f"Image loaded: {self.image.shape}")
        return self.image_rgb
    
    def detect_colored_points(self):
        """Detect yellow and orange points in the image"""
        print("Detecting colored points...")
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges in HSV
        # Yellow range
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        
        # Orange range  
        orange_lower = np.array([5, 100, 100])
        orange_upper = np.array([15, 255, 255])
        
        # Create masks
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        
        # Find contours and get centroids
        def get_centroids(mask, color_name):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            centroids = []
            for contour in contours:
                if cv2.contourArea(contour) > 50:  # Filter small noise
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centroids.append((cx, cy))
                        print(f"Found {color_name} point at: ({cx}, {cy})")
            return centroids
        
        self.colored_points['yellow'] = get_centroids(yellow_mask, 'yellow')
        self.colored_points['orange'] = get_centroids(orange_mask, 'orange')
        
        print(f"Detected {len(self.colored_points['yellow'])} yellow points")
        print(f"Detected {len(self.colored_points['orange'])} orange points")
        
        return yellow_mask, orange_mask
    
    def extract_path_structure(self):
        """Extract the black line paths from the image"""
        print("Extracting path structure...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Create binary image - black paths on white background
        # Use adaptive thresholding for better curve detection
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Clean up noise with smaller kernel to preserve curve details
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Remove very small noise
        kernel_small = np.ones((1,1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        
        # Skeletonize to get single-pixel-wide paths
        skeleton = skeletonize(binary // 255).astype(np.uint8) * 255
        
        return binary, skeleton
    
    def find_key_points(self, skeleton):
        """Find intersection points and endpoints in the skeleton"""
        print("Finding key points (intersections and endpoints)...")
        
        # Define neighborhood kernel for intersection detection
        kernel = np.array([[1, 1, 1],
                          [1, 10, 1], 
                          [1, 1, 1]], dtype=np.uint8)
        
        # Convolve to find intersections
        filtered = cv2.filter2D(skeleton, -1, kernel)
        
        # Find intersection points (where many lines meet)
        intersections = []
        endpoints = []
        
        height, width = skeleton.shape
        for y in range(1, height-1):
            for x in range(1, width-1):
                if skeleton[y, x] == 255:  # On a path
                    # Count neighbors
                    neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2] == 255) - 1
                    
                    if neighbors == 1:  # Endpoint
                        endpoints.append((x, y))
                    elif neighbors > 2:  # Intersection
                        intersections.append((x, y))
        
        print(f"Found {len(intersections)} intersections")
        print(f"Found {len(endpoints)} endpoints")
        
        return intersections, endpoints
    
    def build_graph(self, skeleton, intersections, endpoints):
        """Build NetworkX graph from the path structure"""
        print("Building graph from path structure...")
        
        # All key points (intersections + endpoints)
        path_key_points = intersections + endpoints
        
        # Remove duplicates
        path_key_points = list(set(path_key_points))
        
        # Add path nodes to graph first
        for i, point in enumerate(path_key_points):
            node_type = 'intersection' if point in intersections else 'endpoint'
            self.graph.add_node(i, pos=point, type=node_type, coords=point)
        
        # Find connections between path nodes
        self._trace_connections(skeleton, path_key_points)
        
        # Now connect colored points to the nearest path points
        self._connect_colored_points_to_paths(skeleton, path_key_points)
        
        print(f"Graph created with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def _connect_colored_points_to_paths(self, skeleton, path_key_points):
        """Connect colored points to the nearest points on the path network"""
        import math
        
        current_node_id = len(path_key_points)
        
        for color, points in self.colored_points.items():
            for colored_point in points:
                # Add the colored point as a node
                self.graph.add_node(current_node_id, pos=colored_point, type=color, coords=colored_point)
                
                # Find the nearest path point that's actually on the skeleton
                nearest_path_point = self._find_nearest_skeleton_point(colored_point, skeleton)
                
                if nearest_path_point:
                    # Check if this skeleton point corresponds to an existing node
                    nearest_node_id = self._find_or_create_path_node(nearest_path_point, path_key_points)
                    
                    # Calculate distance
                    distance = math.sqrt(
                        (colored_point[0] - nearest_path_point[0])**2 + 
                        (colored_point[1] - nearest_path_point[1])**2
                    )
                    
                    # Create a direct connection path
                    connection_path = [colored_point, nearest_path_point]
                    
                    # Add edge between colored point and path network
                    self.graph.add_edge(current_node_id, nearest_node_id, 
                                      weight=distance, path=connection_path)
                    
                    print(f"Connected {color} point at {colored_point} to path network via node {nearest_node_id} (distance: {distance:.1f})")
                
                current_node_id += 1
    
    def _find_nearest_skeleton_point(self, colored_point, skeleton):
        """Find the nearest point on the skeleton to the colored point"""
        import math
        
        cx, cy = colored_point
        min_distance = float('inf')
        nearest_point = None
        
        # Search in a reasonable radius around the colored point
        search_radius = 50
        height, width = skeleton.shape
        
        for y in range(max(0, cy - search_radius), min(height, cy + search_radius + 1)):
            for x in range(max(0, cx - search_radius), min(width, cx + search_radius + 1)):
                if skeleton[y, x] == 255:  # Point is on skeleton
                    distance = math.sqrt((x - cx)**2 + (y - cy)**2)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_point = (x, y)
        
        return nearest_point
    
    def _find_or_create_path_node(self, skeleton_point, path_key_points):
        """Find existing node for skeleton point or create a new one"""
        import math
        
        # Check if skeleton point is close to an existing key point
        for i, key_point in enumerate(path_key_points):
            distance = math.sqrt(
                (skeleton_point[0] - key_point[0])**2 + 
                (skeleton_point[1] - key_point[1])**2
            )
            if distance < 5:  # Close enough to existing point
                return i
        
        # Create new node for this skeleton point
        new_node_id = self.graph.number_of_nodes()
        self.graph.add_node(new_node_id, pos=skeleton_point, type='connection', coords=skeleton_point)
        return new_node_id
    
    def _trace_connections(self, skeleton, key_points):
        """Trace connections between key points"""
        # Create a mapping from coordinates to node indices
        coord_to_node = {point: i for i, point in enumerate(key_points)}
        
        # For each key point, trace paths to find connections
        for start_idx, start_point in enumerate(key_points):
            # Use a fresh visited set for each starting point
            visited = set()
            self._dfs_trace(skeleton, start_point, start_idx, coord_to_node, visited, [start_point])
            
        # Remove duplicate edges (since we might find the same path from both directions)
        self._remove_duplicate_edges()
    
    def _dfs_trace(self, skeleton, current_pos, start_idx, coord_to_node, visited, path):
        """DFS to trace paths between key points"""
        x, y = current_pos
        
        if (x, y) in visited:
            return
        visited.add((x, y))
        
        # Check if we reached another key point
        if (x, y) in coord_to_node and (x, y) != path[0] and len(path) > 5:
            end_idx = coord_to_node[(x, y)]
            
            # Densify path for better curve representation
            densified_path = self._densify_path(path)
            
            # Calculate ACTUAL ARC LENGTH by summing distances along the curved path
            arc_length = self._calculate_arc_length(densified_path)
            
            self.graph.add_edge(start_idx, end_idx, weight=arc_length, path=densified_path.copy())
            return
        
        # Explore neighbors
        height, width = skeleton.shape
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = x + dx, y + dy
                if (0 <= nx < width and 0 <= ny < height and 
                    skeleton[ny, nx] == 255 and (nx, ny) not in visited):
                    
                    new_path = path + [(nx, ny)]
                    self._dfs_trace(skeleton, (nx, ny), start_idx, coord_to_node, visited, new_path)
    
    def _calculate_arc_length(self, path):
        """Calculate the actual arc length along the curved path"""
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(path)):
            x1, y1 = path[i-1]
            x2, y2 = path[i]
            # Sum up Euclidean distances between consecutive points
            segment_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_length += segment_length
        
        return total_length
    
    def _densify_path(self, path, max_segment_length=3.0):
        """Add intermediate points to ensure accurate curve representation"""
        if len(path) < 2:
            return path
        
        densified_path = [path[0]]
        
        for i in range(1, len(path)):
            x1, y1 = path[i-1]
            x2, y2 = path[i]
            
            # Calculate distance between consecutive points
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # If distance is too large, add intermediate points
            if distance > max_segment_length:
                num_segments = int(math.ceil(distance / max_segment_length))
                for j in range(1, num_segments):
                    t = j / num_segments
                    interp_x = int(x1 + t * (x2 - x1))
                    interp_y = int(y1 + t * (y2 - y1))
                    densified_path.append((interp_x, interp_y))
            
            densified_path.append((x2, y2))
        
        return densified_path
    
    def _remove_duplicate_edges(self):
        """Remove duplicate edges that might have been created during path tracing"""
        edges_to_remove = []
        
        for edge in list(self.graph.edges()):
            node1, node2 = edge
            # Check if reverse edge exists with longer path
            if self.graph.has_edge(node2, node1):
                # Keep the edge with the longer, more detailed path
                path1 = self.graph[node1][node2].get('path', [])
                path2 = self.graph[node2][node1].get('path', [])
                
                if len(path1) < len(path2):
                    edges_to_remove.append((node1, node2))
                elif len(path2) < len(path1):
                    edges_to_remove.append((node2, node1))
                elif node1 > node2:  # Keep only one if paths are same length
                    edges_to_remove.append((node1, node2))
        
        # Remove the identified duplicate edges
        for edge in edges_to_remove:
            if self.graph.has_edge(*edge):
                self.graph.remove_edge(*edge)
    
    def visualize_graph(self, binary=None, skeleton=None):
        """Visualize the original image and the extracted graph"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Original image
        axes[0,0].imshow(self.image_rgb)
        axes[0,0].set_title('Original Image')
        axes[0,0].axis('off')
        
        # Binary image (if provided)
        if binary is not None:
            axes[0,1].imshow(binary, cmap='gray')
            axes[0,1].set_title('Binary (Thresholded)')
            axes[0,1].axis('off')
        else:
            axes[0,1].axis('off')
        
        # Skeleton image (if provided)  
        if skeleton is not None:
            axes[0,2].imshow(skeleton, cmap='gray')
            axes[0,2].set_title('Skeleton (Single-pixel paths)')
            axes[0,2].axis('off')
        else:
            axes[0,2].axis('off')
        
        # Detected colored points
        axes[1,0].imshow(self.image_rgb)
        for point in self.colored_points['yellow']:
            axes[1,0].plot(point[0], point[1], 'yo', markersize=10, label='Yellow')
        for point in self.colored_points['orange']:
            axes[1,0].plot(point[0], point[1], 'ro', markersize=10, label='Orange')
        axes[1,0].set_title('Detected Colored Points')
        axes[1,0].legend()
        axes[1,0].axis('off')
        
        # Graph visualization on original image
        axes[1,1].imshow(self.image_rgb)
        
        # Draw nodes
        pos = nx.get_node_attributes(self.graph, 'pos')
        node_types = nx.get_node_attributes(self.graph, 'type')
        
        for node, position in pos.items():
            color = 'blue'
            size = 50
            if node_types[node] == 'yellow':
                color = 'yellow'
                size = 100
            elif node_types[node] == 'orange':
                color = 'orange'
                size = 100
            elif node_types[node] == 'intersection':
                color = 'red'
                size = 80
            elif node_types[node] == 'endpoint':
                color = 'green'
                size = 60
                
            axes[1,1].scatter(position[0], position[1], c=color, s=size, 
                            edgecolors='black', linewidth=1, zorder=5)
        
        # Draw edges following actual curved paths
        for edge in self.graph.edges(data=True):
            node1, node2, edge_data = edge
            
            # Get the actual path stored in the edge
            if 'path' in edge_data and len(edge_data['path']) > 1:
                path = edge_data['path']
                # Extract x and y coordinates
                x_coords = [point[0] for point in path]
                y_coords = [point[1] for point in path]
                # Draw the curved path
                axes[1,1].plot(x_coords, y_coords, 'b-', alpha=0.7, linewidth=2)
            else:
                # Fallback to straight line if no path data
                pos1 = pos[node1]
                pos2 = pos[node2]
                axes[1,1].plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'r--', alpha=0.5, linewidth=1)
        
        axes[1,1].set_title('Extracted Graph Overlay')
        axes[1,1].axis('off')
        
        # Pure graph visualization
        pos_dict = {node: pos[node] for node in self.graph.nodes()}
        
        # Flip y-coordinates for proper display (matplotlib vs image coordinates)
        height = self.image_rgb.shape[0]
        pos_flipped = {node: (x, height - y) for node, (x, y) in pos_dict.items()}
        
        node_colors = []
        node_sizes = []
        for node in self.graph.nodes():
            if node_types[node] == 'yellow':
                node_colors.append('yellow')
                node_sizes.append(300)
            elif node_types[node] == 'orange':
                node_colors.append('orange')
                node_sizes.append(300)
            elif node_types[node] == 'intersection':
                node_colors.append('red')
                node_sizes.append(200)
            else:
                node_colors.append('lightblue')
                node_sizes.append(150)
        
        # Draw nodes only first (without edges)
        nx.draw_networkx_nodes(self.graph, pos_flipped, ax=axes[1,2],
                              node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_labels(self.graph, pos_flipped, ax=axes[1,2],
                               font_size=8, font_weight='bold')
        
        # Draw curved edges manually
        for edge in self.graph.edges(data=True):
            node1, node2, edge_data = edge
            
            if 'path' in edge_data and len(edge_data['path']) > 1:
                path = edge_data['path']
                # Flip y-coordinates for graph display
                x_coords = [point[0] for point in path]
                y_coords = [height - point[1] for point in path]  # Flip Y
                axes[1,2].plot(x_coords, y_coords, 'gray', alpha=0.7, linewidth=2)
            else:
                # Fallback to straight line
                pos1 = pos_flipped[node1]
                pos2 = pos_flipped[node2]
                axes[1,2].plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                              'red', alpha=0.5, linewidth=1, linestyle='--')
        
        axes[1,2].set_title('Graph Structure (Curved Paths)')
        
        plt.tight_layout()
        plt.savefig('graph_extraction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print graph statistics
        print(f"\n=== Graph Statistics ===")
        print(f"Nodes: {self.graph.number_of_nodes()}")
        print(f"Edges: {self.graph.number_of_edges()}")
        print(f"Yellow points: {len(self.colored_points['yellow'])}")
        print(f"Orange points: {len(self.colored_points['orange'])}")
        
        # Print edge details with arc lengths
        print(f"\n=== Edge Details (Arc Lengths) ===")
        for edge in self.graph.edges(data=True):
            node1, node2, data = edge
            node1_type = self.graph.nodes[node1]['type']
            node2_type = self.graph.nodes[node2]['type']
            arc_length = data['weight']
            path_points = len(data.get('path', []))
            print(f"Edge {node1}({node1_type}) → {node2}({node2_type}): Arc Length = {arc_length:.2f} pixels, Path Points = {path_points}")
        
        # Print node details
        print(f"\n=== Node Details ===")
        for node in self.graph.nodes(data=True):
            print(f"Node {node[0]}: {node[1]}")

def main():
    # Initialize the converter
    converter = ImageToGraph('Path maps image.jpg')
    
    try:
        # Step 1: Load and preprocess image
        image = converter.load_and_preprocess_image()
        
        # Step 2: Detect colored points
        yellow_mask, orange_mask = converter.detect_colored_points()
        
        # Step 3: Extract path structure
        binary, skeleton = converter.extract_path_structure()
        
        # Step 4: Find key points
        intersections, endpoints = converter.find_key_points(skeleton)
        
        # Step 5: Build graph
        graph = converter.build_graph(skeleton, intersections, endpoints)
        
        # Step 6: Visualize results
        converter.visualize_graph(binary, skeleton)
        
        print("Graph extraction completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 