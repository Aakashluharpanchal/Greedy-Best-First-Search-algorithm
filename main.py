import numpy as np
import matplotlib.pyplot as plt
import heapq

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.h = 0  # Heuristic value

    def __lt__(self, other):
        return self.h < other.h  # Compare nodes based on heuristic (for priority queue)


def heuristic(a, b):
    """Calculate the Manhattan distance between points a and b."""
    return abs(a.x - b.x) + abs(a.y - b.y)


def generate_random_map(width, height, obstacle_prob):
    """Generate a random map with obstacles."""
    map_grid = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if np.random.random() < obstacle_prob:
                map_grid[i, j] = 1  # 1 indicates an obstacle
    return map_grid


def is_valid_node(map_grid, node):
    """Check if the node is within bounds and not an obstacle."""
    if 0 <= node.y < map_grid.shape[0] and 0 <= node.x < map_grid.shape[1]:
        return map_grid[node.y, node.x] == 0  # 0 means walkable
    return False


def greedy_best_first_search(map_grid, start, goal):
    """Greedy Best-First Search implementation."""
    open_list = []
    closed_list = set()

    start_node = Node(*start)
    goal_node = Node(*goal)
    start_node.h = heuristic(start_node, goal_node)

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        if (current_node.x, current_node.y) == (goal_node.x, goal_node.y):
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]  # Return path from start to goal

        closed_list.add((current_node.x, current_node.y))

        # Move in 4 possible directions
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_pos = (current_node.x + dx, current_node.y + dy)
            neighbor_node = Node(*neighbor_pos, parent=current_node)

            if not is_valid_node(map_grid, neighbor_node) or (neighbor_node.x, neighbor_node.y) in closed_list:
                continue

            neighbor_node.h = heuristic(neighbor_node, goal_node)
            heapq.heappush(open_list, neighbor_node)

    return None  # Return None if no path is found


def plot_path(map_grid, path, start, goal):
    """Visualize the path on the map."""
    plt.imshow(map_grid, cmap='Greys', origin='lower')
    plt.plot(start[0], start[1], 'ro', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'go', markersize=10, label='Goal')

    if path:
        for i in range(1, len(path)):
            plt.plot([path[i - 1][0], path[i][0]], [path[i - 1][1], path[i][1]], 'b-', linewidth=2)

    plt.title('Greedy Best-First Search Pathfinding')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


def main():
    width = 50
    height = 50
    obstacle_prob = 0.3  # Probability of obstacles

    map_grid = generate_random_map(width, height, obstacle_prob)
    start = (5, 5)
    goal = (45, 45)

    path = greedy_best_first_search(map_grid, start, goal)

    if path:
        print("Path found!")
        plot_path(map_grid, path, start, goal)
    else:
        print("No path found.")


if __name__ == "__main__":
    main()
