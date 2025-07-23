from collections import deque
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

class Solution:
    def cloneGraph(self, node):
        if not node: return node
        visited = {}
        Q = deque([node])
        visited[node] = Node(node.val)

        while Q :
            curr = Q.popleft()
            for child in curr.neighbors:
                if child not in visited : 
                    visited[child] = Node(child.val)
                    Q.append(child)
                visited[curr].neighbors.append(visited[child])

        return visited[node]

# Process the graph cloning
# since we neded to clone graoh, need to traverse each of the nodes
# need to use traversal (bfs/dfs) to visit each node
# We need to track already cloned nodes using some data structure
# Visited can be maintained to track already cloned nodes. 
# So visited contains the cloned node itself


def test_cloneGraph():
    # Helper function to compare two graphs
    def are_graphs_equal(node1, node2, visited1=None, visited2=None):
        if visited1 is None:
            visited1 = set()
        if visited2 is None:
            visited2 = set()

        if not node1 or not node2:
            return node1 == node2

        if node1.val != node2.val or len(node1.neighbors) != len(node2.neighbors):
            return False

        visited1.add(node1)
        visited2.add(node2)

        for n1, n2 in zip(node1.neighbors, node2.neighbors):
            if n1 not in visited1 and n2 not in visited2:
                if not are_graphs_equal(n1, n2, visited1, visited2):
                    return False

        return True

    # Test case 1: Empty graph
    solution = Solution()
    assert solution.cloneGraph(None) is None

    # Test case 2: Single node graph
    node1 = Node(1)
    cloned_graph = solution.cloneGraph(node1)
    assert cloned_graph is not node1
    assert cloned_graph.val == node1.val
    assert len(cloned_graph.neighbors) == len(node1.neighbors)

    # Test case 3: Graph with multiple nodes
    node1 = Node(1)
    node2 = Node(2)
    node3 = Node(3)
    node1.neighbors = [node2, node3]
    node2.neighbors = [node1, node3]
    node3.neighbors = [node1, node2]

    cloned_graph = solution.cloneGraph(node1)
    assert are_graphs_equal(node1, cloned_graph)

    print("All test cases passed!")

if __name__ == "__main__":
    test_cloneGraph()