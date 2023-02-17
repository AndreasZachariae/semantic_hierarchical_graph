# Semantic Hierarchical Graph for Navigation

Creates a hierarchical data structure where each node in a graph is a new graph.

Makes it possible to plan recursive:

```python
path_dict = G.plan_recursive(["Building F", "Floor 0", "Lab"], ["Building A", "Floor 0", "Entrance"])
```

Solution path:

![path](docs/path.png)

```json
// path.json
{
    "LTC Campus": {
        "Building F": {
            "Building A_h_bridge": {},
            "Floor 0": {
                "Lab": {},
                "Staircase": {},
                "Floor 1_h_bridge": {}
            },
            "Floor 1": {
                "Floor 0_h_bridge": {},
                "Staircase": {},
                "Corridor": {},
                "Kitchen": {},
                "Building A_Floor 1_h_bridge": {}
            }
        },
        "Building A": {
            "Building F_h_bridge": {},
            "Floor 1": {
                "Building F_Floor 1_h_bridge": {},
                "Cantina": {},
                "Staircase": {},
                "Floor 0_h_bridge": {}
            },
            "Floor 0": {
                "Floor 1_h_bridge": {},
                "Staircase": {},
                "Entrance": {}
            }
        }
    }
}
```
## Compare durations

Comparison for graph shown above with **3 levels of hierarchy and 23 leaf nodes**

Recursive function in hierarchical tree: 

```bash
python -m timeit -r 10 -s 'from semantic_hierarchical_graph.main import main; G = main()' 'G.plan_recursive(["Building F", "Floor 0", "Lab"], ["Building A", "Floor 0", "Entrance"])'

5000 loops, best of 10: 85.7 usec per loop
```
Package networkx.shortest_path on full leaf graph:

```bash
python -m timeit -r 10 -s 'from semantic_hierarchical_graph.main import main; G = main()' 'G.plan(["Building F", "Floor 0", "Lab"], ["Building A", "Floor 0", "Entrance"])'

10000 loops, best of 10: 24.3 usec per loop
```

## TODO
Improvements
- [ ] (optional) use dist_transform instead of ws_erosion
- [x] remove brisge points which are in collision with ws_erosion
- [x] clear bridge edges in the beginning with dist_transform or ws_erosion
- [ ] Detect if the whole room is a corridor (With corridor width thresholds) and don't collapse rectangles in other rooms
- [ ] Adjust planning in graph for multiple possible paths. Plan all paths and compare lenghts for shortest
- [ ] Make sure correct distances are in the graph edges

Every bridge point has to be connected
- [x] 1. straight line connection
- [ ] 2. visibility algorithm with lines in random directions
- [ ] 2.1. Find orthogonal lines between bridge point and closest path and find third point on this line which connects both
- [ ] 2.2. Change the angle of the line in x degree steps and change distance of thrid point in x steps
- [ ] 3. A*
- [ ] Check if valid path between all bridge points

Smoothing
- [ ] Find shortcuts in the graph but not through largest rectangle

Make 3 levels of connectivity:
- [x] 1. only connect in straight lines
- [x] 3. connect all vertexes which are possible
- [ ] Implement metric to measure effects

Integrate with SH-Graph
- [x] create graph from paths
- [x] integrate room graph into sh-graph
