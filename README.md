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
Every bridge point has to be connected
- [ ] 1. straight line connection
- [ ] 2. visibility algorithm with lines in random directions
- [ ] 3. A*
- [ ] Check if valid path between all bridge points

Make 3 levels of cennectivity:
- [ ] 1. only connect in straight lines
- [ ] 3. connect all vertexes which are possible
- [ ] Implement metric to measure effects

Integrate with SH-Graph
- [ ] create graph from paths
- [ ] integrate room graph into sh-graph
