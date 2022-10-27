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
