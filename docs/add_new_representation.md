# Add New Map4D Representation

GT-only ManiSkill task needs these files/edits.

## 1. Add Representation Class

Create:

```text
map4d/representation/maps4d/maniskill_xxx.py
```

Implement a map class with the current GT constructor signature:

```python
class Map4d_Xxx(Map_4d):
    def __init__(self, positions, rotations, size_parameters, relation_parameters):
        ...
```

`positions`, `rotations`, `size_parameters`, and `relation_parameters` are flat tensors.

## 2. Add Metadata JSON

Create:

```text
map4d/representation/maps4d/maniskill_xxx.json
```

Required fields:

```json
{
  "TaskName-v1": {
    "representation": "Map4d_Xxx",
    "actor_names": ["actor_a", "actor_b"],
    "size_parameters": {"dim": 0, "default": []},
    "relation_parameters": {"dim": 0, "default": []},
    "objects": {
      "actor_a": {
        "actor_name": "actor_a",
        "position_slice": [0, 3],
        "rotation_slice": [0, 6],
        "size_slice": [0, 0]
      }
    }
  }
}
```

`actor_names` must match `env_states/actors/<name>` in the ManiSkill HDF5.

## 3. Register Metadata

Edit:

```text
map4d/representation/maps4d/metadata.py
```

Add:

```python
TASK_METADATA_FILES["TaskName-v1"] = "maniskill_xxx.json"
```

## 4. Register Constructor

Edit:

```text
map4d/construction/map_constructor.py
```

Import:

```python
from maniskill_xxx import Map4d_Xxx
```

Add to `ManiSkillGTMap4dConstructor.MAP_CLASSES`:

```python
"TaskName-v1": Map4d_Xxx,
```
