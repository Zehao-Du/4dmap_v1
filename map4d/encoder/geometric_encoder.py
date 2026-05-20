import torch
import torch.nn as nn


class GeometricEncoder(nn.Module):
    """Encode 4D map sequences into per-step features and predict pose deltas."""

    def __init__(
        self,
        num_objects: int,
        input_dim: int = 30,
        node_dim: int = 128,
        relation_dim: int = 64,
        temporal_dim: int = 128,
        feature_dim: int = 128,
    ):
        super().__init__()
        if num_objects <= 0:
            raise ValueError("num_objects must be positive")
        self.num_objects = num_objects
        self.feature_dim = feature_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.relation_mlp = nn.Sequential(
            nn.Linear(10, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim),
        )
        self.relation_proj = nn.Linear(relation_dim, node_dim)
        self.temporal_gru = nn.GRU(node_dim, temporal_dim, batch_first=True)
        self.obj_proj = nn.Linear(temporal_dim, feature_dim)
        self.scene_proj = nn.Linear(feature_dim, feature_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 9),
        )

    def forward(self, rep_seq):
        sizes, positions, rotations = _parse_representation(rep_seq, self.num_objects)
        sizes, positions, rotations = _ensure_batch_time(sizes, positions, rotations)
        device = next(self.parameters()).device
        sizes = sizes.to(device)
        positions = positions.to(device)
        rotations = rotations.to(device)
        B, T, N, _ = sizes.shape

        vel_pos = _time_diff(positions)
        vel_rot = _time_diff(rotations)
        acc_pos = _time_diff(vel_pos)
        acc_rot = _time_diff(vel_rot)

        vel_pos = _pad_time_front(vel_pos, positions)
        vel_rot = _pad_time_front(vel_rot, rotations)
        acc_pos = _pad_time_front(acc_pos, positions)
        acc_rot = _pad_time_front(acc_rot, rotations)

        node_input = torch.cat(
            [sizes, positions, rotations, vel_pos, vel_rot, acc_pos, acc_rot], dim=-1
        )
        node_feat = self.node_mlp(node_input)

        rel_feat = _pairwise_relation(positions, rotations)
        rel_emb = self.relation_mlp(rel_feat)
        rel_emb = _mask_self(rel_emb)
        rel_agg = rel_emb.mean(dim=3)
        node_feat = node_feat + self.relation_proj(rel_agg)

        node_feat = node_feat.transpose(1, 2).contiguous().view(B * N, T, -1)
        temporal_out, _ = self.temporal_gru(node_feat)
        temporal_out = temporal_out.view(B, N, T, -1).transpose(1, 2)

        obj_feat = self.obj_proj(temporal_out)
        scene_feat = obj_feat.mean(dim=2)
        map_feature_seq = self.scene_proj(scene_feat)

        pred = {}
        if T > 1:
            pred_delta = self.pred_head(obj_feat[:, :-1])
            pred_delta_pos = pred_delta[..., :3]
            pred_delta_rot = pred_delta[..., 3:]
            pred_pos = positions[:, :-1] + pred_delta_pos
            pred_rot = rotations[:, :-1] + pred_delta_rot
            pred["pred_delta_pos"] = pred_delta_pos
            pred["pred_delta_rot"] = pred_delta_rot
            pred["pred_pos"] = pred_pos
            pred["pred_rot"] = pred_rot
            pred["valid_mask"] = torch.ones((B, T - 1, N), device=positions.device)
        else:
            pred["pred_delta_pos"] = positions.new_zeros((B, 0, N, 3))
            pred["pred_delta_rot"] = rotations.new_zeros((B, 0, N, 6))
            pred["pred_pos"] = positions.new_zeros((B, 0, N, 3))
            pred["pred_rot"] = rotations.new_zeros((B, 0, N, 6))
            pred["valid_mask"] = positions.new_zeros((B, 0, N))

        pred["sizes"] = sizes
        pred["positions"] = positions
        pred["rotations"] = rotations
        pred["object_features"] = obj_feat
        pred["scene_features"] = scene_feat

        return map_feature_seq, pred


def _parse_representation(rep_seq, num_objects: int):
    if _is_map4d(rep_seq):
        return _extract_map4d(rep_seq)

    if isinstance(rep_seq, (list, tuple)) and len(rep_seq) == 3:
        if all(_is_tensor_like(item) for item in rep_seq):
            return rep_seq[0], rep_seq[1], rep_seq[2]

    if isinstance(rep_seq, (list, tuple)) and len(rep_seq) > 0:
        if all(_is_map4d(item) for item in rep_seq):
            return _extract_map4d_sequence(rep_seq)

    if isinstance(rep_seq, dict):
        sizes = rep_seq.get("sizes") or rep_seq.get("size")
        positions = rep_seq.get("positions") or rep_seq.get("position")
        rotations = rep_seq.get("rotations") or rep_seq.get("rotation")
        if sizes is None or positions is None or rotations is None:
            raise ValueError("rep_seq dict must include sizes/positions/rotations")
        sizes = _reshape_attr(sizes, num_objects, 3)
        positions = _reshape_attr(positions, num_objects, 3)
        rotations = _reshape_attr(rotations, num_objects, 6)
        return sizes, positions, rotations

    if torch.is_tensor(rep_seq):
        rep = rep_seq
        if rep.ndim == 4 and rep.shape[-1] == 12:
            sizes, positions, rotations = rep[..., :3], rep[..., 3:6], rep[..., 6:12]
            return sizes, positions, rotations
        if rep.ndim == 3:
            if rep.shape[-1] % 12 != 0:
                raise ValueError("Flattened rep_seq must have last dim divisible by 12")
            N = num_objects or rep.shape[-1] // 12
            rep = rep.view(rep.shape[0], rep.shape[1], N, 12)
            return rep[..., :3], rep[..., 3:6], rep[..., 6:12]
    raise TypeError("Unsupported rep_seq format")


def _is_map4d(obj):
    return hasattr(obj, "Objects") or hasattr(obj, "objects")


def _is_tensor_like(value):
    return torch.is_tensor(value) or hasattr(value, "shape")


def _extract_map4d_sequence(seq):
    sizes_list = []
    positions_list = []
    rotations_list = []
    for item in seq:
        sizes, positions, rotations = _extract_map4d(item)
        sizes_list.append(sizes)
        positions_list.append(positions)
        rotations_list.append(rotations)
    sizes = torch.stack(sizes_list, dim=0).transpose(0, 1)
    positions = torch.stack(positions_list, dim=0).transpose(0, 1)
    rotations = torch.stack(rotations_list, dim=0).transpose(0, 1)
    return sizes, positions, rotations


def _extract_map4d(map4d):
    objects = getattr(map4d, "Objects", None)
    if objects is None:
        objects = getattr(map4d, "objects", None)
    if objects is None:
        raise ValueError("Map4d object must have Objects")

    sizes_list = []
    positions_list = []
    rotations_list = []
    device = None
    for obj in objects:
        nodes = getattr(obj, "Nodes", None)
        if nodes is None or len(nodes) == 0:
            continue
        node = nodes[0]
        pos = _to_tensor(getattr(node, "position", None))
        rot = _to_tensor(getattr(node, "rotation", None))
        if pos is None or rot is None:
            raise ValueError("Map4d node must include position and rotation")
        if device is None:
            device = pos.device if torch.is_tensor(pos) else rot.device
        pos = _to_tensor(pos, device=device)
        rot = _to_tensor(rot, device=device)
        size = _node_size(node, device=device)
        sizes_list.append(size)
        positions_list.append(pos)
        rotations_list.append(rot)

    if len(sizes_list) == 0:
        raise ValueError("Map4d object contains no nodes to parse")

    sizes = torch.stack(sizes_list, dim=1)
    positions = torch.stack(positions_list, dim=1)
    rotations = torch.stack(rotations_list, dim=1)
    return sizes, positions, rotations


def _node_size(node, device=None):
    if all(hasattr(node, name) for name in ("height", "top_length", "top_width")):
        height = _to_tensor(getattr(node, "height"), device=device)
        top_length = _to_tensor(getattr(node, "top_length"), device=device)
        top_width = _to_tensor(getattr(node, "top_width"), device=device)
        return torch.stack([height, top_length, top_width], dim=-1)

    node_pos = getattr(node, "Node_Position", None)
    if node_pos is not None:
        pos = _to_tensor(node_pos, device=device)
        size = pos.max(dim=1).values - pos.min(dim=1).values
        return size

    raise ValueError("Map4d node size cannot be inferred")


def _to_tensor(value, device=None):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value if device is None else value.to(device)
    return torch.as_tensor(value, device=device)


def _reshape_attr(value, num_objects, attr_dim: int):
    if torch.is_tensor(value):
        tensor = value
    else:
        tensor = torch.as_tensor(value)

    if tensor.ndim == 4:
        return tensor
    if tensor.ndim == 3:
        if tensor.shape[-1] == attr_dim:
            return tensor.unsqueeze(2)
        if tensor.shape[-1] % attr_dim == 0:
            N = num_objects or tensor.shape[-1] // attr_dim
            return tensor.view(tensor.shape[0], tensor.shape[1], N, attr_dim)
    if tensor.ndim == 2:
        if tensor.shape[-1] == attr_dim:
            return tensor.unsqueeze(0).unsqueeze(2)
        if tensor.shape[-1] % attr_dim == 0:
            N = num_objects or tensor.shape[-1] // attr_dim
            return tensor.view(1, tensor.shape[0], N, attr_dim)
    raise ValueError("Unexpected attribute shape for map4d representation")


def _ensure_batch_time(sizes, positions, rotations):
    if sizes.ndim == 3:
        sizes = sizes.unsqueeze(0)
    if positions.ndim == 3:
        positions = positions.unsqueeze(0)
    if rotations.ndim == 3:
        rotations = rotations.unsqueeze(0)
    return sizes, positions, rotations


def _time_diff(values):
    return values[:, 1:] - values[:, :-1]


def _pad_time_front(values, ref):
    if values.shape[1] == 0:
        return values
    pad = ref[:, :1].new_zeros((ref.shape[0], 1, ref.shape[2], values.shape[-1]))
    return torch.cat([pad, values], dim=1)


def _pairwise_relation(positions, rotations):
    pos_i = positions.unsqueeze(3)
    pos_j = positions.unsqueeze(2)
    rot_i = rotations.unsqueeze(3)
    rot_j = rotations.unsqueeze(2)
    rel_pos = pos_j - pos_i
    rel_rot = rot_j - rot_i
    dist = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
    return torch.cat([rel_pos, rel_rot, dist], dim=-1)


def _mask_self(rel_emb):
    B, T, N, _, _ = rel_emb.shape
    mask = 1.0 - torch.eye(N, device=rel_emb.device).view(1, 1, N, N, 1)
    return rel_emb * mask
