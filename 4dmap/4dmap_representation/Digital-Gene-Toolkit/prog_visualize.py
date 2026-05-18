import argparse
import importlib
import json
import os
import pickle

import open3d as o3d


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category", default="item", type=str, help="category of objects to generate"
    )
    parser.add_argument(
        "--gen_num", default=10, type=int, help="number of objects to generate"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    module = importlib.import_module(
        f"code.categories.{args.category}.procedural_generation"
    )
    get_type = getattr(module, f"get_{(args.category).lower()}_type")

    concept_list = []
    for obj_idx in range(args.gen_num):
        item_type = get_type()
        existing_concept_templates = module.concept_template_existence(item_type)
        default_params_path = f"{os.path.dirname(os.path.abspath(__file__))}/assets/default_params/{args.category}.json"
        default_params = json.load(open(default_params_path, "r"))
        concepts = []
        for template in existing_concept_templates:
            concepts.append(
                {"template": template, "parameters": default_params[template]}
            )
        new_concepts = module.jitter_parameters(concepts, item_type)
        concept_list.append(new_concepts)

        vertices, faces = module.get_overall_model(new_concepts)
        vertices = o3d.utility.Vector3dVector(vertices)
        faces = o3d.utility.Vector3iVector(faces)
        opt_mesh = o3d.geometry.TriangleMesh(vertices, faces)
        opt_mesh.compute_vertex_normals()

        o3d.visualization.draw_geometries([opt_mesh])

    if not os.path.exists("./assets/conceptualizations"):
        os.mkdir("./assets/conceptualizations")

    if not os.path.exists("./assets/conceptualizations/prog"):
        os.mkdir("./assets/conceptualizations/prog")

    save_path = f"./assets/conceptualizations/prog/{args.category}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(concept_list, f)


if __name__ == "__main__":
    main()
