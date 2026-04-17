"""
Flask server that accepts GLB uploads, auto-rigs them using pynocchio,
and returns rigged GLB files with skeleton and skin weights.
"""

import io
import os
import json
import struct
import tempfile
import traceback

import numpy as np
import trimesh
from flask import Flask, request, jsonify, send_file

import pynocchio
from pynocchio import skeletons

app = Flask(__name__)

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_SKELETON_TYPES = {
    "human": skeletons.HumanSkeleton,
    "quad": skeletons.QuadSkeleton,
    "horse": skeletons.HorseSkeleton,
    "centaur": skeletons.CentaurSkeleton,
}


def glb_to_obj(glb_bytes):
    """Load a GLB file and export as OBJ to a temp file. Returns (obj_path, trimesh_mesh)."""
    scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", force="mesh")
    if isinstance(scene, trimesh.Scene):
        scene = scene.dump(concatenate=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    tmp.close()
    scene.export(tmp.name, file_type="obj")
    return tmp.name, scene


def extract_rig_data(mesh, attachment, skeleton):
    """Extract joint positions, parent indices, and per-vertex bone weights."""
    embedding = attachment.embedding
    num_joints = len(embedding)
    num_verts = len(mesh.vertices)

    joint_positions = np.array([[embedding[j][k] for k in range(3)] for j in range(num_joints)], dtype=np.float32)
    parent_indices = list(skeleton.parent_indices)

    # Extract per-vertex weights
    weights_per_vertex = []
    for i in range(num_verts):
        w = attachment.get_weights(i)
        weights_per_vertex.append(w)

    return joint_positions, parent_indices, weights_per_vertex


def build_rigged_glb(tri_mesh, joint_positions, parent_indices, weights_per_vertex):
    """
    Build a rigged GLB file with skeleton (joints) and skin weights.
    Uses raw glTF JSON construction + binary buffer packing.
    """
    vertices = np.array(tri_mesh.vertices, dtype=np.float32)
    normals = np.array(tri_mesh.vertex_normals, dtype=np.float32)
    faces = np.array(tri_mesh.faces, dtype=np.uint32)

    num_verts = len(vertices)
    num_joints = len(joint_positions)

    # For glTF skinning: each vertex needs JOINTS_0 (4 x uint8) and WEIGHTS_0 (4 x float32)
    # We pick the top 4 bone influences per vertex
    joints_attr = np.zeros((num_verts, 4), dtype=np.uint8)
    weights_attr = np.zeros((num_verts, 4), dtype=np.float32)

    for i, w in enumerate(weights_per_vertex):
        # w is a list of weights, one per bone
        indexed = list(enumerate(w))
        indexed.sort(key=lambda x: -x[1])
        top4 = indexed[:4]
        total = sum(x[1] for x in top4)
        if total > 0:
            for j, (bone_idx, weight) in enumerate(top4):
                joints_attr[i, j] = bone_idx
                weights_attr[i, j] = weight / total  # normalize

    # Build inverse bind matrices (identity-based since joints are in model space)
    inverse_bind_matrices = np.zeros((num_joints, 16), dtype=np.float32)
    for j in range(num_joints):
        # Inverse bind = inverse of the joint's world transform (just translation)
        mat = np.eye(4, dtype=np.float32)
        mat[0, 3] = -joint_positions[j][0]
        mat[1, 3] = -joint_positions[j][1]
        mat[2, 3] = -joint_positions[j][2]
        # glTF uses column-major
        inverse_bind_matrices[j] = mat.T.flatten()

    # --- Pack binary buffer ---
    buffer_parts = []

    def add_buffer_view(data_bytes, target=None):
        offset = sum(len(p) for p in buffer_parts)
        # Align to 4 bytes
        padding = (4 - (offset % 4)) % 4
        if padding:
            buffer_parts.append(b'\x00' * padding)
            offset += padding
        buffer_parts.append(data_bytes)
        bv = {"buffer": 0, "byteOffset": offset, "byteLength": len(data_bytes)}
        if target is not None:
            bv["target"] = target
        return bv

    # Accessors and buffer views
    buffer_views = []
    accessors = []

    # 0: positions
    pos_data = vertices.tobytes()
    bv = add_buffer_view(pos_data, target=34962)
    bv_idx = len(buffer_views)
    buffer_views.append(bv)
    accessors.append({
        "bufferView": bv_idx, "componentType": 5126, "count": num_verts,
        "type": "VEC3",
        "max": vertices.max(axis=0).tolist(),
        "min": vertices.min(axis=0).tolist(),
    })

    # 1: normals
    norm_data = normals.tobytes()
    bv = add_buffer_view(norm_data, target=34962)
    bv_idx = len(buffer_views)
    buffer_views.append(bv)
    accessors.append({
        "bufferView": bv_idx, "componentType": 5126, "count": num_verts,
        "type": "VEC3",
    })

    # 2: indices
    idx_data = faces.flatten().astype(np.uint32).tobytes()
    bv = add_buffer_view(idx_data, target=34963)
    bv_idx = len(buffer_views)
    buffer_views.append(bv)
    accessors.append({
        "bufferView": bv_idx, "componentType": 5125, "count": faces.size,
        "type": "SCALAR",
    })

    # 3: JOINTS_0
    joints_data = joints_attr.tobytes()
    bv = add_buffer_view(joints_data, target=34962)
    bv_idx = len(buffer_views)
    buffer_views.append(bv)
    accessors.append({
        "bufferView": bv_idx, "componentType": 5121, "count": num_verts,
        "type": "VEC4",
    })

    # 4: WEIGHTS_0
    weights_data = weights_attr.tobytes()
    bv = add_buffer_view(weights_data, target=34962)
    bv_idx = len(buffer_views)
    buffer_views.append(bv)
    accessors.append({
        "bufferView": bv_idx, "componentType": 5126, "count": num_verts,
        "type": "VEC4",
    })

    # 5: inverse bind matrices
    ibm_data = inverse_bind_matrices.tobytes()
    bv = add_buffer_view(ibm_data)
    bv_idx = len(buffer_views)
    buffer_views.append(bv)
    accessors.append({
        "bufferView": bv_idx, "componentType": 5126, "count": num_joints,
        "type": "MAT4",
    })

    # --- Build joint node hierarchy ---
    # Node 0 = mesh node, nodes 1..N = joint nodes
    joint_node_offset = 1
    nodes = []

    # Mesh node (node 0)
    mesh_node = {"mesh": 0, "skin": 0, "name": "RiggedMesh"}
    nodes.append(mesh_node)

    # Joint nodes
    children_map = {}
    root_joints = []
    for j in range(num_joints):
        parent = parent_indices[j]
        if parent < 0:
            root_joints.append(j)
        else:
            children_map.setdefault(parent, []).append(j)

    for j in range(num_joints):
        node = {
            "name": f"Joint_{j}",
            "translation": joint_positions[j].tolist(),
        }
        if j in children_map:
            node["children"] = [c + joint_node_offset for c in children_map[j]]
        nodes.append(node)

    # Convert joint translations to local space (relative to parent)
    for j in range(num_joints):
        parent = parent_indices[j]
        if parent >= 0:
            nodes[j + joint_node_offset]["translation"] = (
                joint_positions[j] - joint_positions[parent]
            ).tolist()

    # Scene root children: mesh node + root joint nodes
    scene_nodes = [0] + [j + joint_node_offset for j in root_joints]

    # Skin
    skin = {
        "joints": list(range(joint_node_offset, joint_node_offset + num_joints)),
        "inverseBindMatrices": 5,  # accessor index for IBM
        "skeleton": root_joints[0] + joint_node_offset if root_joints else joint_node_offset,
    }

    # Mesh primitive
    primitive = {
        "attributes": {
            "POSITION": 0,
            "NORMAL": 1,
            "JOINTS_0": 3,
            "WEIGHTS_0": 4,
        },
        "indices": 2,
    }

    # Total buffer
    total_buffer_bytes = b''.join(buffer_parts)
    # Pad to 4-byte alignment
    buf_padding = (4 - (len(total_buffer_bytes) % 4)) % 4
    total_buffer_bytes += b'\x00' * buf_padding

    gltf = {
        "asset": {"version": "2.0", "generator": "pynocchio-rigging-server"},
        "scene": 0,
        "scenes": [{"nodes": scene_nodes}],
        "nodes": nodes,
        "meshes": [{"primitives": [primitive], "name": "RiggedMesh"}],
        "skins": [skin],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(total_buffer_bytes)}],
    }

    # --- Pack as GLB ---
    gltf_json = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    # Pad JSON to 4-byte alignment
    json_padding = (4 - (len(gltf_json) % 4)) % 4
    gltf_json += b' ' * json_padding

    # GLB header: magic, version, total length
    total_length = 12 + 8 + len(gltf_json) + 8 + len(total_buffer_bytes)
    header = struct.pack('<III', 0x46546C67, 2, total_length)  # glTF magic

    # JSON chunk
    json_chunk_header = struct.pack('<II', len(gltf_json), 0x4E4F534A)  # JSON type

    # BIN chunk
    bin_chunk_header = struct.pack('<II', len(total_buffer_bytes), 0x004E4942)  # BIN type

    glb_data = header + json_chunk_header + gltf_json + bin_chunk_header + total_buffer_bytes
    return glb_data


@app.route("/rig", methods=["POST"])
def rig_model():
    """
    POST /rig
    
    Form data:
      - file: GLB file upload
      - skeleton: skeleton type (human, quad, horse, centaur). Default: human
      - scale: skeleton scale factor (float). Default: 1.0
    
    Returns: rigged GLB file
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use 'file' form field."}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "Empty filename"}), 400

    skeleton_type = request.form.get("skeleton", "human").lower()
    if skeleton_type not in ALLOWED_SKELETON_TYPES:
        return jsonify({
            "error": f"Unknown skeleton type: {skeleton_type}",
            "allowed": list(ALLOWED_SKELETON_TYPES.keys()),
        }), 400

    scale = float(request.form.get("scale", "1.0"))

    glb_bytes = uploaded.read()
    if len(glb_bytes) > MAX_UPLOAD_SIZE:
        return jsonify({"error": f"File too large (max {MAX_UPLOAD_SIZE // 1024 // 1024}MB)"}), 413

    obj_path = None
    try:
        # 1. Convert GLB to OBJ
        obj_path, tri_mesh = glb_to_obj(glb_bytes)

        # 2. Load mesh in pynocchio
        pino_mesh = pynocchio.Mesh(obj_path)

        # 3. Create skeleton
        skeleton = ALLOWED_SKELETON_TYPES[skeleton_type]()
        if scale != 1.0:
            skeleton.scale(scale)

        # 4. Auto-rig
        attachment = pynocchio.auto_rig(skeleton, pino_mesh)

        # 5. Extract rig data
        joint_positions, parent_indices, weights_per_vertex = extract_rig_data(
            pino_mesh, attachment, skeleton
        )

        # 6. Build rigged GLB
        rigged_glb = build_rigged_glb(tri_mesh, joint_positions, parent_indices, weights_per_vertex)

        return send_file(
            io.BytesIO(rigged_glb),
            mimetype="model/gltf-binary",
            as_attachment=True,
            download_name="rigged_model.glb",
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        if obj_path and os.path.exists(obj_path):
            os.unlink(obj_path)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "skeletons": list(ALLOWED_SKELETON_TYPES.keys())})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
