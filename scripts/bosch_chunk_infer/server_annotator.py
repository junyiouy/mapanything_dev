import argparse
import json
import os
import time
import numpy as np
import trimesh
import viser
import viser.transforms as tf
from scipy.spatial.transform import Rotation

# --- Math Helpers ---

def get_bbox_edge_points(extent):
    hx, hy, hz = extent / 2
    corners = np.array([
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz]
    ])
    lines_indices = [
        (0,1), (1,2), (2,3), (3,0), 
        (4,5), (5,6), (6,7), (7,4), 
        (0,4), (1,5), (2,6), (3,7)
    ]
    segments = []
    for start, end in lines_indices:
        segments.append([corners[start], corners[end]])
    return np.array(segments)

def ray_point_intersection(ray_origin, ray_dir, points, threshold=0.1):
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    vec_op = points - ray_origin
    t = np.dot(vec_op, ray_dir)
    closest_on_line = ray_origin + t[:, np.newaxis] * ray_dir
    dist_sq = np.sum((points - closest_on_line)**2, axis=1)
    mask = (t > 0) & (dist_sq < threshold**2)
    if not np.any(mask): return None
    valid_indices = np.where(mask)[0]
    sorted_by_dist = valid_indices[np.argsort(dist_sq[valid_indices])]
    candidates = sorted_by_dist[:50]
    best_idx = candidates[np.argmin(t[candidates])]
    return points[best_idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_json", type=str, required=True, help="Path to scene_meta.json")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if not os.path.exists(args.meta_json):
        print(f"Error: {args.meta_json} not found")
        return

    with open(args.meta_json, 'r') as f:
        meta_data = json.load(f)
    frames = meta_data['frames']
    base_dir = os.path.dirname(args.meta_json)
    # Flexible chunk/group name retrieval
    chunk_name = meta_data.get('group_name', meta_data.get('chunk_name', 'unknown'))

    server = viser.ViserServer(port=args.port)
    print(f"Server started at http://localhost:{args.port}")

    # --- Global State ---
    state = {
        "frame_idx": 0,
        "points": np.zeros((1, 3)), # Currently displayed points (Single or All)
        "colors": np.zeros((1, 3)),
        "bbox_pos": np.zeros(3),
        "bbox_rot": np.array([1.0, 0.0, 0.0, 0.0]), 
        "bbox_extent": np.array([2.0, 2.0, 2.0]),
        "c2w": np.eye(4),
        
        # New State for "Show All"
        "show_all_frames": False,
        "cached_all_points": None,
        "cached_all_colors": None
    }

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        print(f"Client connected: {client.client_id}")

        handles = {
            "pcd": None,
            "gizmo": None,
            "bbox_wireframe": None,
            "cam_frame": None
        }

        # --- GUI ---
        with client.gui.add_folder("1. Navigation"):
            slider_frame = client.gui.add_slider("Frame", 0, len(frames)-1, 1, 0)
            # 【新增功能】显示所有帧的开关
            chk_show_all = client.gui.add_checkbox("Show All Frames (Global Map)", False)
            btn_reset_cam = client.gui.add_button("Reset Camera (View)")

        with client.gui.add_folder("2. Annotation Tools"):
            chk_edit = client.gui.add_checkbox("Enable Edit Mode", False)
            btn_place = client.gui.add_button("Click to Place (Snap)", visible=False)
            slider_x = client.gui.add_slider("Len X", 0.1, 10.0, 0.1, 2.0, disabled=True)
            slider_y = client.gui.add_slider("Wid Y", 0.1, 10.0, 0.1, 2.0, disabled=True)
            slider_z = client.gui.add_slider("Hei Z", 0.1, 10.0, 0.1, 2.0, disabled=True)
            btn_save = client.gui.add_button("SAVE JSON", visible=False)

        # --- Visual Helpers ---
        def remove_edit_visuals():
            if handles["gizmo"]: handles["gizmo"].remove(); handles["gizmo"] = None
            if handles["bbox_wireframe"]: handles["bbox_wireframe"].remove(); handles["bbox_wireframe"] = None

        def update_edit_visuals():
            remove_edit_visuals()
            if not chk_edit.value: return
            
            handles["gizmo"] = client.scene.add_transform_controls(
                "/gizmo", position=state["bbox_pos"], wxyz=state["bbox_rot"], scale=1.0
            )
            segments = get_bbox_edge_points(state["bbox_extent"])
            handles["bbox_wireframe"] = client.scene.add_line_segments(
                "/gizmo/bbox_lines", points=segments, colors=(0, 255, 0), line_width=3.0,
            )
            
            @handles["gizmo"].on_update
            def _(_):
                state["bbox_pos"] = np.array(handles["gizmo"].position)
                state["bbox_rot"] = np.array(handles["gizmo"].wxyz)

        # --- Data Loading Helpers ---
        
        def load_all_frames_to_cache():
            """Lazy load all frames and merge them."""
            if state["cached_all_points"] is not None:
                return

            print("Loading all frames into memory... this may take a moment.")
            all_pts = []
            all_cols = []
            
            # Simple downsampling factor for global view to save memory/render time
            # If you have >100 frames, maybe skip every 2nd point or similar
            # For now, we load raw.
            
            for frame in frames:
                path = os.path.join(base_dir, frame['ply_path'])
                if os.path.exists(path):
                    pcd = trimesh.load(path)
                    all_pts.append(pcd.vertices)
                    c = pcd.colors[:, :3] if hasattr(pcd, 'colors') else np.ones_like(pcd.vertices)*255
                    all_cols.append(c)
            
            if all_pts:
                state["cached_all_points"] = np.concatenate(all_pts, axis=0)
                state["cached_all_colors"] = np.concatenate(all_cols, axis=0)
                print(f"Global Map Loaded: {len(state['cached_all_points'])} points.")
            else:
                state["cached_all_points"] = np.zeros((1, 3))
                state["cached_all_colors"] = np.zeros((1, 3))

        def load_frame(idx):
            info = frames[idx]
            
            # 1. Update Camera Pose (Always current frame)
            state["c2w"] = np.array(info['transform_matrix'])
            if handles["cam_frame"]: handles["cam_frame"].remove()
            handles["cam_frame"] = client.scene.add_frame(
                "/cam_pose", wxyz=tf.SO3.from_matrix(state["c2w"][:3, :3]).wxyz,
                position=state["c2w"][:3, 3], axes_length=0.5
            )

            # 2. Update Point Cloud (Depends on Show All)
            if state["show_all_frames"]:
                # Ensure cache is populated
                load_all_frames_to_cache()
                state["points"] = state["cached_all_points"]
                state["colors"] = state["cached_all_colors"]
            else:
                # Load Single Frame
                path = os.path.join(base_dir, info['ply_path'])
                if os.path.exists(path):
                    pcd = trimesh.load(path)
                    state["points"] = pcd.vertices
                    state["colors"] = pcd.colors[:, :3] if hasattr(pcd, 'colors') else np.ones_like(pcd.vertices)*255
                else:
                    # Empty if missing
                    state["points"] = np.zeros((1, 3))
                    state["colors"] = np.zeros((1, 3))

            # Render Points
            if handles["pcd"]: handles["pcd"].remove()
            handles["pcd"] = client.scene.add_point_cloud(
                "/pcd", 
                points=state["points"], 
                colors=state["colors"], 
                point_size=0.05
            )

        # --- Event Handlers ---

        @slider_frame.on_update
        def _(event):
            state["frame_idx"] = event.target.value
            load_frame(event.target.value)

        # 【新增】监听 Show All 开关
        @chk_show_all.on_update
        def _(event):
            state["show_all_frames"] = event.target.value
            # Reload frame logic (which decides whether to show single or all)
            load_frame(state["frame_idx"])

        @chk_edit.on_update
        def _(event):
            is_editing = event.target.value
            btn_place.visible = is_editing
            btn_save.visible = is_editing
            slider_x.disabled = not is_editing
            slider_y.disabled = not is_editing
            slider_z.disabled = not is_editing
            update_edit_visuals()

        def on_size_change(_):
            state["bbox_extent"] = np.array([slider_x.value, slider_y.value, slider_z.value])
            if chk_edit.value and handles["bbox_wireframe"]:
                 segments = get_bbox_edge_points(state["bbox_extent"])
                 handles["bbox_wireframe"].points = segments
        
        slider_x.on_update(on_size_change)
        slider_y.on_update(on_size_change)
        slider_z.on_update(on_size_change)

        @btn_place.on_click
        def _(_):
            btn_place.disabled = True
            btn_place.label = "Click Point Cloud to Snap..."
            remove_edit_visuals() 

            @client.scene.on_pointer_event(event_type="click")
            def _(event: viser.ScenePointerEvent):
                client.scene.remove_pointer_callback()
                ray_o, ray_d = np.array(event.ray_origin), np.array(event.ray_direction)
                # Note: this works on state["points"], so if Show All is on, it snaps to global map!
                hit_point = ray_point_intersection(ray_o, ray_d, state["points"])
                if hit_point is not None: state["bbox_pos"] = hit_point
                btn_place.disabled = False
                btn_place.label = "Click to Place (Snap)"
                update_edit_visuals()

        @btn_save.on_click
        def _(_):
            w, x, y, z = state["bbox_rot"]
            quat_xyzw = [x, y, z, w]
            annotation = {
                "group_name": chunk_name,
                "frame_idx": state["frame_idx"],
                "bbox": {
                    "center": state["bbox_pos"].tolist(),
                    "extent": state["bbox_extent"].tolist(),
                    "rotation": Rotation.from_quat(quat_xyzw).as_matrix().tolist(),
                    "quat_xyzw": quat_xyzw
                }
            }
            path = os.path.join(base_dir, f"bbox_annotation_frame_{state['frame_idx']:06d}.json")
            with open(path, 'w') as f: json.dump(annotation, f, indent=4)
            print(f"Saved: {path}")

        @btn_reset_cam.on_click
        def _(_):
            c2w = state["c2w"]
            client.camera.position = c2w[:3, 3]
            client.camera.wxyz = tf.SO3.from_matrix(c2w[:3, :3]).wxyz
            client.camera.look_at = c2w[:3, 3] + (c2w[:3, :3] @ np.array([0,0,1])) * 5.0
            client.camera.up_direction = c2w[:3, :3] @ np.array([0,-1,0])

        load_frame(0)

    while True: time.sleep(1.0)

if __name__ == "__main__":
    main()