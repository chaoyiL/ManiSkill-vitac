import cv2
import numpy as np
import time
from user_client.robot_visualization.src.viz_vb_data import (CombinedVisualizer, ROBOT_IDS, resize_with_label, 
                         hstack_with_sep, create_combined_image)
import trimesh
import pyrender
from scipy.spatial.transform import Rotation
import os

class Enhanced3DVisualizer(CombinedVisualizer):
    """3D增强可视化器"""
    
    def __init__(self, *args, **kwargs):
        self.window_name = "Robot Monitor"
        self.playback_speed = 1.0
        self.auto_next_episode = True  # 自动播放下一个Episode
        self._world_scene = None
        self._world_dynamic_nodes = []
        self._world_camera_pose = None
        self._cached_axis_mesh = None
        self._cached_wrist_mesh = None
        self._cached_base_mesh = None
        self._cached_sensor_meshes = None
        self._cached_controller_meshes = None
        self._cached_gripper_mesh_left = None
        self._cached_gripper_mesh_right = None
        super().__init__(*args, **kwargs)

    def setup_renderers(self):
        super().setup_renderers()
        self._init_world_scene_cache()

    def _init_world_scene_cache(self):
        """初始化并缓存世界场景、相机和静态网格"""
        from user_client.robot_visualization.src.viz_vb_data import (_axis_mesh, _lookat_camera_pose)

        scene = pyrender.Scene(bg_color=[0.05, 0.08, 0.12, 1.0])

        floor_grid = self._create_floor_grid_solid(size=1.5, step=0.2)
        if floor_grid:
            scene.add(floor_grid)

        coord_axes = self._create_coordinate_axes(length=0.2)
        for axis_mesh in coord_axes:
            scene.add(axis_mesh)

        self._cached_axis_mesh = _axis_mesh(size=0.05)

        wrist = trimesh.creation.cylinder(radius=0.02, height=0.03, sections=16)
        wrist.visual.vertex_colors = np.array([180, 180, 180, 255], dtype=np.uint8)
        self._cached_wrist_mesh = pyrender.Mesh.from_trimesh(wrist, smooth=True)

        base = trimesh.creation.box(extents=[0.08, 0.04, 0.02])
        base.visual.vertex_colors = np.array([200, 200, 200, 255], dtype=np.uint8)
        self._cached_base_mesh = pyrender.Mesh.from_trimesh(base, smooth=False)

        self._cached_sensor_meshes = {
            'left': self._create_sensor_mesh([0, 255, 0]),
            'right': self._create_sensor_mesh([255, 0, 0])
        }

        self._cached_controller_meshes = {
            'left': self._create_realistic_quest_controller(is_left=True),
            'right': self._create_realistic_quest_controller(is_left=False)
        }

        gripper_base_mesh = self._load_real_gripper_stl()
        if gripper_base_mesh is not None:
            left_gripper = gripper_base_mesh.copy()
            left_gripper.visual.vertex_colors = np.array([180, 180, 180, 255], dtype=np.uint8)
            self._cached_gripper_mesh_left = pyrender.Mesh.from_trimesh(left_gripper, smooth=True)

            right_gripper = gripper_base_mesh.copy()
            mirror = np.eye(4)
            mirror[1, 1] = -1
            right_gripper.apply_transform(mirror)
            right_gripper.visual.vertex_colors = np.array([180, 180, 180, 255], dtype=np.uint8)
            self._cached_gripper_mesh_right = pyrender.Mesh.from_trimesh(right_gripper, smooth=True)

        cam_pose = _lookat_camera_pose([0.05, -0.3, 0.4], [0, 0, 0], [0, 0, 1])
        self._world_camera_pose = cam_pose
        camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(60.0))
        scene.add(camera, pose=cam_pose)

        main_light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
        scene.add(main_light, pose=cam_pose)

        fill_pose = cam_pose.copy()
        fill_pose[:3, 3] = -cam_pose[:3, 3]
        fill_light = pyrender.DirectionalLight(color=[0.8, 0.9, 1.0], intensity=2.0)
        scene.add(fill_light, pose=fill_pose)

        self._world_scene = scene
        self._world_dynamic_nodes = []

    def _create_sensor_mesh(self, color_rgb):
        sensor = trimesh.creation.cylinder(radius=0.012, height=0.003, sections=16)
        sensor.visual.vertex_colors = np.array(color_rgb + [255], dtype=np.uint8)
        return pyrender.Mesh.from_trimesh(sensor, smooth=True)
    
    def _create_realistic_quest_controller(self, is_left=True):
        """创建Quest手柄"""
        mesh_file = 'src/meshes/Oculus_Meta_Quest_Touch_Plus_Controller_Left.stl' if is_left else 'src/meshes/Oculus_Meta_Quest_Touch_Plus_Controller_Right.stl'
        
        if not os.path.exists(mesh_file):
            cyl = trimesh.creation.cylinder(radius=0.015, height=0.08, sections=16)
            rgba = np.array([0.4, 0.4, 0.4, 1.0])
            cyl.visual.vertex_colors = (rgba * 255).astype(np.uint8)
            return pyrender.Mesh.from_trimesh(cyl, smooth=False)
        
        mesh = trimesh.load(mesh_file)
        mesh.apply_scale(0.0015)
        rgba = np.array([0.4, 0.4, 0.4, 1.0])
        mesh.visual.vertex_colors = (rgba * 255).astype(np.uint8)
        return pyrender.Mesh.from_trimesh(mesh, smooth=True)
    
    def _load_real_gripper_stl(self):
        """加载真实的夹爪STL文件"""
        filepath = 'src/meshes/夹爪.STL'
        if not os.path.exists(filepath):
            return None
        try:
            mesh = trimesh.load(filepath)
            mesh.apply_scale(0.001)
            center = (mesh.bounds[0] + mesh.bounds[1]) / 2
            mesh.apply_translation(-center)
            rot_matrix = Rotation.from_euler('y', 180, degrees=True).as_matrix()
            transform = np.eye(4)
            transform[:3, :3] = rot_matrix
            mesh.apply_transform(transform)
            mesh.visual.vertex_colors = np.array([180, 180, 180, 255], dtype=np.uint8)
            return mesh
        except:
            return None
    
    def _create_floor_grid_solid(self, size=2.0, step=0.2):
        """创建地面网格"""
        meshes = []
        radius = 0.002
        
        for y in np.arange(-size, size + step, step):
            line = trimesh.creation.cylinder(radius=radius, height=size*2, sections=8)
            line.visual.vertex_colors = np.array([100, 120, 140, 255], dtype=np.uint8)
            rot_tf = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
            line.apply_transform(rot_tf)
            line.apply_translation([0, y, -0.5])
            meshes.append(line)
        
        for x in np.arange(-size, size + step, step):
            line = trimesh.creation.cylinder(radius=radius, height=size*2, sections=8)
            line.visual.vertex_colors = np.array([100, 120, 140, 255], dtype=np.uint8)
            rot_tf = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
            line.apply_transform(rot_tf)
            line.apply_translation([x, 0, -0.5])
            meshes.append(line)
        
        if meshes:
            combined = trimesh.util.concatenate(meshes)
            return pyrender.Mesh.from_trimesh(combined, smooth=False)
        return None
    
    def _create_coordinate_axes(self, length=0.2):
        """创建RGB坐标轴"""
        meshes = []
        radius = 0.004
        
        # X轴（红色）
        x_cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
        x_cyl.visual.vertex_colors = np.array([255, 0, 0, 255], dtype=np.uint8)
        x_tf = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
        x_tf[:3, 3] = [length/2, 0, 0]
        x_cyl.apply_transform(x_tf)
        meshes.append(pyrender.Mesh.from_trimesh(x_cyl, smooth=True))
        
        x_cone = trimesh.creation.cone(radius=radius*2, height=0.02, sections=12)
        x_cone.visual.vertex_colors = np.array([255, 0, 0, 255], dtype=np.uint8)
        x_cone_tf = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
        x_cone_tf[:3, 3] = [length + 0.01, 0, 0]
        x_cone.apply_transform(x_cone_tf)
        meshes.append(pyrender.Mesh.from_trimesh(x_cone, smooth=True))
        
        # Y轴（绿色）
        y_cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
        y_cyl.visual.vertex_colors = np.array([0, 255, 0, 255], dtype=np.uint8)
        y_tf = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        y_tf[:3, 3] = [0, length/2, 0]
        y_cyl.apply_transform(y_tf)
        meshes.append(pyrender.Mesh.from_trimesh(y_cyl, smooth=True))
        
        y_cone = trimesh.creation.cone(radius=radius*2, height=0.02, sections=12)
        y_cone.visual.vertex_colors = np.array([0, 255, 0, 255], dtype=np.uint8)
        y_cone_tf = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        y_cone_tf[:3, 3] = [0, length + 0.01, 0]
        y_cone.apply_transform(y_cone_tf)
        meshes.append(pyrender.Mesh.from_trimesh(y_cone, smooth=True))
        
        # Z轴（蓝色）
        z_cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
        z_cyl.visual.vertex_colors = np.array([0, 0, 255, 255], dtype=np.uint8)
        z_cyl.apply_translation([0, 0, length/2])
        meshes.append(pyrender.Mesh.from_trimesh(z_cyl, smooth=True))
        
        z_cone = trimesh.creation.cone(radius=radius*2, height=0.02, sections=12)
        z_cone.visual.vertex_colors = np.array([0, 0, 255, 255], dtype=np.uint8)
        z_cone.apply_translation([0, 0, length + 0.01])
        meshes.append(pyrender.Mesh.from_trimesh(z_cone, smooth=True))
        
        return meshes
    
    def render_world_scene(self, current_idx):
        """渲染世界场景"""
        from user_client.robot_visualization.src.viz_vb_data import (_line_mesh, _pointcloud_mesh, RenderFlags)

        if self._world_scene is None:
            self._init_world_scene_cache()

        scene = self._world_scene

        for node in self._world_dynamic_nodes:
            scene.remove_node(node)
        self._world_dynamic_nodes = []

        colors = {0: [1, 0, 0], 1: [0, 1, 0]}
        for r in ROBOT_IDS:
            prefix = f'robot{r}'
            poses = self.data[prefix].get('poses', [])
            if not poses or current_idx >= len(poses):
                continue

            pts = np.array([p[:3, 3] for p in poses[:current_idx + 1]], dtype=np.float64)

            line_mesh = _line_mesh(pts, color=colors[r], radius=0.012)
            if line_mesh:
                self._world_dynamic_nodes.append(scene.add(line_mesh))

            pts_mesh = _pointcloud_mesh(pts, color=colors[r])
            if pts_mesh:
                self._world_dynamic_nodes.append(scene.add(pts_mesh))

            frame_pose = poses[current_idx]
            combined_rotation = Rotation.from_euler('xz', [180, 90], degrees=True).as_matrix()
            flip_tf = np.eye(4)
            flip_tf[:3, :3] = combined_rotation
            flipped_pose = frame_pose @ flip_tf
            
            if self._cached_axis_mesh is not None:
                self._world_dynamic_nodes.append(scene.add(self._cached_axis_mesh, pose=flipped_pose))
            
            ctrl = self._cached_controller_meshes['left'] if r == 1 else self._cached_controller_meshes['right']
            if ctrl:
                ctrl_tf = np.eye(4)
                rot = Rotation.from_euler('y', 90, degrees=True).as_matrix()
                ctrl_tf[:3, :3] = rot
                ctrl_tf[:3, 3] = [0, 0, 0.05]
                self._world_dynamic_nodes.append(scene.add(ctrl, pose=flipped_pose @ ctrl_tf))
            
            wrist_tf = np.eye(4)
            wrist_tf[:3, 3] = [0, 0, 0.01]
            if self._cached_wrist_mesh is not None:
                self._world_dynamic_nodes.append(scene.add(self._cached_wrist_mesh, pose=flipped_pose @ wrist_tf))
            
            base_tf = np.eye(4)
            base_tf[:3, 3] = [0, 0, -0.02]
            if self._cached_base_mesh is not None:
                self._world_dynamic_nodes.append(scene.add(self._cached_base_mesh, pose=flipped_pose @ base_tf))
            
            gripper = self.data[prefix].get('gripper', [])
            if gripper and current_idx < len(gripper):
                grip_width = float(gripper[current_idx])
                offset = max(grip_width * 0.5, 0.03)
                
                if self._cached_gripper_mesh_left is not None and self._cached_gripper_mesh_right is not None:
                    left_tf = np.eye(4)
                    left_tf[:3, 3] = [0.02, -offset, -0.04]
                    self._world_dynamic_nodes.append(scene.add(self._cached_gripper_mesh_left, pose=flipped_pose @ left_tf))
                    
                    right_tf = np.eye(4)
                    right_tf[:3, 3] = [0.02, offset, -0.04]
                    self._world_dynamic_nodes.append(scene.add(self._cached_gripper_mesh_right, pose=flipped_pose @ right_tf))
                
                for side, sign, color_rgb in [('left', -1, [0, 255, 0]), ('right', 1, [255, 0, 0])]:
                    sensor_mesh = self._cached_sensor_meshes[side] if self._cached_sensor_meshes else None
                    sensor_tf = np.eye(4)
                    sensor_tf[:3, :3] = Rotation.from_euler('x', 90, degrees=True).as_matrix()
                    sensor_tf[:3, 3] = [0.05, sign * (offset - 0.01), -0.04]
                    if sensor_mesh is not None:
                        self._world_dynamic_nodes.append(scene.add(sensor_mesh, pose=flipped_pose @ sensor_tf))

        color_img, _ = self.renderers['world'].render(scene, flags=RenderFlags.RGBA)
        return color_img[:, :, :3]
    
    def _create_trajectory_plots(self, frame_idx):
        """创建轨迹曲线图面板 - 虚线预览+实线高亮"""
        w, h = 450, 750
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        panel[:] = [25, 30, 40]
        
        cv2.rectangle(panel, (0, 0), (w, 40), (15, 20, 30), -1)
        cv2.putText(panel, "Real-time Trajectories", (15, 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        
        y_offset = 50
        plot_h = 110
        plot_w = w - 50
        plot_x_start = 40
        
        max_frames = len(self.data['robot0']['poses'])
        
        plots = [
            ("Robot 0 Position (m)", ['robot0'], ['X', 'Y', 'Z'], [(255, 10, 10), (10, 255, 10), (10, 10, 255)]),
            ("Robot 1 Position (m)", ['robot1'], ['X', 'Y', 'Z'], [(255, 10, 10), (10, 255, 10), (10, 10, 255)]),
            ("Gripper Width (m)", ['robot0', 'robot1'], ['R0', 'R1'], [(255, 100, 100), (100, 255, 100)])
        ]
        
        for plot_name, robots, labels, colors in plots:
            cv2.putText(panel, plot_name, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            y_offset += 20
            
            cv2.rectangle(panel, (plot_x_start, y_offset), (plot_x_start + plot_w, y_offset + plot_h - 20), 
                         (35, 40, 50), -1)
            cv2.rectangle(panel, (plot_x_start, y_offset), (plot_x_start + plot_w, y_offset + plot_h - 20), 
                         (60, 65, 75), 1)
            
            if "Position" in plot_name:
                robot_id = 0 if "Robot 0" in plot_name else 1
                prefix = f'robot{robot_id}'
                poses = self.data[prefix].get('poses', [])
                
                if poses and len(poses) > 0:
                    all_positions = np.array([poses[i][:3, 3] for i in range(len(poses))])
                    
                    for axis_idx, (axis_name, color) in enumerate(zip(labels, colors)):
                        if len(all_positions) > 1:
                            data = all_positions[:, axis_idx]
                            data_min, data_max = data.min(), data.max()
                            data_range = data_max - data_min if data_max > data_min else 1.0
                            
                            # 虚线（整条轨迹）
                            for i in range(1, len(data)):
                                x1 = int(plot_x_start + (i - 1) / max_frames * plot_w)
                                y1 = int(y_offset + (plot_h - 20) - ((data[i-1] - data_min) / data_range) * (plot_h - 30))
                                x2 = int(plot_x_start + i / max_frames * plot_w)
                                y2 = int(y_offset + (plot_h - 20) - ((data[i] - data_min) / data_range) * (plot_h - 30))
                                dark_color = tuple(int(c * 0.3) for c in color)
                                cv2.line(panel, (x1, y1), (x2, y2), dark_color, 1, cv2.LINE_AA)
                            
                            # 实线（当前进度）
                            for i in range(1, min(frame_idx + 1, len(data))):
                                x1 = int(plot_x_start + (i - 1) / max_frames * plot_w)
                                y1 = int(y_offset + (plot_h - 20) - ((data[i-1] - data_min) / data_range) * (plot_h - 30))
                                x2 = int(plot_x_start + i / max_frames * plot_w)
                                y2 = int(y_offset + (plot_h - 20) - ((data[i] - data_min) / data_range) * (plot_h - 30))
                                cv2.line(panel, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                            
                            # Y轴标签（按颜色区分，位置错开）
                            label_x = 5
                            label_y_up = y_offset + 10 + axis_idx * 12
                            label_y_down = y_offset + plot_h - 25 - axis_idx * 12
                            cv2.putText(panel, f"{data_max:.3f}", (label_x, label_y_up), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
                            cv2.putText(panel, f"{data_min:.3f}", (label_x, label_y_down), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
                    
                    # X轴标签
                    cv2.putText(panel, "0", (plot_x_start, y_offset + plot_h - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
                    cv2.putText(panel, f"{max_frames}", (plot_x_start + plot_w - 20, y_offset + plot_h - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
                    
                    # 图例
                    legend_x = plot_x_start + 5
                    for axis_name, color in zip(labels, colors):
                        cv2.circle(panel, (legend_x, y_offset + 10), 4, color, -1)
                        cv2.putText(panel, axis_name, (legend_x + 10, y_offset + 13), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
                        legend_x += 40
            
            elif "Gripper" in plot_name:
                all_gripper_series = []
                for robot_prefix in robots:
                    gripper_data = self.data[robot_prefix].get('gripper', [])
                    if gripper_data and len(gripper_data) > 0:
                        all_gripper_series.append(np.array([gripper_data[i] for i in range(len(gripper_data))]))

                if all_gripper_series:
                    all_grippers_concat = np.concatenate(all_gripper_series)
                    data_min, data_max = all_grippers_concat.min(), all_grippers_concat.max()
                    data_range = data_max - data_min if data_max > data_min else 0.01

                    for robot_idx, (robot_prefix, label, color) in enumerate(zip(robots, labels, colors)):
                        gripper_data = self.data[robot_prefix].get('gripper', [])
                        if gripper_data and len(gripper_data) > 0:
                            all_grippers = np.array([gripper_data[i] for i in range(len(gripper_data))])

                            if len(all_grippers) > 1:
                                # 虚线（全部）
                                for i in range(1, len(all_grippers)):
                                    x1 = int(plot_x_start + (i - 1) / max_frames * plot_w)
                                    y1 = int(y_offset + (plot_h - 20) - ((all_grippers[i-1] - data_min) / data_range) * (plot_h - 30))
                                    x2 = int(plot_x_start + i / max_frames * plot_w)
                                    y2 = int(y_offset + (plot_h - 20) - ((all_grippers[i] - data_min) / data_range) * (plot_h - 30))
                                    dark_color = tuple(int(c * 0.3) for c in color)
                                    cv2.line(panel, (x1, y1), (x2, y2), dark_color, 1, cv2.LINE_AA)

                                # 实线（当前）
                                for i in range(1, min(frame_idx + 1, len(all_grippers))):
                                    x1 = int(plot_x_start + (i - 1) / max_frames * plot_w)
                                    y1 = int(y_offset + (plot_h - 20) - ((all_grippers[i-1] - data_min) / data_range) * (plot_h - 30))
                                    x2 = int(plot_x_start + i / max_frames * plot_w)
                                    y2 = int(y_offset + (plot_h - 20) - ((all_grippers[i] - data_min) / data_range) * (plot_h - 30))
                                    cv2.line(panel, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                    # Y轴标签（统一范围）
                    cv2.putText(panel, f"{data_max:.3f}", (5, y_offset + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1, cv2.LINE_AA)
                    cv2.putText(panel, f"{data_min:.3f}", (5, y_offset + plot_h - 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1, cv2.LINE_AA)
                
                # X轴标签
                cv2.putText(panel, "0", (plot_x_start, y_offset + plot_h - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
                cv2.putText(panel, f"{max_frames}", (plot_x_start + plot_w - 20, y_offset + plot_h - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
                
                # 图例
                legend_x = plot_x_start + 5
                for label, color in zip(labels, colors):
                    cv2.circle(panel, (legend_x, y_offset + 10), 4, color, -1)
                    cv2.putText(panel, label, (legend_x + 10, y_offset + 13), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
                    legend_x += 50
            
            y_offset += plot_h + 15
        
        return panel
    
    def render_frame(self, frame_idx):
        """渲染完整帧"""
        self.frame_idx = frame_idx
        world_image = self.render_world_scene(frame_idx)
        return self._create_complete_layout(frame_idx, {}, world_image)
    
    def _enhance_3d_image(self, img):
        """增强3D图像效果"""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        return np.clip(cv2.addWeighted(img, 0.7, sharpened, 0.3, 0) * 1.15, 0, 255).astype(np.uint8)
    
    def _create_complete_layout(self, frame_idx, pc_images, world_image):
        """创建完整布局"""
        header = self._create_header(frame_idx)
        camera_row = self._create_camera_row(frame_idx)
        world_panel = self._add_panel_frame(world_image if world_image is not None else np.zeros((700, 900, 3), dtype=np.uint8), "3D World View", (100, 200, 255))
        plots_panel = self._create_trajectory_plots(frame_idx)
        
        h = max(world_panel.shape[0], plots_panel.shape[0])
        sep = np.zeros((h, 5, 3), dtype=np.uint8)
        sep[:] = [15, 20, 30]
        bottom_row = np.hstack([self._pad_height(world_panel, h), sep, self._pad_height(plots_panel, h)])
        control_bar = self._create_control_bar(frame_idx)
        max_w = max(header.shape[1], camera_row.shape[1], bottom_row.shape[1], control_bar.shape[1])
        sep_h = np.zeros((3, max_w, 3), dtype=np.uint8)
        sep_h[:] = [15, 20, 30]
        return np.vstack([self._pad_width(header, max_w), sep_h, self._pad_width(camera_row, max_w), sep_h, self._pad_width(bottom_row, max_w), sep_h, self._pad_width(control_bar, max_w)])
    
    def _create_header(self, frame_idx):
        """创建顶部标题栏"""
        w, h = 1600, 60
        header = np.zeros((h, w, 3), dtype=np.uint8)
        header[:] = [20, 25, 35]
        
        cv2.putText(header, "Robot Monitor", (20, 38), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.circle(header, (250, 30), 6, (100, 255, 100), -1)
        cv2.putText(header, f"Speed: {self.playback_speed}x", (270, 38), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
        
        ep_id, max_frames = self.episodes[self.ep_idx], len(self.data['robot0']['poses'])
        cv2.putText(header, f"Ep {ep_id}/{len(self.episodes)} | Frame {frame_idx}/{max_frames-1}", 
                   (w - 400, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return header
    
    def _create_camera_row(self, frame_idx):
        """创建相机视图行"""
        height, all_images = 250, []
        for r in ROBOT_IDS:
            prefix, color = f'robot{r}', ((100, 100, 255) if r == 0 else (100, 255, 100))
            robot_images = [resize_with_label(self.data[prefix][s][frame_idx].copy(), f"R{r} {l}", height, color) 
                           for s, l in [('visual', 'Visual'), ('left_tactile', 'L-Tact'), ('right_tactile', 'R-Tact')]
                           if s in self.data[prefix] and frame_idx < len(self.data[prefix][s])]
            if robot_images:
                all_images.append(self._add_robot_label(hstack_with_sep(robot_images, 2, 40), f"Robot {r}", color))
        return hstack_with_sep(all_images, 5, 15) if all_images else np.zeros((height, 1600, 3), dtype=np.uint8)
    
    def _add_robot_label(self, img, label, color):
        """添加机器人标签"""
        header_h = 35
        panel = np.zeros((img.shape[0] + header_h, img.shape[1], 3), dtype=np.uint8)
        panel[:] = [25, 30, 40]
        cv2.rectangle(panel, (0, 0), (panel.shape[1], header_h), (15, 20, 30), -1)
        cv2.circle(panel, (15, header_h//2), 5, color, -1)
        cv2.putText(panel, label, (30, header_h//2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        panel[header_h:, :] = img
        return panel
    
    def _create_control_bar(self, frame_idx):
        """创建控制栏"""
        w, h = 1600, 80
        bar = np.zeros((h, w, 3), dtype=np.uint8)
        bar[:] = [20, 25, 35]
        max_frames = len(self.data['robot0']['poses'])
        bar_x1, bar_x2, bar_y = 200, w - 450, 30
        
        cv2.rectangle(bar, (bar_x1, bar_y-4), (bar_x2, bar_y+4), (50, 55, 65), -1)
        if max_frames > 1:
            progress = int((frame_idx / (max_frames-1)) * (bar_x2 - bar_x1))
            cv2.rectangle(bar, (bar_x1, bar_y-4), (bar_x1+progress, bar_y+4), (59, 130, 246), -1)
            cv2.circle(bar, (bar_x1+progress, bar_y), 9, (59, 130, 246), -1)
            cv2.circle(bar, (bar_x1+progress, bar_y), 10, (255, 255, 255), 2)
        
        cv2.putText(bar, f"{frame_idx}/{max_frames-1}", (bar_x1, bar_y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        
        # 倍速按钮
        speed_x = bar_x2 + 20
        cv2.putText(bar, "Speed:", (speed_x, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        
        speeds = [0.25, 0.5, 1.0, 2.0, 5.0]
        btn_w, btn_h = 50, 25
        btn_y = 35
        
        for i, speed in enumerate(speeds):
            btn_x = speed_x + i * (btn_w + 5)
            
            if abs(self.playback_speed - speed) < 0.01:
                btn_color = (59, 130, 246)
            else:
                btn_color = (60, 65, 75)
            
            cv2.rectangle(bar, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), 
                         btn_color, -1)
            cv2.rectangle(bar, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), 
                         (80, 85, 95), 1)
            
            label = f"{speed}x" if speed >= 1 else f"{speed}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            text_x = btn_x + (btn_w - text_size[0]) // 2
            cv2.putText(bar, label, (text_x, btn_y + 17), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
        
        controls = "[A/D]Frame [W/S]Ep [P]Play [1-5]Speed [R]Reset [C]Shot [Q]Quit"
        cv2.putText(bar, controls, (20, 68), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        
        return bar
    
    def _add_panel_frame(self, img, title, color):
        """添加面板框架"""
        border, header_h = 5, 40
        panel = np.zeros((img.shape[0]+header_h+border*2, img.shape[1]+border*2, 3), dtype=np.uint8)
        panel[:] = [25, 30, 40]
        cv2.rectangle(panel, (0, 0), (panel.shape[1], header_h), (15, 20, 30), -1)
        cv2.circle(panel, (15, header_h//2), 5, color, -1)
        cv2.putText(panel, title, (30, header_h//2+5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        panel[header_h+border:header_h+border+img.shape[0], border:border+img.shape[1]] = img
        return panel
    
    def _pad_width(self, img, target_w):
        """填充宽度"""
        if img.shape[1] >= target_w: return img
        pad = np.zeros((img.shape[0], target_w - img.shape[1], 3), dtype=np.uint8)
        pad[:] = [20, 25, 35]
        return np.hstack([img, pad])
    
    def _pad_height(self, img, target_h):
        """填充高度"""
        if img.shape[0] >= target_h: return img
        pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
        pad[:] = [25, 30, 40]
        return np.vstack([img, pad])
    
    def run(self):
        """运行可视化循环"""
        auto_play = False
        frame_counter = 0
        
        print("\n" + "=" * 70)
        print("  Robot Monitor")
        print("=" * 70)
        print("  Controls:")
        print("    [A/D]    前后帧")
        print("    [W]      下一个Episode")
        print("    [S]      上一个Episode")
        print("    [P]      自动播放（Episode结束后自动跳转下一个）")
        print("    [1-5]    速度: 0.25x, 0.5x, 1x, 2x, 5x")
        print("    [R]      重置视角")
        print("    [C]      截图")
        print("    [Q]      退出")
        print("=" * 70 + "\n")
        
        while True:
            frame = self.render_frame(self.frame_idx)
            if frame is not None:
                if auto_play:
                    cv2.circle(frame, (frame.shape[1] - 50, 35), 8, (100, 255, 100), -1)
                cv2.imshow(self.window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # 自动播放逻辑
            if auto_play:
                frame_counter += self.playback_speed
                if frame_counter >= 1.0:
                    steps = int(frame_counter)
                    frame_counter -= steps
                    
                    max_f = len(self.data['robot0']['poses'])
                    
                    if self.frame_idx + steps < max_f:
                        # 正常前进
                        self.frame_idx += steps
                    else:
                        # 到达当前Episode末尾
                        if self.auto_next_episode and self.ep_idx < len(self.episodes) - 1:
                            # 自动跳转下一个Episode
                            print(f"Episode {self.episodes[self.ep_idx]} 完成，自动加载下一个...")
                            self.ep_idx += 1
                            self.load_episode()
                            frame_counter = 0
                        else:
                            # 最后一个Episode或关闭自动跳转
                            self.frame_idx = max_f - 1
                            auto_play = False
                            print("播放完成")
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('d'):
                # D键：下一帧，如果到末尾自动下一个Episode
                max_f = len(self.data['robot0']['poses'])
                if self.frame_idx < max_f - 1:
                    self.frame_idx += 1
                elif self.ep_idx < len(self.episodes) - 1:
                    self.ep_idx += 1
                    self.load_episode()
                    print(f"跳转到 Episode {self.episodes[self.ep_idx]}")
                    
            elif key == ord('a'):
                # A键：上一帧
                if self.frame_idx > 0:
                    self.frame_idx -= 1
                    
            elif key == ord('w'):
                # W键：下一个Episode（修正）
                if self.ep_idx < len(self.episodes) - 1:
                    self.ep_idx += 1
                    self.load_episode()
                    print(f"跳转到 Episode {self.episodes[self.ep_idx]}")
                else:
                    print("已经是最后一个Episode")
                    
            elif key == ord('s'):
                # S键：上一个Episode（修正）
                if self.ep_idx > 0:
                    self.ep_idx -= 1
                    self.load_episode()
                    print(f"跳转到 Episode {self.episodes[self.ep_idx]}")
                else:
                    print("已经是第一个Episode")
                    
            elif key == ord('p'): 
                auto_play = not auto_play
                frame_counter = 0
                print(f"自动播放: {'开启' if auto_play else '关闭'}")
                
            elif key == ord('1'): self.playback_speed = 0.25; print("速度: 0.25x")
            elif key == ord('2'): self.playback_speed = 0.5; print("速度: 0.5x")
            elif key == ord('3'): self.playback_speed = 1.0; print("速度: 1x")
            elif key == ord('4'): self.playback_speed = 2.0; print("速度: 2x")
            elif key == ord('5'): self.playback_speed = 5.0; print("速度: 5x")
            elif key == ord('r'): self.setup_camera_params(); print("视角重置")
            elif key == ord('c'):
                import datetime
                filename = f"monitor_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"截图: {filename}")
            elif key == ord('q'): 
                print("\n退出\n")
                break
        
        cv2.destroyAllWindows()

    def record_episode(self):
        """录制当前episode（计时）"""
        start_time = time.perf_counter()
        super().record_episode()
        elapsed = time.perf_counter() - start_time
        print(f" 录制耗时 {elapsed:.2f}s")

def main():
    import argparse
    from zarr.storage import ZipStore
    import zarr
    from user_client.robot_visualization.src.replay_buffer import ReplayBuffer
    
    parser = argparse.ArgumentParser()
    parser.add_argument('zarr_path', nargs='?', default='data/_0115_bi_pick_and_place_2ver.zarr.zip')
    parser.add_argument('--record', '-r', action='store_true')
    parser.add_argument('--record_episode', '-e', type=int, default=1)
    parser.add_argument('--output_video', '-o', type=str, default=None)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--continue_after_record', "-c", action='store_true')
    args = parser.parse_args()
    
    if not os.path.exists(args.zarr_path): 
        return
    
    store = ZipStore(args.zarr_path, mode='r')
    try:
        rb = ReplayBuffer.create_from_group(zarr.open_group(store=store, mode='r'))
        print(f"加载: {rb.n_steps:,} 帧, {rb.n_episodes} episodes\n")
        Enhanced3DVisualizer(rb, np.arange(rb.n_episodes), args.record, args.record_episode, args.output_video, args.fps, args.continue_after_record)
    finally:
        store.close()

if __name__ == "__main__":
    main()
