import numpy as np
from scipy.spatial.transform import Rotation
import sys
import os
import cv2
import zarr
import trimesh
import pyrender
from pyrender import RenderFlags
from zarr.storage import ZipStore

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from user_client.robot_visualization.src.replay_buffer import ReplayBuffer
from user_client.robot_visualization.src.imagecodecs_numcodecs import register_codecs
register_codecs()

# 传感器配置
ROBOT_IDS = [0, 1]
SENSOR_KEY_CANDIDATES = {
    'visual': [
        'robot{}_visual',
        'camera{}_rgb',
    ],
    'left_tactile': [
        'robot{}_left_tactile',
        'camera{}_left_tactile',
    ],
    'right_tactile': [
        'robot{}_right_tactile',
        'camera{}_right_tactile',
    ],
    'left_pc': [
        'robot{}_left_tactile_points',
        'camera{}_left_tactile_points',
    ],
    'right_pc': [
        'robot{}_right_tactile_points',
        'camera{}_right_tactile_points',
    ],
}
def get_transform(pos, rot_axis_angle):
    """获取变换矩阵"""
    rotation_matrix, _ = cv2.Rodrigues(rot_axis_angle)
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = pos
    return T

def decode_image(img_data):
    """统一的图像解码"""
    if isinstance(img_data, bytes):
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_data

def load_pointcloud(points_data):
    """统一的点云加载和过滤"""
    if points_data is None or len(points_data) == 0:
        return np.empty((0, 3), dtype=np.float32)
    points = np.array(points_data, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float32)
    mask = np.any(points != 0, axis=1)
    return points[mask] if np.any(mask) else np.empty((0, 3), dtype=np.float32)

def calc_camera_params(points):
    """计算点云的相机参数"""
    if len(points) == 0:
        return {'center': [0, 0, 40], 'eye': [50, -30, 80], 'up': [0, 0, 1]}
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    extent = pts.max(axis=0) - pts.min(axis=0)
    dist = max(np.max(extent) * 2.5, 1.0)
    eye = center + np.array([dist * 0.2, -dist * 0.1, dist * 1.2])
    return {'center': center.tolist(), 'eye': eye.tolist(), 'up': [0, 0, 1]}

def resize_with_label(img, label, height, color=(255, 255, 255)):
    """调整图像大小并添加标签"""
    if img is None:
        return None
    h, w = img.shape[:2]
    resized = cv2.resize(img, (int(w * height / h), height))
    cv2.putText(resized, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(resized, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return resized

def hstack_with_sep(images, sep_width=2, sep_color=255):
    """水平拼接图像，带分隔线"""
    valid = [img for img in images if img is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    h = valid[0].shape[0]
    sep = np.ones((h, sep_width, 3), dtype=np.uint8) * sep_color
    parts = []
    for i, img in enumerate(valid):
        parts.append(img)
        if i < len(valid) - 1:
            parts.append(sep)
    return np.hstack(parts)

def load_episode_data(replay_buffer, episode_idx):
    """加载episode数据"""
    ep_slice = replay_buffer.get_episode_slice(episode_idx)
    data = {f'robot{r}': {
        'poses': [], 'gripper': [],
        'visual': [], 'left_tactile': [], 'right_tactile': [],
        'left_pc': [], 'right_pc': []
    } for r in ROBOT_IDS}
    
    first_tx = {0: None, 1: None}
    keys = set(replay_buffer.keys())

    def _resolve_key(sensor, robot_id):
        for template in SENSOR_KEY_CANDIDATES[sensor]:
            key = template.format(robot_id)
            if key in keys:
                return key
        return None

    resolved_keys = {
        f'robot{r}_{s}': _resolve_key(s, r)
        for r in ROBOT_IDS for s in SENSOR_KEY_CANDIDATES
    }

    # 检查可用数据
    has = {name: (resolved_keys[name] is not None) for name in resolved_keys}
    has['robot1_pose'] = 'robot1_eef_pos' in keys
    has['robot0_gripper'] = 'robot0_gripper_width' in keys
    has['robot1_gripper'] = 'robot1_gripper_width' in keys
    
    for i in range(ep_slice.start, ep_slice.stop):
        for r in ROBOT_IDS:
            prefix = f'robot{r}'
            cam_id = r
            
            # 加载位姿
            if r == 0 or has['robot1_pose']:
                # 直接使用数据（已经是右手坐标系）
                pos = replay_buffer[f'robot{r}_eef_pos'][i]
                rot = replay_buffer[f'robot{r}_eef_rot_axis_angle'][i]
                
                # 生成变换矩阵
                tx = get_transform(pos, rot)
                data[prefix]['poses'].append(tx)
            
            # 加载夹爪
            if has[f'{prefix}_gripper']:
                data[prefix]['gripper'].append(replay_buffer[f'{prefix}_gripper_width'][i][0])
            
            # 加载图像和点云
            for sensor in SENSOR_KEY_CANDIDATES:
                key = resolved_keys.get(f'{prefix}_{sensor}')
                if not key:
                    continue
                raw = replay_buffer[key][i]
                if 'pc' in sensor:
                    data[prefix][sensor].append(load_pointcloud(raw))
                else:
                    data[prefix][sensor].append(decode_image(raw))
    
    return data, has

def create_combined_image(data, frame_idx, pc_images, world_image, height=250):
    """创建组合图像"""
    rows = []
    colors = {0: (0, 0, 255), 1: (0, 255, 0)}  # robot0红色, robot1绿色

    if world_image is not None:
        world_row = resize_with_label(world_image, "World 3D", height * 2, (255, 255, 255))
        rows.append(world_row)
    
    for r in ROBOT_IDS:
        prefix = f'robot{r}'
        row_parts = []
        
        # 将所有图像放在一起水平拼接
        all_imgs = []
        
        # 相机图像
        for sensor, label in [('visual', 'Visual'), ('left_tactile', 'L-Tact'), ('right_tactile', 'R-Tact')]:
            imgs = data[prefix].get(sensor, [])
            if imgs and frame_idx < len(imgs):
                all_imgs.append(resize_with_label(imgs[frame_idx].copy(), f"R{r} {label}", height, colors[r]))
        
        if all_imgs:
            row_parts.append(hstack_with_sep(all_imgs))
        
        
        
        if row_parts:
            rows.append(hstack_with_sep(row_parts, sep_width=5, sep_color=128))
    
    if not rows:
        return None
    
    # 对齐宽度
    max_w = max(r.shape[1] for r in rows)
    aligned = []
    for row in rows:
        if row.shape[1] < max_w:
            pad = np.zeros((row.shape[0], max_w - row.shape[1], 3), dtype=np.uint8)
            row = np.hstack([row, pad])
        aligned.append(row)
    
    sep = np.ones((5, max_w, 3), dtype=np.uint8) * 128
    if len(aligned) == 1:
        return aligned[0]
    stacked = []
    for i, row in enumerate(aligned):
        stacked.append(row)
        if i < len(aligned) - 1:
            stacked.append(sep)
    return np.vstack(stacked)

def _lookat_camera_pose(eye, center, up):
    eye = np.array(eye, dtype=np.float64)
    center = np.array(center, dtype=np.float64)
    up = np.array(up, dtype=np.float64)
    f = center - eye
    f = f / (np.linalg.norm(f) + 1e-9)
    u = up / (np.linalg.norm(up) + 1e-9)
    s = np.cross(f, u)
    s = s / (np.linalg.norm(s) + 1e-9)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ eye
    return np.linalg.inv(m)

def _quest_controller_mesh(is_left=True):
    """加载Quest手柄STL模型"""
    import os
    mesh_file = 'src/meshes/Oculus_Meta_Quest_Touch_Plus_Controller_Left.stl' if is_left else 'src/meshes/Oculus_Meta_Quest_Touch_Plus_Controller_Right.stl'
    
    if not os.path.exists(mesh_file):
        # 如果文件不存在，返回简单圆柱体
        cyl = trimesh.creation.cylinder(radius=0.02, height=0.15, sections=16)
        rgba = np.array([0.3, 0.3, 0.8, 1.0])
        cyl.visual.vertex_colors = (rgba * 255).astype(np.uint8)
        return pyrender.Mesh.from_trimesh(cyl, smooth=False)
    
    # 加载STL
    mesh = trimesh.load(mesh_file)
    
    # 调整大小（Quest手柄实际尺寸较大，可能需要缩放）
    scale = 0.001  # STL单位可能是mm，转为m
    mesh.apply_scale(scale)
    
    # 设置颜色
    rgba = np.array([0.3, 0.3, 0.8, 1.0])
    mesh.visual.vertex_colors = (rgba * 255).astype(np.uint8)
    
    return pyrender.Mesh.from_trimesh(mesh, smooth=True)

def _axis_mesh(size=0.05):
    axis = trimesh.creation.axis(origin_size=size * 0.2, axis_length=size)
    return pyrender.Mesh.from_trimesh(axis, smooth=False)

def _pointcloud_mesh(points, color=[1.0, 1.0, 0.0]):
    if len(points) == 0:
        return None
    pts = np.asarray(points, dtype=np.float64)
    colors = np.tile(np.array(color, dtype=np.float64), (pts.shape[0], 1))
    return pyrender.Mesh.from_points(pts, colors=colors)

def _cylinder_between(p0, p1, radius, color):
    v = p1 - p0
    h = np.linalg.norm(v)
    if h < 1e-6:
        return None
    cyl = trimesh.creation.cylinder(radius=radius, height=h, sections=12)
    T = trimesh.geometry.align_vectors([0, 0, 1], v / h)
    cyl.apply_transform(T)
    cyl.apply_translation(p0 + v * 0.5)
    rgba = np.array(list(color) + [1.0])
    cyl.visual.vertex_colors = (rgba * 255).astype(np.uint8)
    return cyl

def _line_mesh(points, color=[1.0, 0.0, 0.0], radius=0.01):
    if len(points) < 2:
        return None
    pts = np.asarray(points, dtype=np.float64)
    meshes = []
    for i in range(len(pts) - 1):
        cyl = _cylinder_between(pts[i], pts[i + 1], radius, color)
        if cyl is not None:
            meshes.append(cyl)
    if not meshes:
        return None
    merged = trimesh.util.concatenate(meshes)
    return pyrender.Mesh.from_trimesh(merged, smooth=False)

def _placeholder_cad_box():
    # TODO: Replace with real CAD model mesh when available.
    box = trimesh.creation.box(extents=[0.03, 0.1, 0.03])
    box.visual.vertex_colors = np.tile(np.array([180, 180, 180, 255], dtype=np.uint8), (len(box.vertices), 1))
    return pyrender.Mesh.from_trimesh(box, smooth=False)

def load_controller_mesh(is_left=True):
    """加载Quest手柄STL"""
    import os
    filename = 'controller_left_simple.stl' if is_left else 'controller_right_simple.stl'
    filepath = os.path.join('src', 'meshes', filename)
    
    if not os.path.exists(filepath):
        return None
    
    mesh = trimesh.load(filepath)
    mesh.apply_scale(0.002)  # mm转m，放大2倍
    mesh.visual.vertex_colors = np.array([100, 100, 200, 255], dtype=np.uint8)
    
    return pyrender.Mesh.from_trimesh(mesh, smooth=True)

def load_gripper_mesh():
    """加载真实夹爪STL模型"""
    import os
    filepath = 'src/meshes/夹爪.STL'
    
    if not os.path.exists(filepath):
        return None
    
    mesh = trimesh.load(filepath)
    mesh.apply_scale(0.002)  # mm转m，放大2倍
    
    # 将模型中心移到原点
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    mesh.apply_translation(-center)
    
    # 设置颜色
    mesh.visual.vertex_colors = np.array([180, 180, 180, 255], dtype=np.uint8)
    
    return pyrender.Mesh.from_trimesh(mesh, smooth=True)



def load_gripper_pair():
    """加载一对对称的夹爪"""
    import os
    filepath = 'src/meshes/夹爪.STL'
    
    if not os.path.exists(filepath):
        return None, None
    
    mesh_base = trimesh.load(filepath)
    mesh_base.apply_scale(0.002)
    center = (mesh_base.bounds[0] + mesh_base.bounds[1]) / 2
    mesh_base.apply_translation(-center)
    
    # 左夹爪
    mesh_left = mesh_base.copy()
    mesh_left.visual.vertex_colors = np.array([180, 180, 180, 255], dtype=np.uint8)
    left_gripper = pyrender.Mesh.from_trimesh(mesh_left, smooth=True)
    
    # 右夹爪（Y轴镜像）
    mesh_right = mesh_base.copy()
    mirror = np.eye(4)
    mirror[1, 1] = -1
    mesh_right.apply_transform(mirror)
    mesh_right.visual.vertex_colors = np.array([180, 180, 180, 255], dtype=np.uint8)
    right_gripper = pyrender.Mesh.from_trimesh(mesh_right, smooth=True)
    
    return left_gripper, right_gripper


def _gripper_boxes(opening_width, color=(1.0, 0.0, 0.0)):
    """Create two symmetric gripper boxes around the tool frame.

    opening_width: distance between inner faces (meters)
    """
    length = 0.15
    width = 0.09
    height = 0.09
    half_offset = max(opening_width * 0.5, 0.0) + width * 0.5
    extents = [length, width, height]

    def _box_mesh():
        box = trimesh.creation.box(extents=extents)
        rgba = np.array([color[0], color[1], color[2], 1.0])
        box.visual.vertex_colors = (rgba * 255).astype(np.uint8)
        return pyrender.Mesh.from_trimesh(box, smooth=False)

    left_mesh = _box_mesh()
    right_mesh = _box_mesh()

    left_pose = np.eye(4)
    right_pose = np.eye(4)
    x_offset = 0.03
    left_pose[:3, 3] = np.array([x_offset, -half_offset, 0.0])
    right_pose[:3, 3] = np.array([x_offset, half_offset, 0.0])
    return (left_mesh, left_pose), (right_mesh, right_pose)

class CombinedVisualizer:
    def __init__(self, replay_buffer, episodes, record_mode=False, record_episode=0, 
                 output_video=None, record_fps=30, continue_after_record=False):
        self.rb = replay_buffer
        self.episodes = episodes
        self.ep_idx = record_episode if record_mode else 0
        self.frame_idx = 0
        self.record_mode = record_mode
        self.output_video = output_video
        self.record_fps = record_fps
        self.continue_after_record = continue_after_record
        
        self.load_episode()
        self.setup_renderers()
        
        if record_mode:
            self.record_episode()
            if not continue_after_record:
                print(" 录制完成，退出程序")
                return
        
        self.print_help()
        self.run()
    
    def load_episode(self):
        """加载当前episode"""
        ep_id = self.episodes[self.ep_idx]
        self.data, self.has = load_episode_data(self.rb, ep_id)
        self.frame_idx = 0
        self.setup_camera_params()
        print(f" 加载 Episode {ep_id} ({self.ep_idx + 1}/{len(self.episodes)}), 帧数: {len(self.data['robot0']['poses'])}")
    
    def setup_camera_params(self):
        """设置点云相机参数"""
        self.cam_params = {}
        for r in ROBOT_IDS:
            for side in ['left', 'right']:
                key = f'robot{r}_{side}_pc'
                pcs = self.data[f'robot{r}'][f'{side}_pc']
                # 找第一个非空点云设置相机
                params = None
                for pc in pcs:
                    if len(pc) > 0:
                        params = calc_camera_params(pc)
                        break
                self.cam_params[key] = params or calc_camera_params(np.empty((0, 3)))
    
    def setup_renderers(self):
        """初始化渲染器"""
        self.render_size = (400, 300)
        self.world_render_size = (self.render_size[0] * 2, self.render_size[1] * 2)
        self.renderers = {}
        
        # 点云渲染器
        for r in ROBOT_IDS:
            for side in ['left', 'right']:
                self.renderers[f'robot{r}_{side}_pc'] = self._create_renderer()
        
        # 统一世界坐标系渲染器
        self.renderers['world'] = self._create_renderer(size=self.world_render_size)
    
    def _create_renderer(self, size=None):
        """创建单个渲染器"""
        try:
            w, h = size if size is not None else self.render_size
            return pyrender.OffscreenRenderer(w, h)
        except Exception as exc:
            raise RuntimeError(f"无法创建渲染器: {exc}")
    
    def render_pointcloud(self, points, renderer, cam_params):
        """渲染点云"""
        scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1, 1.0])
        pc_mesh = _pointcloud_mesh(points, color=[1.0, 1.0, 0.0])
        if pc_mesh:
            scene.add(pc_mesh)
        scene.add(_axis_mesh(size=0.05))

        cam_pose = _lookat_camera_pose(cam_params['eye'], cam_params['center'], cam_params['up'])
        camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(60.0))
        scene.add(camera, pose=cam_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
        scene.add(light, pose=cam_pose)

        color, _ = renderer.render(scene, flags=RenderFlags.RGBA)
        return color[:, :, :3]
    
    def render_trajectory(self, poses, current_idx, renderer, color, gripper_width=None):
        """渲染轨迹"""
        scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1, 1.0])

        if not poses or current_idx >= len(poses):
            cam_pose = _lookat_camera_pose([0.5, -0.3, 0.8], [0, 0, 0], [0, 0, 1])
            camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(60.0))
            scene.add(camera, pose=cam_pose)
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
            scene.add(light, pose=cam_pose)
            color_img, _ = renderer.render(scene, flags=RenderFlags.RGBA)
            return color_img[:, :, :3]

        pts = np.array([p[:3, 3] for p in poses[:current_idx + 1]], dtype=np.float64)

        line_mesh = _line_mesh(pts, color=color, radius=0.01)
        if line_mesh:
            scene.add(line_mesh)

        pts_mesh = _pointcloud_mesh(pts, color=color)
        if pts_mesh:
            scene.add(pts_mesh)

        frame_axis = _axis_mesh(size=0.05)
        frame_pose = poses[current_idx]
        scene.add(frame_axis, pose=frame_pose)

        placeholder_cad = _placeholder_cad_box()
        cad_offset = np.eye(4)
        cad_offset[0, 3] = -0.05
        scene.add(placeholder_cad, pose=frame_pose @ cad_offset)

        if gripper_width is not None:
            (left_mesh, left_pose), (right_mesh, right_pose) = _gripper_boxes(gripper_width)
            scene.add(left_mesh, pose=frame_pose @ left_pose)
            scene.add(right_mesh, pose=frame_pose @ right_pose)
            
            # Quest手柄（垂直向上）
            ctrl = _quest_controller_mesh(is_left=(r==1))  # 左手装右手柄，右手装左手柄
            if ctrl:
                from scipy.spatial.transform import Rotation
                ctrl_tf = np.eye(4)
                # 旋转90度使其垂直
                rot = Rotation.from_euler('y', 90, degrees=True).as_matrix()
                ctrl_tf[:3, :3] = rot
                ctrl_tf[:3, 3] = [0, 0, 0.03]  # 在底座上方
                scene.add(ctrl, pose=frame_pose @ ctrl_tf)
                

        origin_axis = _axis_mesh(size=0.03)
        scene.add(origin_axis, pose=np.eye(4))

        center = pts.mean(axis=0)
        extent = pts.max(axis=0) - pts.min(axis=0)
        dist = max(np.max(extent) * 2.0, 0.3)
        eye = center + np.array([dist * 0.6, -dist * 0.4, dist * 0.8])
        cam_pose = _lookat_camera_pose(eye, center, [0, 0, 1])
        camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(60.0))
        scene.add(camera, pose=cam_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
        scene.add(light, pose=cam_pose)

        color_img, _ = renderer.render(scene, flags=RenderFlags.RGBA)
        return color_img[:, :, :3]

    def render_world_scene(self, current_idx):
        """统一世界坐标系渲染"""
        scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1, 1.0])
        scene.add(_axis_mesh(size=0.15))

        colors = {0: [1, 0, 0], 1: [0, 1, 0]}
        all_pts = []

        for r in ROBOT_IDS:
            prefix = f'robot{r}'
            poses = self.data[prefix].get('poses', [])
            if not poses or current_idx >= len(poses):
                continue

            pts = np.array([p[:3, 3] for p in poses[:current_idx + 1]], dtype=np.float64)
            all_pts.append(pts)

            line_mesh = _line_mesh(pts, color=colors[r], radius=0.01)
            if line_mesh:
                scene.add(line_mesh)

            pts_mesh = _pointcloud_mesh(pts, color=colors[r])
            if pts_mesh:
                scene.add(pts_mesh)

            frame_pose = poses[current_idx]
            scene.add(_axis_mesh(size=0.06), pose=frame_pose)
            cad_offset = np.eye(4)
            cad_offset[0, 3] = -0.05
            scene.add(_placeholder_cad_box(), pose=frame_pose @ cad_offset)

            gripper = self.data[prefix].get('gripper', [])
            if gripper and current_idx < len(gripper):
                # 加载一对真实夹爪
                left_gripper, right_gripper = load_gripper_pair()
                if left_gripper and right_gripper:
                    # 左夹爪
                    left_tf = np.eye(4)
                    left_tf[:3, 3] = [0.05, -0.06, -0.03]
                    scene.add(left_gripper, pose=frame_pose @ left_tf)
                    
                    # 右夹爪
                    right_tf = np.eye(4)
                    right_tf[:3, 3] = [0.05, 0.06, -0.03]
                    scene.add(right_gripper, pose=frame_pose @ right_tf)
                else:
                    # 备用
                    (left_mesh, left_pose), (right_mesh, right_pose) = _gripper_boxes(float(gripper[current_idx]))
                    scene.add(left_mesh, pose=frame_pose @ left_pose)
                    scene.add(right_mesh, pose=frame_pose @ right_pose)
            
            # 手腕连接件（圆柱）
            wrist = trimesh.creation.cylinder(radius=0.02, height=0.04, sections=16)
            wrist.visual.vertex_colors = np.array([150, 150, 150, 255], dtype=np.uint8)
            wrist_mesh = pyrender.Mesh.from_trimesh(wrist, smooth=False)
            wrist_tf = np.eye(4)
            wrist_tf[:3, 3] = [0, 0, 0.01]  # 手柄下方
            scene.add(wrist_mesh, pose=frame_pose @ wrist_tf)
            
            # 夹爪底座
            base = trimesh.creation.box(extents=[0.04, 0.16, 0.025])
            base.visual.vertex_colors = np.array([130, 130, 130, 255], dtype=np.uint8)
            base_mesh = pyrender.Mesh.from_trimesh(base, smooth=False)
            base_tf = np.eye(4)
            base_tf[:3, 3] = [0, 0, -0.01]  # 手腕下方
            scene.add(base_mesh, pose=frame_pose @ base_tf)
            
            # Quest手柄（垂直向上）
            ctrl = _quest_controller_mesh(is_left=(r==1))  # 左手装右手柄，右手装左手柄
            if ctrl:
                from scipy.spatial.transform import Rotation
                ctrl_tf = np.eye(4)
                # 旋转90度使其垂直
                rot = Rotation.from_euler('y', 90, degrees=True).as_matrix()
                ctrl_tf[:3, :3] = rot
                ctrl_tf[:3, 3] = [0, 0, 0.03]  # 在底座上方
                scene.add(ctrl, pose=frame_pose @ ctrl_tf)
                

        if not all_pts:
            cam_pose = _lookat_camera_pose([0.5, -0.3, 0.8], [0, 0, 0], [0, 0, 1])
        else:
            pts = np.vstack(all_pts)
            center = pts.mean(axis=0)
            extent = pts.max(axis=0) - pts.min(axis=0)
            dist = max(np.max(extent) * 2.0, 0.3)
            eye = center + np.array([-dist * 1.5, 0, dist * 0.05])
            cam_pose = _lookat_camera_pose(eye, center, [0, 0, 1])

        camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(60.0))
        scene.add(camera, pose=cam_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
        scene.add(light, pose=cam_pose)

        color_img, _ = self.renderers['world'].render(scene, flags=RenderFlags.RGBA)
        return color_img[:, :, :3]
    
    def get_frame_images(self):
        """获取当前帧所有渲染图像"""
        pc_images = {}
        world_image = self.render_world_scene(self.frame_idx)
        
        for r in ROBOT_IDS:
            prefix = f'robot{r}'
            
            # 点云
            for side in ['left', 'right']:
                key = f'{prefix}_{side}_pc'
                pcs = self.data[prefix].get(f'{side}_pc', [])
                if pcs and self.frame_idx < len(pcs):
                    pc_images[key] = self.render_pointcloud(
                        pcs[self.frame_idx], self.renderers[key], self.cam_params[key])
            
        
        return pc_images, world_image
    
    def create_info_bar(self, frame_idx, width, height=120):
        """创建信息栏"""
        bar = np.zeros((height, width, 3), dtype=np.uint8)
        bar[:] = [30, 30, 30]
        
        ep_id = self.episodes[self.ep_idx]
        max_frames = len(self.data['robot0']['poses'])
        
        lines = [f"Episode {ep_id} ({self.ep_idx + 1}/{len(self.episodes)}) | Frame {frame_idx}/{max_frames - 1}"]
        
        for r in ROBOT_IDS:
            prefix = f'robot{r}'
            poses = self.data[prefix].get('poses', [])
            if poses and frame_idx < len(poses):
                pos = poses[frame_idx][:3, 3]
                lines.append(f"Robot{r} Pose: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            
            gripper = self.data[prefix].get('gripper', [])
            if gripper and frame_idx < len(gripper):
                lines.append(f"Robot{r} Gripper: {gripper[frame_idx]:.4f}m")
        
        # 点云信息
            pc_info = []
        for r in ROBOT_IDS:
            for side in ['left', 'right']:
                pcs = self.data[f'robot{r}'].get(f'{side}_pc', [])
                if pcs and frame_idx < len(pcs):
                    pc_info.append(f"R{r}-{side[0].upper()}:{len(pcs[frame_idx])}")
        if pc_info:
            lines.append(f"Point Clouds: {' | '.join(pc_info)}")
        
        for i, line in enumerate(lines):
            cv2.putText(bar, line, (10, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return bar
    
    def render_frame(self, frame_idx):
        """渲染单帧"""
        self.frame_idx = frame_idx
        pc_images, world_image = self.get_frame_images()
        combined = create_combined_image(self.data, frame_idx, pc_images, world_image)
        
        if combined is None:
            return None
        
        info_bar = self.create_info_bar(frame_idx, combined.shape[1])
        return np.vstack([combined, info_bar])
    
    def update_display(self):
        """更新显示"""
        frame = self.render_frame(self.frame_idx)
        if frame is not None:
            cv2.imshow("Combined View", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def record_episode(self):
        """录制当前episode"""
        max_frames = len(self.data['robot0']['poses'])
        print(f" 开始录制 Episode {self.episodes[self.ep_idx]}, 帧数: {max_frames}, FPS: {self.record_fps}")
        
        # 获取第一帧确定尺寸
        first_frame = self.render_frame(0)
        if first_frame is None:
            print(" 无法生成帧，录制失败")
            return
        
        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.output_video, fourcc, self.record_fps, (w, h))
        
        if not writer.isOpened():
            print(f" 无法创建视频文件: {self.output_video}")
            return
        
        for i in range(max_frames):
            frame = self.render_frame(i)
            if frame is not None:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if (i + 1) % 10 == 0:
                print(f"   进度: {i + 1}/{max_frames} ({(i + 1) / max_frames * 100:.1f}%)")
        
        writer.release()
        print(f" 录制完成: {self.output_video}")
        self.frame_idx = 0
    
    def print_help(self):
        """打印帮助"""
        print("\n" + "=" * 50)
        print(" 控制: A/D=前/后帧  W/S=前/后Episode  R=重置视角  Q=退出")
        print("=" * 50 + "\n")
    
    def run(self):
        """运行可视化循环"""
        while True:
            self.update_display()
            key = cv2.waitKey(30) & 0xFF
            
            if key in [ord('d'), ord('D')]:
                max_f = len(self.data['robot0']['poses'])
                if self.frame_idx < max_f - 1:
                    self.frame_idx += 1
                elif self.ep_idx < len(self.episodes) - 1:
                    self.ep_idx += 1
                    self.load_episode()
            elif key in [ord('a'), ord('A')]:
                if self.frame_idx > 0:
                    self.frame_idx -= 1
            elif key in [ord('w'), ord('W')]:
                if self.ep_idx < len(self.episodes) - 1:
                    self.ep_idx += 1
                    self.load_episode()
            elif key in [ord('s'), ord('S')]:
                if self.ep_idx > 0:
                    self.ep_idx -= 1
                    self.load_episode()
            elif key in [ord('r'), ord('R')]:
                self.setup_camera_params()
                print("📷 重置视角")
            elif key in [ord('q'), ord('Q')]:
                print("👋 退出")
                break
        
        cv2.destroyAllWindows()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Combined visualizer')
    parser.add_argument('zarr_path', nargs='?', 
                       default='C:\\Users\\ruich\\Downloads\\_0115_bi_pick_and_place_2ver.zarr.zip')
    parser.add_argument('--record', type=bool, default=True)
    parser.add_argument('--record_episode', type=int, default=1)
    parser.add_argument('--output_video', type=str, default=None)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--continue_after_record', type=bool, default=True)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.zarr_path):
        print(f" 找不到文件: {args.zarr_path}")
        return
    
    print(f" 加载: {args.zarr_path}")
    
    store = ZipStore(args.zarr_path, mode='r')
    try:
        root = zarr.open_group(store=store, mode='r')
        rb = ReplayBuffer.create_from_group(root)
        print(f" 加载完成, 帧数: {rb.n_steps}, Episodes: {rb.n_episodes}")

        if args.record and args.record_episode >= rb.n_episodes:
            print(f" Episode {args.record_episode} 超出范围 (共 {rb.n_episodes} 个)")
            return

        if args.record and args.output_video is None:
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            name = os.path.basename(args.zarr_path).replace('.zarr.zip', '')
            args.output_video = f"recorded_ep{args.record_episode}_{name}_{ts}.mp4"

        CombinedVisualizer(rb, np.arange(rb.n_episodes), args.record, args.record_episode,
                           args.output_video, args.fps, args.continue_after_record)
    finally:
        store.close()

if __name__ == "__main__":
    main()


