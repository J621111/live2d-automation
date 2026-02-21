"""
ArtMesh 网格生成器
为 Live2D 创建可变形网格
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.spatial import Delaunay
from loguru import logger
import cv2


class ArtMeshGenerator:
    """ArtMesh 网格生成器"""

    def __init__(self):
        self.mesh_density = 0.02  # 网格密度参数
        self.min_triangle_area = 100  # 最小三角形面积

    async def generate_from_layers(self, layers: List[Dict]) -> Dict[str, Any]:
        """
        从图层生成 ArtMesh

        Args:
            layers: 图层列表

        Returns:
            各图层的网格信息
        """
        meshes = {}

        logger.info("生成 ArtMesh 网格...")

        for layer in layers:
            layer_name = layer["name"]
            bounds = layer["bounds"]

            # 生成网格
            mesh = self._generate_mesh_for_layer(layer, bounds)

            meshes[layer_name] = {
                "layer_name": layer_name,
                "mesh": mesh,
                "bounds": bounds,
            }

        return meshes

    def _generate_mesh_for_layer(self, layer: Dict, bounds: Dict) -> Dict:
        """为单个图层生成网格"""
        width = bounds["width"]
        height = bounds["height"]

        # 根据部位类型调整网格密度
        layer_name = layer["name"]
        if "eye" in layer_name or "eyebrow" in layer_name:
            # 眼睛和眉毛需要更精细的网格
            density = 0.03
        elif "mouth" in layer_name or "nose" in layer_name:
            # 嘴和鼻子也需要精细网格
            density = 0.025
        else:
            # 身体部位可以用较粗的网格
            density = self.mesh_density

        # 计算网格点数量
        num_points_x = max(3, int(width * density))
        num_points_y = max(3, int(height * density))

        # 生成网格点
        points = self._generate_grid_points(width, height, num_points_x, num_points_y)

        # 添加边缘控制点
        edge_points = self._generate_edge_points(width, height)
        points = np.vstack([points, edge_points])

        # 使用 Delaunay 三角化
        try:
            tri = Delaunay(points)
            triangles = points[tri.simplices]

            # 过滤过小的三角形
            valid_triangles = self._filter_triangles(triangles)

        except Exception as e:
            logger.warning(f"Delaunay 三角化失败，使用简单网格: {e}")
            valid_triangles = self._create_simple_grid(width, height)

        # 构建顶点列表
        vertices = []
        for i, point in enumerate(points):
            vertices.append(
                {
                    "index": i,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "u": float(point[0] / width) if width > 0 else 0.0,
                    "v": float(point[1] / height) if height > 0 else 0.0,
                }
            )

        # 构建面列表（三角形）
        faces = []
        for tri_idx, tri in enumerate(valid_triangles):
            # 找到三角形的顶点索引
            v_indices = []
            for v in tri:
                for i, p in enumerate(points):
                    if np.allclose(v, p):
                        v_indices.append(i)
                        break

            if len(v_indices) == 3:
                faces.append({"index": tri_idx, "vertices": v_indices})

        return {
            "vertices": vertices,
            "faces": faces,
            "num_vertices": len(vertices),
            "num_faces": len(faces),
        }

    def _generate_grid_points(
        self, width: float, height: float, nx: int, ny: int
    ) -> np.ndarray:
        """生成规则网格点"""
        x = np.linspace(0, width, nx)
        y = np.linspace(0, height, ny)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        # 添加一些随机扰动，使网格更自然
        noise = np.random.randn(*points.shape) * 2
        points = points + noise

        # 确保点在边界内
        points[:, 0] = np.clip(points[:, 0], 0, width)
        points[:, 1] = np.clip(points[:, 1], 0, height)

        return points

    def _generate_edge_points(self, width: float, height: float) -> np.ndarray:
        """生成边缘控制点"""
        edge_points = []

        # 上边缘
        for i in range(5):
            x = width * i / 4
            edge_points.append([x, 0])

        # 下边缘
        for i in range(5):
            x = width * i / 4
            edge_points.append([x, height])

        # 左边缘
        for i in range(3):
            y = height * i / 2
            edge_points.append([0, y])

        # 右边缘
        for i in range(3):
            y = height * i / 2
            edge_points.append([width, y])

        return np.array(edge_points)

    def _filter_triangles(self, triangles: np.ndarray) -> np.ndarray:
        """过滤掉过小的三角形"""
        valid = []

        for tri in triangles:
            # 计算三角形面积
            area = self._triangle_area(tri)
            if area >= self.min_triangle_area:
                valid.append(tri)

        return np.array(valid) if valid else triangles

    def _triangle_area(self, tri: np.ndarray) -> float:
        """计算三角形面积"""
        p1, p2, p3 = tri
        return 0.5 * abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
        )

    def _create_simple_grid(self, width: float, height: float) -> np.ndarray:
        """创建简单的四边形网格（三角化后）"""
        # 创建 4x4 网格
        nx, ny = 4, 4
        x = np.linspace(0, width, nx)
        y = np.linspace(0, height, ny)

        triangles = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                # 每个四边形分成两个三角形
                p1 = [x[i], y[j]]
                p2 = [x[i + 1], y[j]]
                p3 = [x[i], y[j + 1]]
                p4 = [x[i + 1], y[j + 1]]

                triangles.append([p1, p2, p3])
                triangles.append([p2, p4, p3])

        return np.array(triangles)

    async def generate_deformer_mesh(self, artmesh: Dict, deformer_type: str) -> Dict:
        """
        为变形器生成专用网格

        Args:
            artmesh: 原始 ArtMesh
            deformer_type: 变形器类型 (warp/rotation)

        Returns:
            变形器网格
        """
        vertices = artmesh["vertices"]

        if deformer_type == "warp":
            # 弯曲变形器需要更密集的网格
            return self._create_warp_mesh(vertices)
        elif deformer_type == "rotation":
            # 旋转变形器需要中心点
            return self._create_rotation_mesh(vertices)
        else:
            return artmesh

    def _create_warp_mesh(self, vertices: List[Dict]) -> Dict:
        """创建弯曲变形网格"""
        # 简化为在原网格上添加控制点
        control_points = []
        for v in vertices:
            control_points.append(
                {
                    "index": len(control_points),
                    "x": v["x"],
                    "y": v["y"],
                    "is_control": True,
                }
            )

        return {
            "vertices": vertices + control_points,
            "control_points": len(control_points),
        }

    def _create_rotation_mesh(self, vertices: List[Dict]) -> Dict:
        """创建旋转变形网格"""
        # 计算中心点
        x_coords = [v["x"] for v in vertices]
        y_coords = [v["y"] for v in vertices]
        center = {
            "x": sum(x_coords) / len(x_coords),
            "y": sum(y_coords) / len(y_coords),
        }

        return {"vertices": vertices, "center": center, "rotation_axis": center}
