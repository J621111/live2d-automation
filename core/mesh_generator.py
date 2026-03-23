"""ArtMesh generation helpers for Live2D layers."""

from typing import Any, cast

import numpy as np
from loguru import logger
from scipy.spatial import Delaunay

JsonDict = dict[str, Any]
JsonList = list[JsonDict]


class ArtMeshGenerator:
    """Generate simplified ArtMesh geometry from texture layer bounds."""

    def __init__(self) -> None:
        self.mesh_density = 0.02
        self.min_triangle_area = 100

    async def generate_from_layers(self, layers: list[JsonDict]) -> JsonDict:
        meshes: JsonDict = {}
        logger.info("Generating ArtMesh data...")

        for layer in layers:
            layer_name = str(layer["name"])
            bounds = dict(layer["bounds"])
            meshes[layer_name] = {
                "layer_name": layer_name,
                "mesh": self._generate_mesh_for_layer(layer, bounds),
                "bounds": bounds,
            }

        return meshes

    def _generate_mesh_for_layer(self, layer: JsonDict, bounds: JsonDict) -> JsonDict:
        width = float(bounds["width"])
        height = float(bounds["height"])
        layer_name = str(layer["name"])
        if "eye" in layer_name or "eyebrow" in layer_name:
            density = 0.03
        elif "mouth" in layer_name or "nose" in layer_name:
            density = 0.025
        else:
            density = self.mesh_density

        num_points_x = max(3, int(width * density))
        num_points_y = max(3, int(height * density))
        points = self._generate_grid_points(width, height, num_points_x, num_points_y)
        points = np.vstack([points, self._generate_edge_points(width, height)])

        try:
            tri = Delaunay(points)
            valid_triangles = self._filter_triangles(points[tri.simplices])
        except Exception as exc:
            logger.warning(f"Delaunay triangulation failed, using fallback grid: {exc}")
            valid_triangles = self._create_simple_grid(width, height)

        vertices: JsonList = []
        for index, point in enumerate(points):
            vertices.append(
                {
                    "index": index,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "u": float(point[0] / width) if width > 0 else 0.0,
                    "v": float(point[1] / height) if height > 0 else 0.0,
                }
            )

        faces: JsonList = []
        for tri_index, tri_points in enumerate(valid_triangles):
            vertex_indices: list[int] = []
            for vertex in tri_points:
                for index, point in enumerate(points):
                    if np.allclose(vertex, point):
                        vertex_indices.append(index)
                        break
            if len(vertex_indices) == 3:
                faces.append({"index": tri_index, "vertices": vertex_indices})

        return {
            "vertices": vertices,
            "faces": faces,
            "num_vertices": len(vertices),
            "num_faces": len(faces),
        }

    def _generate_grid_points(self, width: float, height: float, nx: int, ny: int) -> np.ndarray:
        x = np.linspace(0, width, nx)
        y = np.linspace(0, height, ny)
        xx: np.ndarray
        yy: np.ndarray
        xx, yy = np.meshgrid(x, y)
        points = cast(np.ndarray, np.column_stack([xx.ravel(), yy.ravel()]))
        noise = np.random.randn(*points.shape) * 2
        points = points + noise
        points[:, 0] = np.clip(points[:, 0], 0, width)
        points[:, 1] = np.clip(points[:, 1], 0, height)
        return cast(np.ndarray, points)

    def _generate_edge_points(self, width: float, height: float) -> np.ndarray:
        edge_points: list[list[float]] = []
        for index in range(5):
            edge_points.append([width * index / 4, 0])
            edge_points.append([width * index / 4, height])
        for index in range(3):
            edge_points.append([0, height * index / 2])
            edge_points.append([width, height * index / 2])
        return cast(np.ndarray, np.array(edge_points, dtype=float))

    def _filter_triangles(self, triangles: np.ndarray) -> np.ndarray:
        valid = [
            triangle
            for triangle in triangles
            if self._triangle_area(triangle) >= self.min_triangle_area
        ]
        return cast(np.ndarray, np.array(valid)) if valid else triangles

    def _triangle_area(self, tri: np.ndarray) -> float:
        p1, p2, p3 = tri
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        return float(area)

    def _create_simple_grid(self, width: float, height: float) -> np.ndarray:
        nx, ny = 4, 4
        x = np.linspace(0, width, nx)
        y = np.linspace(0, height, ny)
        triangles: list[list[list[float]]] = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                p1 = [float(x[i]), float(y[j])]
                p2 = [float(x[i + 1]), float(y[j])]
                p3 = [float(x[i]), float(y[j + 1])]
                p4 = [float(x[i + 1]), float(y[j + 1])]
                triangles.append([p1, p2, p3])
                triangles.append([p2, p4, p3])
        return cast(np.ndarray, np.array(triangles, dtype=float))

    async def generate_deformer_mesh(self, artmesh: JsonDict, deformer_type: str) -> JsonDict:
        vertices = list(artmesh["vertices"])
        if deformer_type == "warp":
            return self._create_warp_mesh(vertices)
        if deformer_type == "rotation":
            return self._create_rotation_mesh(vertices)
        return artmesh

    def _create_warp_mesh(self, vertices: list[JsonDict]) -> JsonDict:
        control_points: JsonList = []
        for vertex in vertices:
            control_points.append(
                {
                    "index": len(control_points),
                    "x": vertex["x"],
                    "y": vertex["y"],
                    "is_control": True,
                }
            )
        return {"vertices": vertices + control_points, "control_points": len(control_points)}

    def _create_rotation_mesh(self, vertices: list[JsonDict]) -> JsonDict:
        x_coords = [float(vertex["x"]) for vertex in vertices]
        y_coords = [float(vertex["y"]) for vertex in vertices]
        center = {"x": sum(x_coords) / len(x_coords), "y": sum(y_coords) / len(y_coords)}
        return {"vertices": vertices, "center": center, "rotation_axis": center}
