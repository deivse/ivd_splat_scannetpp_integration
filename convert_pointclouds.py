import argparse
import os
import numpy as np
import open3d  # type: ignore
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def load_pointcloud_ply(
    path: Path | str,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Loads a point cloud from a PLY file.

    Returns:
        pts: Nx3 array of point coordinates
        rgbs: Nx3 array of point colors (float, <0, 1> range), or None if no colors are present
    """
    pcd = open3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    rgbs = None
    if pcd.has_colors():
        rgbs = np.asarray(pcd.colors)
    if pts is None:
        raise RuntimeError(
            f"Failed to load point cloud from {path}, see open3d log messages."
        )
    return pts, rgbs


def export_pointcloud_ply(
    pts: np.ndarray,
    rgbs: Optional[np.ndarray],
    path: Path | str,
):
    """Exports a point cloud to a PLY file."""
    pcd = open3d.geometry.PointCloud()

    pcd.points = open3d.utility.Vector3dVector(pts)

    if rgbs is not None:
        if rgbs.max() > 1.0:
            rgbs = rgbs.astype(np.float64) / 255.0
        pcd.colors = open3d.utility.Vector3dVector(rgbs)

    open3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def convert_pointcloud(
    input_path: Path | str,
    output_path: Path | str,
):
    points, colors = load_pointcloud_ply(input_path)
    points[:, 1:3] *= -1
    rot_z_90 = np.array(
        [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    points = points @ rot_z_90.T
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    export_pointcloud_ply(points, colors, output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert point cloud data between different formats."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the input scannetpp_data/data directory.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory where converted point clouds will be saved under <scene_name>/scans/pc_aligned.ply.",
    )
    return parser.parse_args()


def process_scene(scene, output_dir):
    scene_name = scene.name
    try:
        convert_pointcloud(
            scene / "scans" / "pc_aligned.ply",
            Path(output_dir) / scene_name / "scans" / "pc_aligned.ply",
        )
    except Exception as e:
        print(f"Error processing {scene_name}: {e}")


def main():
    args = parse_args()

    scenes = Path(args.input_dir).glob("*")
    import concurrent.futures
    import functools

    scene_list = [s for s in scenes if s.is_dir()]

    process_func = functools.partial(process_scene, output_dir=Path(args.output_dir))

    max_workers = None
    slurm_max_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", -1))
    if slurm_max_cpus > 0:
        max_workers = slurm_max_cpus

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(process_func, scene_list),
                total=len(scene_list),
                desc="Converting point clouds",
            )
        )


if __name__ == "__main__":
    main()
