import numpy as np
import open3d as o3d


def main():
    pcd = o3d.io.read_point_cloud("Cloud_2.ply")
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
