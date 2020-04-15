#!/usr/bin/env python3

import os
import argparse
import re
import open3d
import prompter
import numpy as np
import copy
import toml
from pathlib import Path
import mathutils

# Global regular expressions
index_re = re.compile('(\d+)(?!.*\d)')  # Gets last number in a string


def get_index(path):
    m = index_re.search(str(path))
    if m is None:
        raise RuntimeError("Index could not be found.")
    return m.group(0)


def is_empty(path):
    return not bool(sorted(path.rglob('*')))


def matrix_to_tum(matrix):
    transform = mathutils.Matrix(matrix)
    q = transform.to_quaternion()
    t = transform.to_translation()
    return [t.x, t.y, t.z, q.x, q.y, q.z, q.w]


def tum_to_matrix(tx, ty, tz, qx, qy, qz, qw):
    m = np.eye(4)
    m[0:3, 0:3] = mathutils.Quaternion((qw, qx, qy, qz)).to_matrix()
    m[0:3, 3] = [tx, ty, tz]
    return m


def make_paths_absolute_run_config(run_config, data_path):
    if 'run' in run_config:
        for r in run_config['run']:
            r['input'] = data_path / r['input']
            assert os.path.exists(r['input']), f'run input not found at: {r["input"]}'
            r['output'] = data_path / r.get('output', f"processed/{r['input'].stem}")
    if 'object' in run_config:
        for o in run_config['object']:
            o['cloud'] = data_path / o['cloud']
            assert os.path.exists(o['cloud']), f'cloud not found at: {o["cloud"]}'
            o['mesh'] = data_path / o['mesh']
            assert os.path.exists(o['mesh']), f'mesh not found at: {o["mesh"]}'


def get_camera_config(config_run, run):
    camera_config = [c for c in config_run['camera'] if c['id'] == run['camera']]
    if len(camera_config) != 1:
        raise RuntimeError('Could not find camera for run.')
    return camera_config[0]


def get_object_configs(config_run, run):
    if 'object' not in config_run or 'objects' not in run:
        return []
    object_configs = [o for o in config_run['object'] if o['id'] in run['objects']]
    if len(object_configs) == 0:
        raise RuntimeError('Sequence has no objects.')
    return object_configs


def parse_and_load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to aligner configuration")
    parser.add_argument("--run", required=True, help="Path to run configuration")
    parser.add_argument("--data", required=True, help="Path to data root directory")
    args, _ = parser.parse_known_args()
    assert os.path.exists(args.config), f'config file not found at: {args.config}'
    assert os.path.exists(args.run), f'run file not found at: {args.run}'
    config = toml.load(args.config)
    config_run = toml.load(args.run)
    make_paths_absolute_run_config(config_run, Path(args.data))
    return config, config_run


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.visualization.draw_geometries([source_temp, target_temp])


def register_selected_points(cloud_object, cloud_scene, show):
    def pick_points(pcd):
        print("")
        print("1) Please pick at least three correspondences using [shift + left click]")
        print("   Press [shift + right click] to undo point picking")
        print("2) Afther picking points, press q for close the window")
        vis = open3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()
        print("")
        return vis.get_picked_points()

    points_object = pick_points(cloud_object)
    points_scene = pick_points(cloud_scene)
    num_points = len(points_object)

    if num_points < 3:
        print("You did not select enough points for manual alignment (>=3 required). Skipping ...")
        return
    if num_points != len(points_scene):
        print("Number of selected points not matching. Skipping ...")
        return

    corr = np.zeros((num_points, 2))
    corr[:, 0] = points_object
    corr[:, 1] = points_scene

    p2p = open3d.registration.TransformationEstimationPointToPoint()
    est = p2p.compute_transformation(cloud_object, cloud_scene, open3d.utility.Vector2iVector(corr))
    print("Manual estimation of transformation:", est)

    if show:
        print(":: Visualize manual alignment ...")
        draw_registration_result(cloud_object, cloud_scene, est)

    return est


class Sequence(object):
    def __init__(self, config):
        self.config = config
        self.name = self.config['input'].stem
        self.path_processed = self.config['output']
        self.path_frames = self.path_processed / "frames"
        self.path_reconstruction = self.path_processed / "reconstruction"
        self.path_camera_poses = self.path_reconstruction / "poses.txt"
        self.path_reconstructed_cloud = self.path_reconstruction / "model.ply"
        self.path_reconstructed_mesh = self.path_reconstruction / "model_mesh.ply"
        self.path_reconstructed_mesh_bg = self.path_reconstruction / "model_mesh_background.ply"
        self.path_annotations = self.path_processed / "annotations"
        self.path_alignments = self.path_processed / "alignments"
        self.rgb_frames = None
        self.depth_frames = None
        self.num_frames = None
        self.trajectory_camera = None  # T_wc (camera -> world)
        self.camera = None
        self.object_alignments = None  # T_wo (object -> world)

    def has_trajectory(self):
        return self.path_camera_poses.exists()

    def has_reconstructed_cloud(self):
        return self.path_reconstructed_cloud.exists()

    def has_reconstructed_mesh(self):
        return self.path_reconstructed_mesh.exists()

    def has_reconstructed_background_mesh(self):
        return self.path_reconstructed_mesh_bg.exists()

    def count_aligned_objects(self):
        self.load_object_alignments()
        return len(self.object_alignments)

    def has_reconstructed_background_mesh(self):
        return self.path_reconstructed_mesh_bg.exists()

    def load_frame_paths(self, force_reload=False):
        if self.num_frames is None or force_reload:
            self.rgb_frames = [self.path_frames / x for x in sorted(os.listdir(self.path_frames)) if "Color" in x]
            self.depth_frames = [self.path_frames / x for x in sorted(os.listdir(self.path_frames)) if "Depth" in x]
            assert len(self.rgb_frames) == len(self.depth_frames)
            self.num_frames = len(self.depth_frames)

    def load_object_alignments(self, force_reload=False):
        if self.object_alignments is None or force_reload:
            self.object_alignments = {}
            for p in [self.path_alignments / o for o in sorted(os.listdir(self.path_alignments)) if ".txt" in o]:
                self.object_alignments[int(get_index(p))] = np.loadtxt(p, delimiter=',')

    def load_trajectory(self, force_reload=False):
        if self.trajectory_camera is None or force_reload:
            T_wc = np.loadtxt(self.path_camera_poses)
            self.trajectory_camera = [tum_to_matrix(*T_wc[i, 1:4], *T_wc[i, 4:]) for i in range(T_wc.shape[0])]
            self.load_object_alignments()

    def get_camera(self, config_sequences):
        self.camera = get_camera_config(config_sequences, self.config)


class Object(object):
    def __init__(self, name, object_id, path_cloud, path_mesh=None, scale=None):
        assert os.path.exists(path_cloud), f'cloud file: {path_cloud} not found'
        if not path_mesh is None:
            assert os.path.exists(path_mesh), f'mesh file: {path_mesh} not found'
        self.path_cloud = path_cloud
        self.path_mesh = path_mesh
        self.cloud = None
        self.mean = None
        self.std = None
        self.bound_min = None
        self.bound_max = None
        self.id = object_id
        self.name = name
        self.scale = scale

    def load(self):
        if self.cloud is None:
            self.cloud = open3d.io.read_point_cloud(str(self.path_cloud))
            if self.scale is not None:
                self.cloud.scale(self.scale, center=False)
            self.compute_object_statistics()

    def compute_object_statistics(self):
        self.mean, cov = open3d.geometry.compute_point_cloud_mean_and_covariance(self.cloud)
        self.std = np.sqrt(np.diagonal(cov))
        self.bound_min, self.bound_max = self.cloud.get_min_bound(), self.cloud.get_max_bound()
        print(f"[{self.name}] object info",
              "\n| Object-space min corner:", self.bound_min,
              "\n| Object-space max corner:", self.bound_max,
              "\n| Mean:", self.mean,
              "\n| Std:", self.std)

    def get_boundingbox_corners(self):
        corners = np.zeros([8, 3])
        corners[0, :] = self.bound_min
        corners[1, :] = [self.bound_max[0], self.bound_min[1], self.bound_min[2]]
        corners[2, :] = [self.bound_min[0], self.bound_max[1], self.bound_min[2]]
        corners[3, :] = [self.bound_min[0], self.bound_min[1], self.bound_max[2]]
        corners[4, :] = [self.bound_max[0], self.bound_max[1], self.bound_min[2]]
        corners[5, :] = [self.bound_max[0], self.bound_min[1], self.bound_max[2]]
        corners[6, :] = [self.bound_min[0], self.bound_max[1], self.bound_max[2]]
        corners[7, :] = self.bound_max
        return corners

    def register(self,
                 cloud_scene,
                 path_output_alignment,
                 distance_threshold,
                 init_with_global_features=True,
                 point_to_plane=True):

        def preprocess_cloud(pcd, voxel_size=0.005):
            print(":: Downsample with a voxel size %.3f." % voxel_size)
            pcd_down = open3d.geometry.voxel_down_sample(pcd, voxel_size)

            radius_normal = voxel_size * 2
            print(":: Estimate normal with search radius %.3f." % radius_normal)
            open3d.geometry.estimate_normals(
                pcd_down, open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

            radius_feature = voxel_size * 5
            print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
            pcd_fpfh = open3d.registration.compute_fpfh_feature(
                pcd_down, open3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            return pcd_down, pcd_fpfh

        object_sampled, object_fpfh = preprocess_cloud(self.cloud)
        scene_sampled, scene_fpfh = preprocess_cloud(cloud_scene)

        if init_with_global_features is False:
            transformation = register_selected_points(self.cloud, cloud_scene, True)
        else:
            print(":: Execute RANSAC alignment")
            transformation = open3d.registration.registration_ransac_based_on_feature_matching(
                object_sampled, scene_sampled, object_fpfh, scene_fpfh,
                distance_threshold,
                open3d.registration.TransformationEstimationPointToPoint(False), 4,
                [open3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                 open3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
                open3d.registration.RANSACConvergenceCriteria(4000000, 500)).transformation
            print(":: Result:\n", transformation)
            print(":: Visualize initial alignment ...")
            draw_registration_result(object_sampled, scene_sampled, transformation)
            if not prompter.yesno('Is this initial alignment good enough? (Otherwise select matching points manually)'):
                transformation = register_selected_points(self.cloud, cloud_scene, True)

        alignment_accepted = False
        if point_to_plane:
            icp_estimation_method = open3d.registration.TransformationEstimationPointToPlane()
        else:
            icp_estimation_method = open3d.registration.TransformationEstimationPointToPoint()
        while not alignment_accepted:
            print(":: Execute ICP alignment")
            transformation = open3d.registration.registration_icp(
                object_sampled, scene_sampled, distance_threshold, transformation, icp_estimation_method).transformation
            print(":: Result:\n", transformation)
            print(":: Visualize refined alignment ...")
            draw_registration_result(object_sampled, scene_sampled, transformation)

            if prompter.yesno('Is alignment good? (Otherwise select matching points manually and run ICP)'):
                alignment_accepted = True
            else:
                transformation = register_selected_points(self.cloud, cloud_scene, True)
                if prompter.yesno('Skip ICP?'):
                    alignment_accepted = True

        # Write alignment to file
        np.savetxt(str(path_output_alignment), transformation, delimiter=",")

    def get_toml_description(self):
        return \
            f"[[object]]\n" \
            f"name = '{self.name}'\n" \
            f"id = {self.id}\n" \
            f"mesh = '{self.path_mesh}'\n" \
            f"mean = {np.array2string(self.mean, separator=', ')}\n" \
            f"stddev = {np.array2string(self.std, separator=', ')}"
