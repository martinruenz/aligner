#!/usr/bin/env python3

from tkinter import *
import subprocess
import os
from pathlib import Path
import shutil
from PIL import Image, ImageDraw
import tqdm
from termcolor import cprint
from common import *


class Aligner(object):

    def __init__(self, config_aligner, config_run, config_camera, config_objects):
        self.config_aligner = config_aligner
        self.config_run = config_run
        self.config_camera = config_camera
        self.path_scene_klg = config_run['input']
        self.path_directory = config_run['output']
        self.path_frames = self.path_directory / "frames"
        self.path_reconstruction = self.path_directory / "reconstruction"
        self.path_camera_poses = self.path_reconstruction / "poses.txt"
        self.path_reconstructed_cloud = self.path_reconstruction / "model.ply"
        self.path_reconstructed_mesh = self.path_reconstruction / "model_mesh.ply"
        self.path_reconstructed_mesh_bg = self.path_reconstruction / "model_mesh_background.ply"
        self.path_annotations = self.path_directory / "annotations"
        self.path_alignments = self.path_directory / "alignments"
        self.rgb_frames = []
        self.depth_frames = []
        self.trajectory = None
        self.compress_ply = self.config_aligner.get('compress_ply_files', False)
        self.objects = [Object(o['name'], o['id'], Path(o['mesh']), Path(o['cloud'])) for o in config_objects]
        self.cloud_scene_full = None
        self.enable_extract_background_model = self.config_aligner.get('enable_extract_background_model', False)
        self.show_intermediate_visualizations = config_aligner.get('show_intermediate_visualizations', False)
        self.camera_intrinsics = open3d.camera.PinholeCameraIntrinsic(config_camera['w'], config_camera['h'],
                                                                      config_camera['fx'], config_camera['fy'],
                                                                      config_camera['cx'], config_camera['cy'])

    def run(self):
        print("= Starting aligner ...")
        print("= Objects in scene:", len(self.objects), f"({[o.name for o in self.objects]})")

        if not self.path_directory.exists():
            print("= Creating", self.path_directory, "...")
            os.makedirs(self.path_directory)

        if not self.path_frames.exists():
            print("= Creating", self.path_frames, "...")
            os.mkdir(self.path_frames)
        else:
            print("=", self.path_frames, "already exists -- skipping creation of dir")

        if len(list(self.path_frames.rglob('*'))) == 0:
            self.extract_klg()
        else:
            print("=", self.path_frames, "extraction files ({}) already exist -- skipping extraction".format(
                len(list(self.path_frames.rglob('*')))))

        self.rgb_frames = [self.path_frames / x for x in sorted(os.listdir(self.path_frames)) if "Color" in x]
        self.depth_frames = [self.path_frames / x for x in sorted(os.listdir(self.path_frames)) if "Depth" in x]
        print("= Check matching of indices ...")
        for c, d in zip(self.rgb_frames, self.depth_frames):
            if get_index(str(c)) != get_index(str(d)):
                raise ("Indices not matching:", c, d)

        if not self.path_camera_poses.exists() or not self.path_reconstructed_cloud.exists() \
                or not self.path_reconstructed_mesh.exists():
            print("= Creating", self.path_reconstruction, "...")
            if not self.path_reconstruction.exists():
                os.mkdir(self.path_reconstruction)
            self.reconstruct()
        else:
            print("=", self.path_reconstruction, "already exists -- skipping reconstruction")

        if not self.config_aligner.get('enable_annotate_objects', True):
            return True

        if not self.path_annotations.exists():
            os.mkdir(self.path_annotations)
        if is_empty(self.path_annotations):
            print("= Creating", self.path_annotations, "...")
            self.annotate_frames()
            if self.config_aligner.get('show_intermediate_visualizations', False):
                self.load_scene_and_trajectory()
                self.visualize_bounding_boxes([self.compute_bounding_box(o.id) for o in self.objects],
                                              self.cloud_scene_full)
        else:
            print("=", self.path_annotations, "already exists -- skipping annotations")

        if not self.config_aligner.get('enable_align_objects', True):
            return True

        print("= Register objects ...")
        if not self.path_alignments.exists():
            os.mkdir(self.path_alignments)
        for o in self.objects:
            path_alignment = self.path_alignments / f"{o.id}.txt"
            if not path_alignment.exists():
                self.load_scene_and_trajectory()
                crop_min, crop_max = self.compute_bounding_box(o.id)
                cloud_scene_cropped = open3d.geometry.crop_point_cloud(self.cloud_scene_full, crop_min, crop_max)
                print(f"= Register object '{o.name}' to scene ...")
                o.load()
                o.register(cloud_scene_cropped,
                           path_alignment,
                           self.config_aligner['alignment']['distance_threshold'],
                           self.config_aligner['alignment'].get('enable_propose_automatic_alignment', True),
                           self.config_aligner['alignment'].get('point_to_plane', True))
            else:
                print("=", path_alignment, "already exists -- skipping alignment of", o.name)

        if self.enable_extract_background_model:
            if self.path_reconstructed_mesh_bg.exists():
                print(self.path_reconstructed_mesh_bg, " exists, skipping extraction of background")
            else:
                if not self.path_reconstructed_mesh.exists():
                    raise RuntimeError(self.path_reconstructed_mesh, " does not exist")
                self.load_scene_and_trajectory()
                mesh = open3d.io.read_triangle_mesh(str(self.path_reconstructed_mesh))
                bb_min, bb_max = mesh.get_min_bound(), mesh.get_max_bound()

                vertices = np.asarray(mesh.vertices)
                for o in self.objects:
                    o.load()

                    transform_wo = np.loadtxt(str(self.path_alignments / f"{o.id}.txt"), delimiter=",")
                    bb_corners = o.get_boundingbox_corners()
                    bb_corners = np.append(bb_corners, np.ones([8, 1]), 1).transpose()
                    bb_corners = np.dot(transform_wo, bb_corners)
                    obb_min, obb_max = bb_corners.min(1), bb_corners.max(1)
                    # obb_min, obb_max = np.dot(transform_wo, np.append(o.bound_min, 1)),\
                    #                    np.dot(transform_wo, np.append(o.bound_max, 1))
                    obb_min, obb_max = obb_min[0:3], obb_max[0:3]
                    print("minmax:", obb_min, obb_max)
                    for i in range(vertices.shape[0]):
                        if (obb_min < vertices[i, :]).all() and (obb_max > vertices[i, :]).all():  # TODO optimize
                            vertices[i, :] = 1e9, 1e9, 1e9

                mesh_bg = open3d.geometry.crop_triangle_mesh(mesh, bb_min, bb_max)
                open3d.io.write_triangle_mesh(str(self.path_reconstructed_mesh_bg), mesh_bg,
                                              compressed=self.compress_ply)
                # scene_copy = copy.deepcopy(self.cloud_scene_full)
                # cloud =

        return True

    def load_scene_and_trajectory(self):
        if self.cloud_scene_full is None:
            self.cloud_scene_full = open3d.io.read_point_cloud(str(self.path_reconstructed_cloud))
        if self.trajectory is None:
            self.trajectory = open3d.io.read_pinhole_camera_trajectory(str(self.path_camera_poses))

    def reconstruct(self):
        success = False
        if self.config_aligner['reconstruction_method'] == "elasticfusion":
            success = self.run_elasticfusion()
        elif self.config_aligner['reconstruction_method'] == "orbslam2":
            success = self.run_orbslam2()
        if not success or not self.path_camera_poses.exists() or not self.path_reconstructed_cloud.exists():
            raise RuntimeError(f"Reconstruction failed (method: {self.config_aligner['reconstruction_method']})")

    def run_elasticfusion(self):
        print("Running ElasticFusion ...")
        path_poses = self.path_scene_klg.parent / (self.path_scene_klg.name + ".freiburg")
        path_model = self.path_scene_klg.parent / (self.path_scene_klg.name + ".ply")
        if path_poses.exists():
            if path_model.exists():
                print("Pose and model file exist, skipping execution of ElasticFusion...")
                shutil.copy(path_poses, self.path_reconstruction / "poses.txt")
                shutil.copy(path_model, self.path_reconstruction / "model.ply")
                return True
            else:
                print("Cannot run ElasticFusion -- pose file already exists:", path_poses)
                return False
        if path_model.exists():
            print("Cannot run ElasticFusion -- model file already exists:", path_model)
            return False

        args = [self.config_aligner['path_elasticfusion'], '-l', self.path_scene_klg, '-icl', '-q']
        rc = subprocess.call(args)
        if rc != 0:
            print("Error running ElasticFusion")
            return False

        if not path_poses.exists():
            print("Elastic Fusion extracted no camera poses")
            return False
        if not path_model.exists():
            print("Elastic Fusion extracted no model")
            return False

        shutil.copy(path_poses, self.path_reconstruction / "poses.txt")
        shutil.copy(path_model, self.path_reconstruction / "model.ply")
        return True

    def run_orbslam2(self):
        print("Running ORB-SLAM2 ...")
        orb_fps = 30

        # Generate associations file
        path_associations = self.path_frames / "associations.txt"
        if path_associations.exists():
            print(path_associations, "already exists, skipping association")
        else:
            with open(path_associations, 'w') as association_file:
                for i, (c, d) in enumerate(zip(self.rgb_frames, self.depth_frames)):
                    association_file.write(f'{float(i) / orb_fps} {c.relative_to(self.path_frames)} '
                                           f'{float(i) / orb_fps} {d.relative_to(self.path_frames)}\n')

        # Generate ORB-SLAM config file
        path_orbslam2_config = self.path_reconstruction / "orbslam2_config.yaml"
        if path_orbslam2_config.exists():
            print(path_orbslam2_config, "already exists")
        else:
            orbslam2_config = \
                f"%YAML:1.0\n\n" \
                f"Camera.fx: {self.config_camera['fx']}\n" \
                f"Camera.fy: {self.config_camera['fy']}\n" \
                f"Camera.cx: {self.config_camera['cx']}\n" \
                f"Camera.cy: {self.config_camera['cy']}\n" \
                f"Camera.k1: {self.config_camera['k1']}\n" \
                f"Camera.k2: {self.config_camera['k2']}\n" \
                f"Camera.k3: {self.config_camera['k3']}\n" \
                f"Camera.p1: {self.config_camera['p1']}\n" \
                f"Camera.p2: {self.config_camera['p2']}\n" \
                f"Camera.width: {self.config_camera['w']}\n" \
                f"Camera.height: {self.config_camera['h']}\n" \
                f"Camera.fps: 30.0\n" \
                f"Camera.bf: 40.0\n" \
                f"Camera.RGB: 1\n" \
                f"ThDepth: 40.0\n" \
                f"DepthMapFactor: 1000.0\n" \
                f"ORBextractor.nFeatures: 1000\n" \
                f"ORBextractor.scaleFactor: 1.2\n" \
                f"ORBextractor.nLevels: 8\n" \
                f"ORBextractor.iniThFAST: 20\n" \
                f"ORBextractor.minThFAST: 7\n" \
                f"Viewer.KeyFrameSize: 0.05\n" \
                f"Viewer.KeyFrameLineWidth: 1\n" \
                f"Viewer.GraphLineWidth: 0.9\n" \
                f"Viewer.PointSize: 2\n" \
                f"Viewer.CameraSize: 0.08\n" \
                f"Viewer.CameraLineWidth: 3\n" \
                f"Viewer.ViewpointX: 0\n" \
                f"Viewer.ViewpointY: -0.7\n" \
                f"Viewer.ViewpointZ: -1.8\n" \
                f"Viewer.ViewpointF: 500"
            path_orbslam2_config.write_text(orbslam2_config)

        path_orbslam2_trajectory_kf = self.path_reconstruction / "KeyFrameTrajectory.txt"
        if path_orbslam2_trajectory_kf.exists():
            print(path_orbslam2_trajectory_kf, "already exists, skipping execution of ORB-SLAM2")
        else:
            args = [Path(self.config_aligner['path_orbslam2']).absolute(),
                    Path(self.config_aligner['path_orbslam2']).resolve().parent / "../../Vocabulary/ORBvoc.txt",
                    path_orbslam2_config,
                    self.path_frames,
                    path_associations]
            rc = subprocess.call(args, cwd=self.path_reconstruction)
            if rc != 0:
                print('Error running ORB-SLAM2: \n in: {} \n command: \n {} \n return code: {}'.format(
                    self.path_reconstruction, ' '.join([str(arg) for arg in args]), rc))
                return False

        if not self.path_camera_poses.exists():
            # Convert timestamps to indices in poses.txt
            data = np.genfromtxt(self.path_reconstruction / "CameraTrajectory.txt", delimiter=' ')
            data[:, 0] = np.rint(data[:, 0] * orb_fps)
            np.savetxt(self.path_camera_poses, data, fmt='%10.12f')

            data = np.genfromtxt(path_orbslam2_trajectory_kf, delimiter=' ')
            data[:, 0] = np.rint(data[:, 0] * orb_fps)
            np.savetxt(self.path_reconstruction / "poses_selected.txt", data, fmt='%10.12f')

        self.integrate_rgbd_frames(self.path_reconstruction / "poses_selected.txt")
        return True

    def integrate_rgbd_frames(self, poses_file):
        if self.path_reconstructed_cloud.exists() and \
                (self.path_reconstructed_mesh.exists() or not self.enable_extract_background_model):
            print(self.path_reconstructed_cloud, " exists, skipping depth map integration")
        else:
            print("Integrating depth maps...")
            frame_indices = np.rint(np.genfromtxt(poses_file, delimiter=' ')[:, 0])
            keyframe_trajectory = open3d.io.read_pinhole_camera_trajectory(str(poses_file))
            voxel_length = self.config_aligner['orbslam2']['tsdf_cubic_size'] / 512.0
            volume = open3d.integration.ScalableTSDFVolume(voxel_length=voxel_length,
                                                           sdf_trunc=0.04,
                                                           color_type=open3d.integration.TSDFVolumeColorType.RGB8)
            assert (len(frame_indices) == len(keyframe_trajectory.parameters))

            for i in tqdm.tqdm(range(len(keyframe_trajectory.parameters))):
                frame_index = int(frame_indices[i])
                path_rgb = self.path_frames / f"Color{frame_index:04}.png"
                path_depth = self.path_frames / f"Depth{frame_index:04}.png"
                color = open3d.io.read_image(str(path_rgb))
                depth = open3d.io.read_image(str(path_depth))
                rgbd = open3d.geometry.create_rgbd_image_from_color_and_depth(color,
                                                                              depth,
                                                                              depth_scale=1000.0,
                                                                              depth_trunc=2.5,
                                                                              convert_rgb_to_intensity=False)
                volume.integrate(rgbd, self.camera_intrinsics, keyframe_trajectory.parameters[i].extrinsic)

            print("Extract a triangle mesh from the volume and visualize it.")
            cloud = volume.extract_point_cloud()
            mesh = volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            if not self.path_reconstructed_mesh.exists():
                print("Writing file:", self.path_reconstructed_mesh)
                open3d.io.write_triangle_mesh(str(self.path_reconstructed_mesh), mesh, compressed=self.compress_ply)
            if not self.path_reconstructed_cloud.exists():
                print("Writing file:", self.path_reconstructed_cloud)
                open3d.io.write_point_cloud(str(self.path_reconstructed_cloud), cloud, compressed=self.compress_ply)
            if self.show_intermediate_visualizations:
                open3d.visualization.draw_geometries([cloud])
                open3d.visualization.draw_geometries([mesh])

    def extract_klg(self):
        print("Extracting klg:", self.path_scene_klg)
        args = [self.config_aligner['path_klg_converter'],
                '-i', self.path_scene_klg, '-o', self.path_frames, '-frames', '-png', '-depthpng', '-depthscale', '1']
        rc = subprocess.call(args)
        if rc != 0:
            print('Failed to extract klg file. Aborting: \n command: \n {} \n return code: {}'.format(
                ' '.join([str(arg) for arg in args]), rc))
            raise RuntimeError("Failed to extract klg file")

    def annotate_frames(self):
        # User interface:
        # +--------+----------+-----------+----------+-------------+----------+--------+
        # | frame: | <slider> | | object: | <slider> | brush-size: | <slider> | [save] |
        # +--------+----------+-----------+----------+-------------+----------+--------+
        # |                      <image>                                               |
        # +----------------------------------------------------------------------------+

        window = Tk()

        canvas = Canvas(window, width=self.config_camera['w'], height=self.config_camera['h'])
        canvas.grid(row=1, column=0, columnspan=7)

        global current_brush_size, current_frame_index, current_object_index
        current_brush_size = 5
        current_frame_index = 0
        current_object_index = self.objects[0].id
        current_image = PhotoImage(file=self.rgb_frames[0])
        current_canvas_image = canvas.create_image(0, 0, anchor=NW, image=current_image)
        current_pil_image = Image.new("I", (self.config_camera['w'], self.config_camera['h']))
        current_pil_draw = ImageDraw.Draw(current_pil_image)
        list_object_items = [f"{o.id}: {o.name}" for o in self.objects]

        def on_slide_frame(index):
            global current_image, current_canvas_image, current_frame_index
            current_frame_index = int(index)
            current_image = PhotoImage(file=self.rgb_frames[current_frame_index])
            current_pil_draw.rectangle([(0, 0), current_pil_image.size], fill=0)
            canvas.delete("all")
            current_canvas_image = canvas.create_image(0, 0, anchor=NW, image=current_image)

        def on_change_object_id(name):
            global current_object_index
            selected_object = self.objects[list_object_items.index(name)]
            current_object_index = selected_object.id
            print(f"Masking object {selected_object.name}({selected_object.id}) now")

        def on_slide_size(index):
            global current_brush_size
            current_brush_size = int(index)

        def on_save():
            # canvas.delete(current_canvas_image)
            # canvas.postscript(file="/tmp/aaa.ps", colormode='color')
            global current_frame_index
            path = self.path_annotations / f"{current_frame_index}.png"
            current_pil_image.save(path)
            print(f"Saving {path} ...")

        def draw_circle(x, y, w):
            global current_object_index
            colors = ['black', '#ff0000', '#71c936', '#4545ff', '#db0000', '#00db58', '#3636c9', '#db3b3b', '#36c971',
                      '#cb40ed', '#ffcc00', '#00beed', '#ff0066', '#dbbb3b', '#0000db', '#ff458f', '#8fff45', '#0000c9']
            current_pil_draw.ellipse([x - w, y - w, x + w, y + w], fill=current_object_index)
            canvas.create_oval(x - w, y - w, x + w, y + w, fill=colors[current_object_index],
                               outline=colors[current_object_index])

        global old_x, old_y
        old_x, old_y = None, None

        def draw(event):
            global old_x, old_y, current_brush_size
            if old_x and old_y:
                draw_circle(event.x, event.y, current_brush_size)
            old_x = event.x
            old_y = event.y

        def reset(event):
            global old_x, old_y
            old_x, old_y = None, None

        canvas.bind('<B1-Motion>', draw)
        canvas.bind('<ButtonRelease-1>', reset)
        Label(window, text="frame:").grid(row=0, column=0)
        slider_frame = Scale(window, from_=0, to=len(self.rgb_frames) - 1, orient=HORIZONTAL, command=on_slide_frame)
        slider_frame.grid(row=0, column=1)

        Label(window, text="object:").grid(row=0, column=2)
        list_object_var = StringVar()
        list_object_var.set(list_object_items[0])
        list_object = OptionMenu(window, list_object_var, *list_object_items, command=on_change_object_id)
        list_object.grid(row=0, column=3)

        Label(window, text="brush-size:").grid(row=0, column=4)
        slider_brush = Scale(window, from_=1, to=30, orient=HORIZONTAL, command=on_slide_size)
        slider_brush.grid(row=0, column=5)

        button = Button(window, text="Save", command=on_save)
        button.grid(row=0, column=6)

        window.mainloop()

    def compute_bounding_box(self, id, extend=0.05):
        """
        Compute a rough bounding box of the object using the annotations provided by the user.
        :return:
        """
        bound_min = np.full([3], +np.inf)
        bound_max = np.full([3], -np.inf)

        annotation_paths = [self.path_annotations / x for x in sorted(os.listdir(self.path_annotations)) if "png" in x]

        for path_annotation in annotation_paths:
            index = int(get_index(str(path_annotation)))
            path_rgb = self.path_frames / f"Color{index:04}.png"
            path_depth = self.path_frames / f"Depth{index:04}.png"
            color = open3d.io.read_image(str(path_rgb))
            depth = open3d.io.read_image(str(path_depth))
            bg_mask = np.asarray(open3d.io.read_image(str(path_annotation))) != id
            depth_np = np.asarray(depth)
            depth_np[bg_mask] = 0
            depth = open3d.geometry.Image(depth_np)
            rgbd = open3d.geometry.create_rgbd_image_from_color_and_depth(color,
                                                                          depth,
                                                                          depth_scale=1000.0,
                                                                          depth_trunc=5.0,
                                                                          convert_rgb_to_intensity=False)
            cloud = open3d.geometry.create_point_cloud_from_rgbd_image(rgbd,
                                                                       self.camera_intrinsics,
                                                                       self.trajectory.parameters[index].extrinsic)

            bound_min = np.minimum(cloud.get_min_bound(), bound_min)
            bound_max = np.maximum(cloud.get_max_bound(), bound_max)

        bound_min -= extend
        bound_max += extend
        return bound_min, bound_max

    @staticmethod
    def visualize_bounding_boxes(boxes, cloud):
        line_sets = []
        for bound_min, bound_max in boxes:
            points = [[bound_min[0], bound_min[1], bound_min[2]],
                      [bound_max[0], bound_min[1], bound_min[2]],
                      [bound_min[0], bound_max[1], bound_min[2]],
                      [bound_max[0], bound_max[1], bound_min[2]],
                      [bound_min[0], bound_min[1], bound_max[2]],
                      [bound_max[0], bound_min[1], bound_max[2]],
                      [bound_min[0], bound_max[1], bound_max[2]],
                      [bound_max[0], bound_max[1], bound_max[2]]
                      ]
            lines = [[0, 1], [0, 2], [1, 3], [2, 3],
                     [4, 5], [4, 6], [5, 7], [6, 7],
                     [0, 4], [1, 5], [2, 6], [3, 7]]
            colors = [[1, 0, 0]] * len(lines)
            line_set = open3d.geometry.LineSet()
            line_set.points = open3d.utility.Vector3dVector(points)
            line_set.lines = open3d.utility.Vector2iVector(lines)
            line_set.colors = open3d.utility.Vector3dVector(colors)
            line_sets.append(line_set)
        open3d.visualization.draw_geometries([*line_sets, cloud])


config, config_run = parse_and_load_config()
for run in config_run['run']:
    cprint(f"======== Working on: {run['input'].resolve()}", "blue")

    try:
        camera_config = get_camera_config(config_run, run)
        object_configs = get_object_configs(config_run, run)

        aligner = Aligner(config, run, camera_config, object_configs)
        success = aligner.run()
        if not success:
            cprint(f"======== Processing failed: {run['input'].resolve()}\n", "red")
        else:
            cprint(f"======== Processing succeeded: {run['input'].resolve()}\n", "green")
    except Exception as e:
        cprint(f"Error: {e}", "red")
        cprint(f"======== Processing failed: {run['input'].resolve()}\n", "red")
