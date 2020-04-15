# Aligner

**Aligner** is a tool to annotate the pose of known 3D objects in RGBD sequences. This information is useful to create datasets for evaluation or training purposes in domains such as object pose estimation. Its functionality is very similar to [LabelFusion](https://github.com/RobotLocomotion/LabelFusion) but aims to be more lightweight and easier to use.

The following steps are executed by Aligner automatically, asking for user-input when required:
  1. Recordings are extracted if necessary.
  2. Aligner computes the camera trajectory and a dense reconstruction.
  2. The user is asked to draw on each object in at least one frame. This annotation can be very rough and is only used to initialize the alignment later.
  3. Objects are aligned either automatically using 3D features or with user input (by selecting at least 3 correspondences) and a refinement is performed using ICP.

Note: Aligner uses a subdirectory in the output folder for each step. If a step needs to be repeated, simply delete the corresponding output files, otherwise the step is skipped during the next execution.

### Example

In order to run Aligner, you need recordings in the klg format (or already extracted) and create a description for each sequences to let the software know which objects and camera calibration is used for which recording. An example is located in `examples/sequences.toml`. Settings, such as reconstruction parameters can be controlled via `config.toml`.

```bash
# TODO: Download example recording and meshes that work out of the box here
python3 aligner.py --run example/sequences.toml --config config.toml --data example/data
```

### Useful commands:

* Remove descriptions for all sequences (similar for other files)
```bash
find . -type f -iname sequence.toml -delete
```

* Rename a parameter in descriptions for all sequences (similar for other files)
```bash
find . -type f -iname "sequence_filtered_new.toml" -exec sed -i s/poses_filtered_new.txt/poses_filtered.txt/g {} +
```

### Data

**Aligner** generates the following files:
* `<data>/alignment/` A directory containing object to world space alignments, stored as 4x4 matrices in `#.txt` (where # is the object id).
* `<data>/annotations/` A directory containing `#.png` (16bit, # is frame index) frames which store the user provided masks.
* `<data>/frames/` A directory with color `Color####.png` and depth `Depth####.png` (16bit, , #### is frame index, in mm) frames.
* `<data>/reconstruction/` A directory containing `model.ply` and `poses.txt`.
* `<data>/recordings/` We usually place input sequences here, such as `example-1.klg`.


### Workflow recommendations

1. Only extract klg files and reconstruct trajectory and geometry (run in background for a while)
    ```
    enable_annotate_objects = false
    ```
2. Only annotate objects
    ```
    show_intermediate_visualizations = false
    enable_annotate_objects = true
    enable_align_objects = false
    ```
3. Only align objects
    ```
    enable_align_objects = true
    ```

### Dependencies
* [Open3D](http://www.open3d.org/)
* [convert_klg](https://github.com/martinruenz/dataset-tools)
* [ElasticFusion](https://github.com/mp3guy/ElasticFusion.git) (or ORBSLAM2)
* [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) (or ElasticFusion, see install script below for applying patches)
    + Apply [#709](https://github.com/raulmur/ORB_SLAM2/pull/709.diff)
    + Apply [#585](https://github.com/raulmur/ORB_SLAM2/pull/585.diff)

## Setup

### Build script
```
git clone https://github.com/martinruenz/aligner.git
cd aligner

# dependencies
sudo apt install libsuitesparse-dev libudev-dev libjpeg-dev libusb-1.0-0-dev python3-tk liblz4-dev openctm-tools


mkdir third_party
cd third_party

# ElasticFusion
git clone https://github.com/mp3guy/ElasticFusion.git
cd ElasticFusion
mkdir -p deps
cd deps
## Pangolin
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build
cd build
cmake ../ -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON
make -j8
cd ../..
## OpenNI2
git clone https://github.com/occipital/OpenNI2.git
cd OpenNI2
make -j8
cd ..
## EF
cd ../Core
mkdir build
cd build
cmake ../src
make -j8
cd ../../GUI
mkdir build
cd build
cmake ../src
make -j8
cd ../..

# ORB-SLAM2
git clone https://github.com/raulmur/ORB_SLAM2.git
cd ORB_SLAM2
wget -qO - https://github.com/raulmur/ORB_SLAM2/pull/585.diff | patch -p1
wget -qO - https://github.com/raulmur/ORB_SLAM2/pull/709.diff | patch -p1
./build.sh
cd ..

# Leave third_party
cd ..

# Collect symlinks to tools in ./third_party/bin
cd third_party
mkdir bin
cd bin
ln -s ../ElasticFusion/GUI/build/ElasticFusion ./
ln -s ../ORB_SLAM2/Examples/RGB-D/rgbd_tum orbslam2_rgbd_tum
cd ../..
```

## Todos

* Allow manual cropping, see: http://www.open3d.org/docs/tutorial/Advanced/interactive_visualization.html
* Figure out why `Bad Window (invalid Window parameter)?` is occurring (related to TK window, not cleaning up properly?)
* Add screenshots
* Add script in `example` that downloads example data to get starting as quickly as possible.

## Acknowledgements
This work has been supported by the SecondHands project, funded from the EU Horizon 2020 Research and Innovation programme under grant agreement No 643950.
