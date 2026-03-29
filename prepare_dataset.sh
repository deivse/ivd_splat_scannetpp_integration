# General workflow
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <scannetpp_token>"
    echo "First, download the official scannet++ download script and configuration from dataset authors."
    echo "Then, run this script with your download token as the only argument."
    echo "This script will modify the download configuration, run the download, and prepare the dataset for use with ivd_splat."
    exit 1
fi

TOKEN=$1
SCENES="c5439f4607,bcd2436daf,b0a08200c9,6115eddb86,f3d64c30f8,3f15a9266d,5eb31827b7,3db0a1c8f3,40aec5fffa,9071e139d9,e7af285f7d,bde1e479ad,5748ce6f01,825d228aec,7831862f02"

# 1. Prepare download_scannetpp.yml and undistort.yml
python prepare_scannetpp_configs.py \
       --config_path download_scannetpp.yml --scannetpp_submodule_path scannetpp \
       --new_token "$TOKEN" --new_download_scenes "$SCENES" 

# 2. Run download script:
python download_scannetpp.py download_scannetpp.yml

# 3. Run undistort script in scannetpp subdir:
pushd scannetpp
python -m dslr.undistort ../undistort.yml
popd

DATA_DIR="scannetpp_data/data"
OUT_DATA_DIR="processed/scannetpp_data"
OUT_REGULAR_DIR="processed/scannetpp"
OUT_EVAL_FROM_TRAIN_DIR="processed/scannetpp_eval_on_train_set"

# 4. Copy undistorted data to processed directory
for scene_dir in $DATA_DIR/*/; do
    scene_name=$(basename "$scene_dir")
    
    mkdir -p "$OUT_DATA_DIR/$scene_name"
    mv "$scene_dir/dslr/nerfstudio_undistorted/"*.JPG "$OUT_DATA_DIR/$scene_name/"
    mv "$scene_dir/dslr/nerfstudio_undistorted/"*.json "$OUT_DATA_DIR/$scene_name/"

    mkdir -p "$OUT_DATA_DIR/$scene_name/colmap/sparse/0/"
    cp "$scene_dir/dslr/colmap/"* "$OUT_DATA_DIR/$scene_name/colmap/sparse/0/"

    mkdir -p "$OUT_DATA_DIR/$scene_name/scans/"

    mkdir -p "$OUT_REGULAR_DIR/$scene_name"
    scene_data_dir_abs=$(realpath "$OUT_DATA_DIR/$scene_name")

    echo '{"loader": "scannet++", "id": "scannet++", "scene": "'"$scene_name"'", "loader_kwargs": {"scannetpp_scene_data_path": "'"$scene_data_dir_abs"'"}}' \
        > "$OUT_REGULAR_DIR/$scene_name/nb-info.json"

    mkdir -p "$OUT_EVAL_FROM_TRAIN_DIR/$scene_name"
    echo '{"loader": "eval_on_train_set_scannet++", "id": "eval_on_train_set_scannet++", "scene": "'"$scene_name"'", "loader_kwargs": {"scannetpp_scene_data_path": "'"$scene_data_dir_abs"'"}}' \
        > "$OUT_EVAL_FROM_TRAIN_DIR/$scene_name/nb-info.json"
done

# Change coordinate system convention
python convert_pointclouds.py "$DATA_DIR" "$OUT_DATA_DIR"


