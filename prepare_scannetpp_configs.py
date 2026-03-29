import argparse
import datetime
from pathlib import Path
import sys
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Modify the download config file for ScanNet++ dataset."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="download_scannetpp.yml",
        help="Path to the download configuration YAML file.",
    )
    parser.add_argument(
        "--scannetpp_submodule_path",
        type=str,
        default="scannetpp",
        help="Path to the ScanNet++ submodule directory from which undistort.yaml will be sourced.",
    )

    parser.add_argument(
        "--new_token",
        type=str,
        required=True,
        help="New token to replace the existing one in the config file.",
    )
    parser.add_argument(
        "--new_download_scenes",
        type=str,
        required=True,
        help="Comma-separated list of scenes to include in the download (e.g., 'scene0000_00,scene0001_00').",
    )
    args = parser.parse_args()

    if not Path(args.config_path).exists():
        print(
            f"Config file {args.config_path} does not exist.\n"
            "Please make sure to put the official ScanNet++ download script and accompanying files into the root directory of this repository."
        )
        sys.exit(1)
    if not Path(args.scannetpp_submodule_path).exists():
        print(
            f"ScanNet++ submodule path {args.scannetpp_submodule_path} does not exist.\n"
            "Please make sure to initialize the ScanNet++ submodule and put the official ScanNet++ download script and accompanying files into the root directory of this repository."
        )
        sys.exit(1)

    # Load the existing config
    with Path(args.config_path).open("r") as f:
        config = yaml.safe_load(f)

    # Update the config with new values
    config["token"] = args.new_token
    config["data_root"] = "scannetpp_data"
    config["download_scenes"] = [
        scene.strip() for scene in args.new_download_scenes.split(",")
    ]
    config.pop("download_splits", None)  # Remove the download_splits key if it exists

    config.pop("default_assets", None)  # Remove the default_assets key if it exists
    config["download_options"] = ["nvs_dslr", "scans"]

    # Backup the original config file
    backup_path = (
        args.config_path + ".backup_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    )
    Path(args.config_path).rename(backup_path)

    # Save the modified config back to the file
    with Path(args.config_path).open("w") as f:
        yaml.dump(config, f)

    # Handle undistort config
    undistort_path = Path(args.scannetpp_submodule_path) / "dslr/configs/undistort.yml"
    with undistort_path.open("r") as f:
        undistort_config = yaml.safe_load(f)

    undistort_config["data_root"] = "../scannetpp_data"
    undistort_config.pop("splits", None)  # Remove the splits key if it exists
    undistort_config["scene_ids"] = config["download_scenes"]

    undistort_config["out_image_dir"] = "nerfstudio_undistorted"
    undistort_config["out_mask_dir"] = "nerfstudio_undistorted"
    undistort_config["out_transforms_path"] = "nerfstudio_undistorted/transforms.json"

    with Path("undistort.yml").open("w") as f:
        yaml.dump(undistort_config, f)


if __name__ == "__main__":
    main()
