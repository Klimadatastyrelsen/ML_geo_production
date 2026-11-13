import argparse
import json
import geopandas as gpd
from pathlib import Path
import os
import time
from copy import deepcopy
import sys # Added for hard exit on critical file errors
from ML_geo_production import model_utils
from ML_geo_production.geo_utils import filter_images_by_bounds 
from ML_geo_production.processing_utils import save_probabilities_data, save_predictions_data
import process_images
import create_diff_polygons

def parse_arguments():
    """Parses command line arguments for the script."""
    parser = argparse.ArgumentParser(description="Process multiple areas defined in a shapefile using a JSON configuration.")
    parser.add_argument("--json", type=str, required=True, help="Path to the JSON configuration file.")
    parser.add_argument("--shapefile", type=str, required=True, help="Path to the shapefile (.shp or .gpkg) containing areas to process.")
    return parser.parse_args()

def init():
    """Initialization: loads JSON config and preloads model weights"""
    args = parse_arguments()

    with open(args.json, 'r') as f:
        parsed_json = json.load(f)

    print("Preloading all model weights into main memory...")
    parsed_json["model_states"] = model_utils.preload_model_states(parsed_json["saved_models"])
    print("loading the geopackage")
    parsed_json["geopackage"] = gpd.read_file(parsed_json["geopackage"])

    print("Initialization complete.")
    return parsed_json, Path(args.shapefile)


def main():
    """Main function to orchestrate the area processing loop."""
    # Errors in loading config (FileNotFound, JSONDecodeError) will propagate and fail hard
    base_config,shapefile_path = init()
    if not shapefile_path.exists():
        # Using sys.exit(1) for a hard, clean failure on missing critical input file
        print(f"Error: Shapefile not found at {args.shapefile}")
        sys.exit(1)

    # Load the shapefile using GeoPandas - Errors will propagate and fail hard
    print(f"Loading shapefile from {shapefile_path}...")
    gdf = gpd.read_file(shapefile_path)

    print(f"Shapefile loaded with {len(gdf)} features. Starting processing loop.")

    # Get the base output folder once
    output_folder = Path(base_config.get("path_to_map_creation_output_folder", "."))
    output_folder.mkdir(parents=True, exist_ok=True) # Ensure base output folder exists

    # Iterate over each shape (row) in the GeoDataFrame
    for idx, row in gdf.iterrows():
        # 0. Get the ID for naming
        shape_id = row.get('id', row.get('tile_id', f"{idx}"))
        print(f"\n--- Processing Shape ID: {shape_id} (Index: {idx}) ---")

        # 1. Extract the extent (bounds)
        initial_bounds = row.geometry.bounds
        print(f"Shape bounds: {initial_bounds}")

        # Create a deep copy of the base config for modification
        parsed_json = deepcopy(base_config)

        # Replace the bounds value in the loaded dictionary
        parsed_json["bounds"] = initial_bounds
        # 2. Get the images that overlap with the current bounds using filter_images_by_bounds
        image_paths, _updated_bounds = filter_images_by_bounds(
            path_to_images=parsed_json["path_to_images"],
            bounds=parsed_json["bounds"],
            bounds_dict=None # Not using a pre-computed bounds dictionary here
        )

        if not image_paths:
            print(f"No overlapping images found for shape {shape_id}. Skipping.")
            continue

        print(f"Found {len(image_paths)} overlapping images.")
        parsed_json["image_paths"] = image_paths

        # 3. Call process_images.process_images_from_dict
        # Errors in processing will propagate and fail hard
        final_probability_array, final_transform, dst_crs = process_images.process_images_from_dict(parsed_json)

        # --- Post-Processing (Saving and Change Detection) ---
        print(f"Processing complete for shape {shape_id}. Running post-processing.")

        # Probability output path modification and direct replacement
        if parsed_json.get("save_probs"):
            original_probs_path = Path(parsed_json["probs_path"])
            new_probs_filename = f"{shape_id}_{original_probs_path.name}"
            final_probs_path = output_folder / new_probs_filename
            
            # Direct replacement in the dictionary
            parsed_json["probs_path"] = str(final_probs_path)
            
            print(f"Saving probabilities to disk: {parsed_json['probs_path']}")
            # Save the array to disk - Errors will propagate and fail hard
            save_probabilities_data(final_probability_array, final_transform, dst_crs, parsed_json["probs_path"])
            print(f"Saved output GeoTIFF to {parsed_json['probs_path']}")

        # Prediction output path modification and direct replacement
        if parsed_json.get("save_preds"):
            original_pred_path = Path(parsed_json["pred_path"])
            new_pred_filename = f"{shape_id}_{original_pred_path.name}"
            final_pred_path = output_folder / new_pred_filename
            
            # Direct replacement in the dictionary
            parsed_json["pred_path"] = str(final_pred_path)
            
            print(f"Saving predictions to disk: {parsed_json['pred_path']}")
            # Save the array to disk - Errors will propagate and fail hard
            save_predictions_data(final_probability_array.argmax(axis=0).astype('uint8'), final_transform, dst_crs, parsed_json["pred_path"])
            print(f"Saved output GeoTIFF to {parsed_json['pred_path']}")

        # Change Detection
        if parsed_json.get("do_change_detection"):
            print("doing change detection")
            # Compare predictions with labels from geopackage - Errors will propagate and fail hard
            diff_polygons_df = create_diff_polygons.create_diff_polygons(
                probs=final_probability_array,
                geopackage_data=parsed_json["geopackage"],
                transform=final_transform,
                crs=dst_crs,
                bounds=parsed_json["bounds"],
                change_detection_threshold=parsed_json["change_detection_threshold"],
                unknown_boarder_size=parsed_json["unknown_boarder_size"],
                buffer_in_buffer_out_size=parsed_json["buffer_in_buffer_out_size"],
            )

            # Save change detection polygons to disk
            if parsed_json.get("save_change_detection_polygons"):
                polygon_output_folder = Path(parsed_json["polygon_output_folder"])
                polygon_output_folder.mkdir(parents=True, exist_ok=True)
                final_polygon_path = polygon_output_folder / Path(f"{shape_id}_diff_polygons.shp")

                if len(diff_polygons_df) == 0:
                    print("No difference polygons found for this area.")
                else:
                    print(f"Saving difference polygons to {final_polygon_path}")
                    # Ensure the GeoDataFrame has a valid CRS before saving
                    if diff_polygons_df.crs is None and dst_crs is not None:
                         # Assign the calculated CRS if not set
                         diff_polygons_df = diff_polygons_df.set_crs(dst_crs.to_string())

                    # Save to file - Errors will propagate and fail hard
                    diff_polygons_df.to_file(final_polygon_path)
                    print(f"Saved difference polygons to {final_polygon_path}")
        else:
            print("Skipping change detection.")

if __name__ == "__main__":
    main()
