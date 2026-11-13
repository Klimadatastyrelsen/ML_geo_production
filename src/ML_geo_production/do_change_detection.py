import argparse
from pathlib import Path
import geopandas as gpd
import json
from ML_geo_production import create_diff_polygons, model_utils, process_images

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run ensemble model inference using a JSON configuration file and direct image paths."
    )
    parser.add_argument("--json", type=str, required=True, help="Path to the JSON configuration file.")
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
    return parsed_json


def main():
    #Preload all model weights into main memory and store weights in parsed_json["model_states"]
    #load the geopackage and store the laoded geodata in parsed_json["geopackage"]
    parsed_json = init()

    # predict using ML model ensamble
    final_probability_array, final_transform, dst_crs = process_images.process_images_from_dict(parsed_json)

    #save probabilities to disk ?
    # not nececeary for change detection but convinient for debugging
    if parsed_json.get("save_probs"):
        # Save the array to disk
        save_output_data(final_probability_array, final_transform, dst_crs, parsed_json["probs_path"])
        print(f"Saved output GeoTIFF to {parsed_json['probs_path']}")
    #save predictions to disk ?
    # not nececeary for change detection but convinient for debugging
    if parsed_json.get("save_preds"):
        save_output_data(final_probability_array.argmax(), final_transform, dst_crs, parsed_json["pred_path"])
        print(f"Saved output polygons to {parsed_json['pred_path']}")

    #do change detection
    #when doing change detection the value for 'do_change_detection' in the json should always be true
    assert(parsed_json.get("do_change_detection"))
    if parsed_json.get("do_change_detection"):
        print("doing change detection")
        # Compare predictions with labels from geopackage
        diff_polygons_df = create_diff_polygons.create_diff_polygons(
            probs=final_probability_array,
            geopackage_data = parsed_json["geopackage"],
            transform = final_transform,
            crs = dst_crs,
            bounds=  parsed_json["bounds"],
            change_detection_threshold=parsed_json["change_detection_threshold"],
            unknown_boarder_size=parsed_json["unknown_boarder_size"],
            buffer_in_buffer_out_size=parsed_json["buffer_in_buffer_out_size"],
            )
        #save change detection polygons to disk
        #
        if parsed_json.get("save_change_detection_polygons"):
                if len(diff_polygons_df) == 0:
                    print("no polygons so we dont save them to file ")
                else:

                    Path(parsed_json["polygon_output_folder"]).mkdir(parents=True, exist_ok=True)
                    polygon_output_path = Path(parsed_json["polygon_output_folder"]) / Path("diff_polygons.shp")

                    print(f"Saving difference polygons to {polygon_output_path}")

                    diff_polygons_df.to_file(polygon_output_path)
                    print(f"Saved difference polygons to {polygon_output_path}")




if __name__ == "__main__":
    main()
