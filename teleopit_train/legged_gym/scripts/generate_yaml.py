import os
import yaml

def generate_config_yaml(folder_path, output_yaml_path):
    # Specify the root path
    config = {
        "root_path": folder_path,
        "motions": []
    }

    # List all .pkl files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pkl"):
            # Append each .pkl file with default settings to the motions list
            config["motions"].append({
                "file": file_name,
                "weight": 1.0,
                "description": "general movement"
            })

    # Write the configuration to the YAML file
    with open(output_yaml_path, "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False, sort_keys=False)

# Specify your folder path and output YAML file
folder_path = "/home/zixuan/projects/humanoid-motion-imitation/legged_gym/motion_data/g1_pkl/locomanip_debug"
output_yaml_path = "g1_locomanip200.yaml"

generate_config_yaml(folder_path, output_yaml_path)
print(f"YAML configuration file generated: {output_yaml_path}")