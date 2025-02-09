import wsi_preprocessing as pp
import os
import json
import shutil
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from torchvision.models import ResNet50_Weights
import random

from multiprocessing import set_start_method

def create_average_embedding(base_dir, case_id, slide_name):
    case_dir = os.path.join(base_dir, case_id)
    biospecimen_dir = os.path.join(case_dir, "Biospecimen")
    tiles_dir = os.path.join(biospecimen_dir, "Tiles")
    metadata_path = os.path.join(case_dir, "aggregated_data", f'{case_id}_data.json')

    tile_csv_path = None

    if not os.path.exists(tiles_dir):
        print(f"No tiles directory found for case {case_id}.")
        return None
    
    for file in os.listdir(tiles_dir):
        if file.endswith('filtered_tiles.csv') and slide_name in file:
            tile_csv_path = os.path.join(tiles_dir, file)
            break

    if not tile_csv_path:
        print(f"No valid tile CSV found for slide {slide_name} in case {case_id}.")
        return
    
    # randomly select a subset of tiles for faster processing
    tile_df = pd.read_csv(tile_csv_path)
    tile_paths = tile_df['tile_path'].dropna().tolist()
    tile_paths = random.sample(tile_paths, min(200, len(tile_paths)))

    if len(tile_paths) == 0:
        print(f"No tiles found in {tile_csv_path}.")
        return
    
    # Load pre-trained ResNet model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.fc = nn.Identity()  # Remove classification head to extract embeddings
    resnet = resnet.to(device)
    resnet.eval()

    # Image preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Calculate embeddings for all tiles
    embeddings = []
    for tile_path in tqdm(tile_paths, desc=f"Processing tiles for slide {slide_name}", unit="tile", leave=False):
        try:
            tile_full_path = os.path.join(tiles_dir, tile_path)  # Full path to tile
            image = Image.open(tile_full_path).convert('RGB')
            input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension
            
            with torch.no_grad():
                embedding = resnet(input_tensor).squeeze(0)  # Remove batch dimension
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing tile {tile_path} in case {case_id}: {e}")
    
    # Compute the average embedding
    if len(embeddings) > 0:
        avg_embedding = torch.stack(embeddings).mean(dim=0).cpu().tolist()
    else:
        print(f"No embeddings generated for slide {slide_name} in case {case_id}.")
        return
    
    print(f"Average embedding for slide {slide_name} in case {case_id}: {avg_embedding}")
    return avg_embedding

def create_WSI_embeddings(base_dir="cases"):
    case_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for case_id in case_dirs:
        case_dir = os.path.join(base_dir, case_id)
        biospecimen_dir = os.path.join(case_dir, "Biospecimen")
        metadata_path = os.path.join(case_dir, "aggregated_data", f'{case_id}_data.json')

        if not os.path.exists(metadata_path):
            print(f"No metadata found for case {case_id}, skipping...")
            continue
        
        with open(metadata_path, 'r') as f:
            case_metadata = json.load(f)
        
        biospecimen_data = case_metadata.get('biospecimen', {}).get('biospecimen_data', [])

        # Collect svs image file names from JSON metadata file
        for sample in biospecimen_data:
            if sample['sample_type'] == "Primary Tumor":
                for slide in sample.get('slides', []):
                    image_name = slide['image_file_name']
                    embedding = create_average_embedding(base_dir, case_id, image_name)
                    slide['embedding'] = embedding
        
        with open(metadata_path, 'w') as f:
            json.dump(case_metadata, f, indent=4)
        print(f"Embeddings added for case {case_id} in the metadata.")



def preprocess_WSI_slides(base_dir="cases", hard_reset=False):
    case_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    general_metadata_dir = os.path.join(base_dir, "GENERAL_METADATA")
    os.makedirs(general_metadata_dir, exist_ok=True)
    
    if "GENERAL_METADATA" in case_dirs:
        case_dirs.remove("GENERAL_METADATA")

    for case_id in case_dirs:
        case_dir = os.path.join(base_dir, case_id)
        biospecimen_dir = os.path.join(case_dir, "Biospecimen")
        metadata_path = os.path.join(case_dir, "aggregated_data", f'{case_id}_data.json')

        if not os.path.exists(metadata_path):
            print(f"No metadata found for case {case_id}, skipping...")
            continue
        
        with open(metadata_path, 'r') as f:
            case_metadata = json.load(f)
        
        # Check for Tiles.part
        tiles_part_dir = os.path.join(biospecimen_dir, 'Tiles.part')
        if os.path.exists(tiles_part_dir):
            shutil.rmtree(tiles_part_dir)
            print(f"Partially downloaded directory {tiles_part_dir} and its contents removed successfully.")

        # If Tiles exists, skip
        tiles_dir = os.path.join(biospecimen_dir, 'Tiles')
        if os.path.exists(tiles_dir):
            if hard_reset:
                shutil.rmtree(tiles_dir)
            else:
                print(f"Skipping case {case_id} because 'Tiles' directory already exists.")
                continue

        slides_to_process = []
        slide_names_to_process = []
        
        biospecimen_data = case_metadata.get('biospecimen', {}).get('biospecimen_data', [])
        
        # Collect svs image file names from JSON metadata file
        for sample in biospecimen_data:
            if sample['sample_type'] == "Primary Tumor":
                for slide in sample.get('slides', []):
                    slide_resolution_level = slide['resolution_level']
                    # make sure the slide is at least level 3 resolution
                    if slide_resolution_level >= 3:
                        image_name = slide['image_file_name']
                        slide_path = os.path.join(biospecimen_dir, image_name)
                        if os.path.exists(slide_path):
                            slides_to_process.append(slide_path)
                            slide_names_to_process.append(image_name)
                            for file in os.listdir(os.getcwd()):
                                # check for garbage files
                                if image_name in file:
                                    file_path = os.path.join(os.getcwd(), file)
                                    if os.path.isdir(file_path):
                                        shutil.rmtree(file_path, ignore_errors=True)
                                        print(f"Removed directory: {file_path}")
                                    elif os.path.isfile(file_path):
                                        os.remove(file_path)
                                        print(f"Removed file: {file_path}")
                        else:
                            print(f"Slide {image_name} not found for case {case_id}")
        
        if not slides_to_process:
            print(f"No Primary Tumor slides found for case {case_id}")
            continue
        
        os.makedirs(tiles_part_dir, exist_ok=True)

        slide_csv_path = os.path.join(tiles_part_dir, 'slides_mpp_otsu.csv')
        consolidated_csv_path = os.path.join(tiles_part_dir, 'consolidated.csv')
        tiles_filter_path = os.path.join(tiles_part_dir, 'tiles_filter.csv')
        filtered_tiles_path = os.path.join(tiles_part_dir, 'filtered_tiles.csv')
        
        # Commence tile cutting for current case
        pp.save_slides_mpp_otsu(slides_to_process, slide_csv_path)
        
        print("SAVED OTSU MPP")

        try:
            pp.run_tiling(
                slide_csv=slide_csv_path,
                consolidated_csv=consolidated_csv_path
            )
        except Exception as e:
            print(f"Exception during tiling for case {case_id}: {e}")
        
        print("DONE TILING")

        print(slide_csv_path)
        print(tiles_filter_path)

        pp.calculate_filters(
            slide_csv_path,
            "",
            tiles_filter_path
        )

        print("DONE CALCULATING")

        # Move generated files to appropriate folders
        for file in os.listdir(os.getcwd()):
            if "TCGA" in file and ".svs" in file:
              old_path = os.path.join(os.getcwd(), file)
              new_path = os.path.join(tiles_part_dir)
              
              # Print paths for debugging
              print(f"Attempting to move: {old_path} to {new_path}")
              
              try:
                  shutil.move(old_path, new_path)
                  print(f"Moved {file} successfully.")
              except Exception as e:
                  print(f"Failed to move {file}: {e}")

        for slide_name in slide_names_to_process:
            filters_csv_path = None
            for file in os.listdir(tiles_part_dir):
                if file.endswith("filters_cleanup.csv") and slide_name in file:
                    filters_csv_path = os.path.join(tiles_part_dir, file)
                    break

            if not filters_csv_path:
                print(f"No filters_cleanup.csv found for slide {slide_name} in case {case_id}.")
                continue

            df = pd.read_csv(filters_csv_path)
            filtered_df = df[df['bg_otsu'].notnull()]
            filtered_tiles_path = filters_csv_path.replace("filters_cleanup.csv", "filtered_tiles.csv")
            filtered_df.to_csv(filtered_tiles_path, index=False)

        if os.path.exists(tiles_part_dir):
            shutil.move(tiles_part_dir, tiles_dir)
            print(f"Renamed {tiles_part_dir} to {tiles_dir}")

        print(f"Processing complete for case {case_id}")

if __name__=='__main__':
  set_start_method("spawn")
  preprocess_WSI_slides("cases")
  create_WSI_embeddings("cases") 