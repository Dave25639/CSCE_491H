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
from timm.models.vision_transformer import VisionTransformer
from autoencoder_training import Autoencoder
import ast

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

from multiprocessing import set_start_method

login(token="ENTER_TOKEN")

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

def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url

def vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        model.load_state_dict(torch.hub.load_state_dict_from_url(pretrained_url, progress=progress))
    return model

class EmbeddingGenerator:
    def __init__(self, WSI_models, methylation_models):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.WSI_models = WSI_models
        self.methylation_models = methylation_models

    def load_WSI_model(self, model_name):
        if model_name == "resnet50":
            # Load pre-trained ResNet model
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)
            model.fc = nn.Identity()  # Remove classification head to extract embeddings
            model.eval()
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif model_name == "vit_DINO":
            # Load ViT model with DINO weights
            model = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16).to(self.device)
            model.eval()
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.70322989, 0.53606487, 0.66096631], std=[0.21716536, 0.26081574, 0.20723464])
            ])
        elif model_name == "UNI_v1":
            model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
            preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
            model.eval()
        elif model_name == "UNI_v2":
            timm_kwargs = {
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }
            model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
            preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
            model.eval()
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")
        
        return model, preprocess

    def load_methylation_model(self, model_name):
        model_classes = {
            "ae": Autoencoder,
            #"vae": VariationalAutoencoder,
        }

        metadata_file = "models/model_metadata.csv"
        metadata_df = pd.read_csv(metadata_file)
        metadata = metadata_df[metadata_df["model_name"] == model_name].iloc[0].to_dict()

        model_type = metadata["encoder_type"]
        input_dim = int(metadata["input_dim"])
        latent_dim = int(metadata["latent_dim"])
        hidden_layers = ast.literal_eval(metadata["hidden_layers"])

        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")

        model = model_classes[model_type](input_dim, latent_dim, hidden_layers)
        PATH = f"models/{model_name}.pth"
        model.load_state_dict(torch.load(PATH, map_location=self.device))

        model.to(self.device)
        model.eval()

        return model

    def create_average_WSI_embedding(self, base_dir, case_id, slide_name, model, preprocess, num_tiles=200):
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
        
        tile_df = pd.read_csv(tile_csv_path)
        tile_paths = tile_df['tile_path'].dropna().tolist()
        
        # Subsample tiles based on user input (either a percentage or a fixed number)
        tile_paths = random.sample(tile_paths, min(num_tiles, len(tile_paths)))

        if len(tile_paths) == 0:
            print(f"No tiles found in {tile_csv_path}.")
            return

        # Calculate embeddings for all tiles
        embeddings = []
        for tile_path in tqdm(tile_paths, desc=f"Processing tiles for slide {slide_name}", unit="tile", leave=False):
            try:
                tile_full_path = os.path.join(tiles_dir, tile_path)
                image = Image.open(tile_full_path).convert('RGB')
                input_tensor = preprocess(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad(): 
                    embedding = model(input_tensor).squeeze(0)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing tile {tile_path} in case {case_id}: {e}")
        
        # Compute the average embedding
        if len(embeddings) > 0:
            avg_embedding = torch.stack(embeddings).mean(dim=0).cpu().tolist()
        else:
            print(f"No embeddings generated for slide {slide_name} in case {case_id}.")
            return
        
        return avg_embedding

    def create_WSI_embeddings(self, base_dir, num_tiles):
        case_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

        for model_name in self.WSI_models:
            model, preprocess = self.load_WSI_model(model_name)
            model.to(self.device)
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
                            embedding = self.create_average_WSI_embedding(base_dir, case_id, image_name, model=model, preprocess=preprocess, num_tiles=num_tiles)
                            embedding_field_name = f"embedding_{model_name}"
                            slide[embedding_field_name] = embedding
                
                with open(metadata_path, 'w') as f:
                    json.dump(case_metadata, f, indent=4)
                print(f"Embeddings added for case {case_id} in the metadata.")

    def create_methylation_embedding(self, base_dir, case_id, methylation_file_name, model):
        methylation_file_path = os.path.join(base_dir, case_id, "DNA Methylation", methylation_file_name)

        if not os.path.exists(methylation_file_path):
            raise FileNotFoundError(f"Methylation file '{methylation_file_path}' not found.")

        df = pd.read_csv(methylation_file_path, sep="\t", header=None, names=["probe", "beta_value"])

        top_cpgs_file = os.path.join('filtered_methylation_data', 'top_250k_most_variable_cpg_sites.parquet')
        top_cpgs_df = pd.read_parquet(top_cpgs_file)
        top_cpg_sites = set(top_cpgs_df['CpG_Site'])

        filtered_df = df[df["probe"].isin(top_cpg_sites)]

        beta_values = filtered_df["beta_value"].astype(float).values

        expected_probe_count = 250000
        if len(beta_values) != expected_probe_count:
            return None

        input_tensor = torch.tensor(beta_values, dtype=torch.float32).to(self.device)
        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            methylation_embedding = model.generate_embedding(input_tensor).tolist()

        return methylation_embedding

    def create_methylation_embeddings(self, base_dir):
        case_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

        top_cpgs_file = os.path.join('filtered_methylation_data', 'top_250k_most_variable_cpg_sites.parquet')
        top_cpgs_df = pd.read_parquet(top_cpgs_file)
        top_cpg_sites = set(top_cpgs_df['CpG_Site'])

        for model_name in self.methylation_models:
            model = self.load_methylation_model(model_name)
            for case_id in case_dirs:
                case_dir = os.path.join(base_dir, case_id)
                methylation_dir = os.path.join(case_dir, "DNA Methylation")
                metadata_path = os.path.join(case_dir, "aggregated_data", f'{case_id}_data.json')

                if not os.path.exists(metadata_path):
                    print(f"No metadata found for case {case_id}, skipping...")
                    continue
                
                with open(metadata_path, 'r') as f:
                    case_metadata = json.load(f)
                
                methylation_JSON_base = case_metadata.get('methylation', {})
                if methylation_JSON_base.get('has_data') is False:
                    print(f"Skipping case {case_id} due to missing data")
                    continue 

                if 'embeddings' not in methylation_JSON_base:
                    methylation_JSON_base['embeddings'] = []

                # Collect methylation file names from JSON metadata file
                for methylation_file_name in methylation_JSON_base['dna_methylation_filename']:
                    embedding = self.create_methylation_embedding(base_dir, case_id, methylation_file_name, model)
                    
                    for file_entry in methylation_JSON_base['embeddings']:
                        if file_entry['filename'] == methylation_file_name:
                            file_entry['embeddings'] = [e for e in file_entry['embeddings'] if e['model_name'] != model_name]
                    
                    embedding_entry = {
                        "model_name": model_name,
                        "embedding": embedding
                    }

                    file_entry = next((entry for entry in methylation_JSON_base['embeddings'] if entry['filename'] == methylation_file_name), None)
                    if not file_entry:
                        file_entry = {
                            "filename": methylation_file_name,
                            "embeddings": []
                        }
                        methylation_JSON_base['embeddings'].append(file_entry)
                    
                    file_entry['embeddings'].append(embedding_entry)
                
                with open(metadata_path, 'w') as f:
                    json.dump(case_metadata, f, indent=4)

                print(f"Methylation embeddings added for case {case_id} in the metadata.")

if __name__=='__main__':
    set_start_method("spawn")
    #preprocess_WSI_slides("cases")

    #base_dir = "cases_TEST"
    #WSI_models = ["resnet50", "vit_DINO", "UNI_v1", "UNI_v2"]
    #methylation_models = ["ae_normalAE"]

    base_dir = "cases"
    WSI_models = ["UNI_v1"]
    methylation_models = ["ae_normalAE"]

    embedding_generator = EmbeddingGenerator(WSI_models=WSI_models, methylation_models=methylation_models)

    embedding_generator.create_WSI_embeddings(base_dir=base_dir, num_tiles=200)
    #embedding_generator.create_methylation_embeddings(base_dir=base_dir)

    print("Embedding generation process completed.")

   