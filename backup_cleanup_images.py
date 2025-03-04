import os
import shutil

HARD_DRIVE_PATH = "/mnt/Windows/RESEARCH_DATA"
CASES_DIR = "/home/dave25639/Desktop/CSCE_491H/cases"

os.makedirs(HARD_DRIVE_PATH, exist_ok=True)

moved_count = 0

for case in os.listdir(CASES_DIR):
    case_path = os.path.join(CASES_DIR, case)
    biospecimen_path = os.path.join(case_path, "Biospecimen")
    tiles_path = os.path.join(biospecimen_path, "Tiles")
    
    if not os.path.isdir(case_path) or case == "GENERAL_METADATA":
        continue
    
    if os.path.exists(biospecimen_path):
        for file in os.listdir(biospecimen_path):
            if file.endswith(".svs"):
                src_file = os.path.join(biospecimen_path, file)
                dst_file = os.path.join(HARD_DRIVE_PATH, file)
                
                if os.path.islink(src_file):
                    continue

                shutil.move(src_file, dst_file)
                os.symlink(dst_file, src_file)
                moved_count += 1

print(f"{moved_count} .svs files moved and symbolic links created.")