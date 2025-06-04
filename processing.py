import os
import datasets
import argparse
from tqdm import tqdm
import json


conditions = [
    "ref", "target", 
    "InstantStyle", "ReduxStyle", 
    "FillEdit", "DepthEdit", 
    "qwen_2_5_mask", "qwen_2_5_bounding_box", 
    "sam2_mask", "uniformer", 
    "foreground", "normal", "depth", "canny", "hed", "mlsd", "openpose"]

def process_dataset(dataset, save_path):
    for cond in conditions:
        if not os.path.exists(os.path.join(save_path, cond)):
            os.makedirs(os.path.join(save_path, cond))

    annotations = []
    for i in tqdm(range(0, len(dataset))):
        d = dataset[i]
        new_row = {
            "quality_assessment": d["quality_assessment"], 
            "description": d["description"], 
            "collection": d["collection"], 
            "condition": dict()}
        for cond in conditions:
            if cond in ["foreground", "normal", "depth", "canny", "hed", "mlsd", "openpose", 
                        "uniformer", "sam2_mask", "ref", "target", "DepthEdit", 
                        "qwen_2_5_mask", "qwen_2_5_bounding_box"]:
                image = d[cond]
                if image is not None:
                    image_name = f"{i}_{cond}.jpg"
                    image_save_path = os.path.join(save_path, cond, image_name)
                    image = d[cond]
                    image.save(image_save_path)

                    if cond == "qwen_2_5_mask":
                        new_row["condition"]["qwen_2_5_meta"] = d["qwen_2_5_meta"]
                        new_row["condition"]["mask"] = os.path.abspath(image_save_path)
                    elif cond == "qwen_2_5_bounding_box":
                        new_row["condition"]["qwen_2_5_meta"] = d["qwen_2_5_meta"]
                        new_row["condition"]["bbox"] = os.path.abspath(image_save_path)
                    elif cond == "ref":
                        new_row["condition"]["reference"] = os.path.abspath(image_save_path)
                    else:
                        new_row["condition"][cond] = os.path.abspath(image_save_path)
            elif cond == "FillEdit":
                images = [d[f"{cond}_image_{idx}"] for idx in range(0, 5)]
                if all([im is not None for im in images]):
                    ret = {"image_path": []}
                    for idx, im in enumerate(images):
                        image_name = f"{i}_{cond}_image_{idx}.jpg"
                        image_save_path = os.path.join(save_path, cond, image_name)
                        im.save(image_save_path)
                        ret["image_path"].append(os.path.abspath(image_save_path))
                    ret["description"] = d["FillEdit_meta"]["description"]
                    ret["name"] = d["FillEdit_meta"]["name"]
                    new_row["condition"][cond] = ret
            elif cond in ["InstantStyle", "ReduxStyle"]:
                images = [d[f"{cond}_image_{idx}"] for idx in range(0, 3)]
                styles = [d[f"{cond}_ref_{idx}"] for idx in range(0, 3)]
                if all([im is not None for im in images]) and all([im is not None for im in styles]):
                    ret = {"image_path": [], "style_path": []}
                    for idx, (im, st) in enumerate(zip(images, styles)):
                        image_name = f"{i}_{cond}_image_{idx}.jpg"
                        style_name = f"{i}_{cond}_ref_{idx}.jpg"
                        image_save_path = os.path.join(save_path, cond, image_name)
                        style_save_path = os.path.join(save_path, cond, style_name)
                        im.save(image_save_path)
                        st.save(style_save_path)
                        ret["image_path"].append(os.path.abspath(image_save_path))
                        ret["style_path"].append(os.path.abspath(style_save_path))
                    new_row["condition"][cond] = ret
        
        annotations.append(new_row)

    with open(os.path.join(save_path, "data.json"), "w") as f:
        json.dump(annotations, f, ensure_ascii=False)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=['train', 'test'])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset = datasets.load_dataset("VisualCloze/Graph200K", split=args.split)
    
    process_dataset(dataset, os.path.join(args.target_path, args.split))
    
