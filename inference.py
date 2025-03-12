from visualcloze import VisualClozeModel
import os
from PIL import Image
import argparse


def get_example():
    """
    An example of Virtual Try-On. 
    """
    layout_prompt = "6 images are organized into a grid of 2 rows and 3 columns, evenly spaced."
    task_prompt = "Each row shows a virtual try-on process that aims to put [IMAGE2] the clothing onto [IMAGE1] the person, producing [IMAGE3] the person wearing the new clothing."
    content_prompt = "" # There is no content prompt in virtual try-on.
    prompts = [layout_prompt, task_prompt, content_prompt]
    # Given one in-context example, the grid_h is set to 2 (one in-context example and the current query). 
    grid_h = 2
    # This task involves three images, including a person image, a cloth image, and the person wearing the new clothing, thus grid_w is set as 3.
    grid_w = 3
    grid = [
        [
            os.path.join('examples/examples/tryon/00700_00.jpg'),
            os.path.join('examples/examples/tryon/03673_00.jpg'),
            os.path.join('examples/examples/tryon/00700_00_tryon_catvton_0.jpg'),
        ],
        [
            os.path.join('examples/examples/tryon/00555_00.jpg'),
            os.path.join('examples/examples/tryon/12265_00.jpg'),
            os.path.join('examples/examples/tryon/00555_00_tryon_catvton_0.jpg'), # The target image.
        ]
    ]
    images = []
    for row in grid:
        images.append([])
        for name in row:
            images[-1].append(Image.open(name))
    images[-1][-1] = None # The target image is set as None.
    return images, prompts, grid_h, grid_w
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=384)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    images, prompts, grid_h, grid_w = get_example()

    model = VisualClozeModel(
        model_path=args.model_path, 
        resolution=args.resolution, 
        lora_rank=256
    )
    '''
    grid_h: 
    The number of in-context examples + 1. It should be set to 1 when no in-context example. 

    grid_w: 
    The number of images involved in a task. For example, it should be 2 in depth-to-image, and 3 in virtual try-on.
    '''
    model.set_grid_size(grid_h, grid_w)
    
    '''
    images: 
    List[List[PIL.Image.Image]]. A grid-layout image collection, each row represents an in-context example or the current query, where the current query should be placed in the last row. 
    The target image should be None, and the other images should be the PIL Image class (Image.Image).

    prompts: 
    List[str]. Three prompts, representing the layout prompt, task prompt, and content prompt, respectively.
    '''
    result = model.process_images(
        images, 
        prompts, 
    )[-1] # return PIL.Image.Image
    
    result.save('example.jpg')