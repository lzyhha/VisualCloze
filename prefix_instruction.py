import random

condition_list = ["canny", "depth_anything_v2", "hed", "midas_depth_normal", "mlsd", "openpose", "sam2_mask", "uniformer"]
lige_2x1_instruction = {
    "canny": "A pair of side by side images, illustrating a canny edge based image generation task.",
    "depth_anything_v2": "A pair of side by side images, illustrating a depth based image generation task.",
    "hed": "A pair of side by side images, illustrating a hed edge based image generation task.",
    "midas_depth_normal": "A pair of side by side images, illustrating a midas normal map based image generation task.",
    "mlsd": "A pair of side by side images, illustrating a mlsd line based image generation task.",
    "openpose": "A pair of side by side images, illustrating a human pose key point based image generation task.",
    "sam2_mask": "A pair of side by side images, illustrating a segmentation mask based image generation task.",
    "uniformer": "A pair of side by side images, illustrating a segmentation mask based image generation task.",
}

lige_3x1_instruction = {
    "canny": "Three side by side images, illustrating an image generation task with canny edge condition and subject reference.",
    "depth_anything_v2": "Three side by side images, illustrating an image generation task with depth condition and subject reference.",
    "hed": "Three side by side images, illustrating an image generation task with hed edge condition and subject reference.",
    "midas_depth_normal": "Three side by side images, illustrating an image generation task with midas normal map condition and subject reference.",
    "mlsd": "Three side by side images, illustrating an image generation task with mlsd line condition and subject reference.",
    "openpose": "Three side by side images, illustrating an image generation task with human pose key point condition and subject reference.",
    "sam2_mask": "Three side by side images, illustrating an image generation task with segmentation mask condition and subject reference.",
    "uniformer": "Three side by side images, illustrating an image generation task with segmentation mask condition and subject reference.",
}

#########
condition_list = ["canny", "depth_anything_v2", "hed", "midas_depth_normal", "mlsd", "openpose", "sam2_mask", "uniformer"]
degradation_list = [
    # blur
    "blur",
    "compression",
    "SRx2",
    "SRx4",
    "pixelate",
    "Defocus",
    "GaussianBlur",
    # sharpen
    "oversharpen",
    # nosie
    "GaussianNoise",
    "PoissonNoise",
    "SPNoise",
    # mosaic
    "mosaic",
    # contrast
    "contrast_strengthen",
    "contrast_weaken",
    # quantization
    "quantization",
    "JPEG",
    # light
    "brighten",
    "darken",
    "LowLight",
    # color
    "saturate_strengthen",
    "saturate_weaken",
    "gray",
    "ColorDistortion",
    # infilling
    "Inpainting",
    # rotate
    "rotate90",
    "rotate180",
    "rotate270",
    # other
    "Barrel",
    "Pincushion",
    "Elastic",
    # spacial effect
    "Rain",
    "Frost",
    ]

def get_image_prompt(image_type):
    image_prompts = {
        "target": [
            "a high-quality image",
            "an aesthetically pleasing photograph",
            "a high-resolution image",
            "an image with vivid details",
            "a visually striking and clear picture",
            "a high-definition image",
            "an image with artistic appeal",
            "a sharp and beautifully composed photograph",
            "a high-aesthetic image",
            "an image with flawless clarity",
            "a vibrant and professionally captured photo",
            "a crystal-clear image",
            "an image with artistic quality"
            "a high-quality image with exceptional detail",
            "a photo realistic image",
        ],
        "reference": [
            "a reference image",
            "an image featuring the primary object"
            "a reference for the main object",
            "a reference image highlighting the central object",
            "an image containing the key object",
            "a reference image with the main subject included",
            "an image providing the main object",
            "a reference image showcasing the dominant object",
            "an image that includes the main object",
            "a reference image capturing the primary subject",
            "an image containing the main subject",
        ],
        # condition
        "canny": [
            "canny edge map with sharp black-and-white contours",
            "black-and-white edge map highlighting crisp boundaries",
            "canny result showing stark white edges on black",
            "edge map with clean white lines on a dark background",
            "canny output featuring precise white object outlines",
            "black background with white edge-detected contours",
            "canny edge map displaying clear white structural edges",
            "white edge lines on black from canny detection",
            "canny map with sharp white edges and dark voids",
            "edge map revealing white outlines of object shapes",
        ],
        "depth_anything_v2": [
            "depth map showing gray-scale object contours",
            "gray-toned depth map with layered outlines",
            "depth map featuring gradient-gray surfaces",
            "gray-shaded depth map with distinct edges",
            "depth map displaying soft gray gradients",
            "gray-scale depth map with clear object boundaries",
            "depth map highlighting gray-level depth variations",
            "gray-textured depth map with smooth transitions",
            "depth map revealing gray-toned spatial layers",
            "gray-based depth map with detailed object contours",
        ],
        "hed": [
            "hed edge map with smooth flowing contours",
            "soft-edged map from hed detection",
            "hed result showing refined continuous edges",
            "edge map with natural well-connected outlines",
            "hed output featuring smooth detailed boundaries",
            "elegant edge map with seamless transitions",
            "hed edge map displaying clean holistic contours",
            "refined edge lines from hed detection",
            "hed map with flowing natural object outlines",
            "edge map revealing smooth interconnected shapes",
        ],
        "midas_depth_normal": [
            "normal map showing surface orientation details",
            "rgb-coded normal map for 3D lighting",
            "normal map with encoded surface normals",
            "detailed normal map for texture shading",
            "normal map highlighting surface curvature",
            "rgb normal map for bump mapping effects",
            "normal map capturing fine geometric details",
            "surface normal visualization in rgb colors",
            "normal map for realistic lighting interaction",
            "normal map displaying directional surface data",
        ],
        "mlsd": [
            "mlsd detected straight line segments",
            "line segments extracted using mlsd",
            "mlsd output showing precise straight lines",
            "straight edges detected by mlsd algorithm",
            "mlsd result with clean line segment boundaries",
            "line segment map generated by mlsd",
            "mlsd-detected straight structural lines",
            "straight line visualization from mlsd",
            "mlsd-based line segment detection output",
            "line segments highlighted by mlsd method",
        ],
        "openpose": [
            "openpose skeleton with colorful connecting lines",
            "body keypoints linked by bright colored lines",
            "openpose output showing joints and vibrant skeleton",
            "human pose with colored lines for bone structure",
            "openpose-detected keypoints and colorful limbs",
            "skeletal lines in vivid colors from openpose",
            "body joints connected by multicolored straight lines",
            "openpose visualization with colorful skeletal links",
            "keypoints and bright lines forming body skeleton",
            "human pose mapped with colored lines by openpose",
        ],
        "sam2_mask": [
            "sam 2 generated colorful segmentation masks",
            "color-coded masks from sam 2 segmentation",
            "sam 2 output with vibrant object masks",
            "segmentation masks in bright colors by sam 2",
            "colorful object masks from sam 2 detection",
            "sam 2 result showing multicolored regions",
            "masks with distinct colors from sam 2",
            "sam 2 segmentation with vivid mask overlays",
            "colorful masks highlighting objects via sam 2",
            "sam 2-generated masks with rich color coding",
        ],
        "uniformer": [
            "color-coded objects in uniformer segmentation",
            "uniformer map with colored object blocks",
            "objects as distinct color patches by uniformer",
            "color blocks representing objects in uniformer",
            "uniformer output with colored object regions",
            "objects highlighted as color zones in uniformer",
            "uniformer segmentation showing color-divided objects",
            "color patches for objects in uniformer result",
            "uniformer map with objects as solid color areas",
            "objects segmented as colored blocks by uniformer",
            "uniformer map with objects as solid color areas",
        ],
        # Blur
        "blur": [
            "a softly blurred image with smooth transitions",
            "a photograph with a gentle motion blur effect",
            "an image exhibiting subtle Gaussian blur",
            "a picture with a light and even blurring",
            "a softly defocused photograph with reduced sharpness",
            "an image featuring a mild blur for artistic effect",
            "a photograph with a gentle out-of-focus appearance",
            "a softly smeared image with smooth edges",
            "a picture with a light blur enhancing the mood",
            "an image with a delicate blur creating a dreamy effect",
        ],
        "compression": [
            "a highly compressed image with noticeable artifacts",
            "a photograph showing compression-induced quality loss",
            "an image with visible compression artifacts and reduced clarity",
            "a picture exhibiting blocky artifacts from compression",
            "a compressed photo with color banding and loss of detail",
            "an image displaying noticeable compression noise",
            "a photograph with degraded quality due to high compression",
            "a picture showing pixelation from aggressive compression",
            "an image with artifacts and reduced resolution from compression",
            "a compressed image featuring loss of sharpness and detail",
        ],
        "SRx2": [
            "an image downsampled by a factor of 2 with enhanced details",
            "a photograph resized to half its original resolution",
            "an downscaled image (2x) maintaining image quality",
            "a picture downsized by 2x with preserved sharpness",
            "an image scaled half its size with clear details",
            "a low-resolution version of the original image (2x)",
            "a half-resolution photograph with maintained clarity",
            "an image decreased in size by 2x with minimal quality loss",
            "a 2x downscaled picture retaining original details",
            "an image resized to half its original dimensions with enhanced quality",
        ],
        "SRx4": [
            "an image downsampled by a factor of 4 with enhanced details",
            "a photograph resized to quarter its original resolution",
            "an downscaled image (4x) maintaining image quality",
            "a picture downsized by 4x with preserved sharpness",
            "an image scaled four times its size with clear details",
            "a low-resolution version of the original image (4x)",
            "a quadruple-resolution photograph with maintained clarity",
            "an image decreased in size by 4x with minimal quality loss",
            "a 4x downscaled picture retaining original details",
            "an image resized to quarter its original dimensions with enhanced quality",
        ],
        "pixelate": [
            "a heavily pixelated image with large blocks",
            "a picture showing strong pixelation effects",
            "an image with noticeable pixel blocks obscuring details",
            "a pixelated photograph with reduced image clarity",
            "an image exhibiting coarse pixelation for a stylized look",
            "a picture with large pixel squares creating a mosaic effect",
            "a highly pixelated photo obscuring fine details",
            "an image featuring prominent pixelation and blockiness",
            "a pixelated image with distinct square blocks",
            "a photograph with exaggerated pixelation for artistic effect",
        ],
        "Defocus": [
            "a defocused image with soft and blurry regions",
            "a photograph with intentional defocus creating a shallow depth of field",
            "an image exhibiting a defocused background with a clear subject",
            "a picture with selective defocus enhancing the main object",
            "a defocused photo with smooth out-of-focus areas",
            "an image showing a defocused effect for artistic blurring",
            "a photograph with a softly defocused foreground",
            "a picture with partial defocus creating a dreamy appearance",
            "an image featuring defocus to highlight specific areas",
            "a defocused photograph with gentle blurring around the subject",
        ],
        "GaussianBlur": [
            "an image with Gaussian blurring creating a soft focus effect",
            "a photograph with a Gaussian blur enhancing the subject",
            "a picture with Gaussian blurring to highlight the main object",
            "an image featuring Gaussian blur to soften the background",
            "a Gaussian-blurred photograph with a soft focus",
            "a Gaussian-blurred image with a gentle focus on the subject",
            "a picture with Gaussian blurring to emphasize the main subject",
            "an image with Gaussian blurring to create a dreamy effect",
            "a Gaussian-blurred photograph with a soft focus on the main object",
        ],
        # Sharpen
        "oversharpen": [
            "an image with excessive sharpening creating halos around edges",
            "a photograph overly sharpened with exaggerated edge contrast",
            "an oversharpened picture showing unnatural edge highlights",
            "a highly sharpened image with pronounced texture details",
            "a picture exhibiting over-sharpening with visible artifacts",
            "an image with extreme sharpening enhancing all details sharply",
            "a photograph with oversharpened edges and increased contrast",
            "an overly sharpened image causing unnatural texture emphasis",
            "a picture with excessive sharpening effects on all elements",
            "an image displaying over-sharpened features with enhanced edges",
        ],
        # Noise
        "GaussianNoise": [
            "an image with subtle Gaussian noise adding grain",
            "a photograph exhibiting Gaussian noise for a textured look",
            "a picture with light Gaussian noise enhancing realism",
            "an image featuring Gaussian noise with smooth distribution",
            "a photo with added Gaussian noise creating a grainy effect",
            "an image showing gentle Gaussian noise for artistic texture",
            "a photograph with mild Gaussian noise increasing depth",
            "a picture with soft Gaussian noise enhancing the image",
            "an image displaying Gaussian noise for a vintage feel",
            "a photo with Gaussian noise subtly integrated into the image",
        ],
        "PoissonNoise": [
            "an image with Poisson noise creating photon distribution effects",
            "a photograph exhibiting Poisson noise for realistic grain",
            "a picture with added Poisson noise enhancing texture",
            "an image featuring Poisson noise with natural variance",
            "a photo with Poisson noise simulating low-light conditions",
            "an image showing Poisson noise for authentic grain patterns",
            "a photograph with mild Poisson noise increasing image depth",
            "a picture with Poisson noise adding subtle texture",
            "an image displaying Poisson noise for a realistic appearance",
            "a photo with Poisson noise integrated for enhanced realism",
        ],
        "SPNoise": [
            "an image with salt and pepper noise introducing random pixels",
            "a photograph exhibiting SP noise with black and white speckles",
            "a picture with added salt and pepper noise creating scattered dots",
            "an image featuring SP noise with random pixel disruptions",
            "a photo with SP noise simulating transmission errors",
            "an image showing salt and pepper noise for a gritty effect",
            "a photograph with mild SP noise adding texture variation",
            "a picture with SP noise introducing random black and white pixels",
            "an image displaying salt and pepper noise for a distressed look",
            "a photo with SP noise integrated for a speckled appearance",
        ],
        # Mosaic
        "mosaic": [
            "an image with a strong mosaic effect obscuring details",
            "a photograph exhibiting mosaic patterns with large tiles",
            "a picture with applied mosaic effect creating a tiled appearance",
            "an image featuring mosaic blocks for privacy masking",
            "a photo with mosaic segmentation highlighting regions",
            "an image showing a mosaic overlay for abstract effect",
            "a photograph with mosaic patterns simplifying the image",
            "a picture with a mosaic filter creating geometric tiles",
            "an image displaying a mosaic effect for stylistic purposes",
            "a photo with mosaic segmentation emphasizing specific areas",
        ],
        # Contrast
        "contrast_strengthen": [
            "an image with enhanced contrast making colors pop",
            "a photograph exhibiting strengthened contrast for vividness",
            "a picture with increased contrast highlighting details",
            "an image featuring heightened contrast for dramatic effect",
            "a photo with boosted contrast enhancing visual depth",
            "an image showing strengthened contrast with pronounced shadows and highlights",
            "a photograph with amplified contrast for greater clarity",
            "a picture with enhanced contrast making elements stand out",
            "an image displaying increased contrast for a striking appearance",
            "a photo with reinforced contrast improving overall image impact",
        ],
        "contrast_weaken": [
            "an image with reduced contrast creating a softer look",
            "a photograph exhibiting weakened contrast for a muted effect",
            "a picture with decreased contrast making colors more subtle",
            "an image featuring lowered contrast for a gentle appearance",
            "a photo with diminished contrast softening the overall image",
            "an image showing weakened contrast with less pronounced shadows and highlights",
            "a photograph with reduced contrast for a flatter visual tone",
            "a picture with softened contrast creating a delicate atmosphere",
            "an image displaying decreased contrast for a subdued look",
            "a photo with lowered contrast enhancing a calm and serene feel",
        ],
        # Quantization
        "quantization": [
            "an image with quantization artifacts reducing color depth",
            "a photograph exhibiting quantization leading to banding effects",
            "a picture with applied quantization simplifying color gradients",
            "an image featuring quantized color levels creating discrete steps",
            "a photo with quantization reducing the number of distinct colors",
            "an image showing quantization leading to posterization effects",
            "a photograph with quantized color palette for a stylized look",
            "a picture with quantization introducing color banding and loss of detail",
            "an image displaying quantization effects on smooth color transitions",
            "a photo with quantization artifacts simplifying the overall color scheme",
        ],
        "JPEG": [
            "a JPEG-compressed image with noticeable compression artifacts",
            "a photograph saved in JPEG format showing quality loss",
            "an image exhibiting JPEG artifacts like blockiness and blurring",
            "a picture with JPEG compression leading to reduced clarity",
            "an image featuring JPEG-induced artifacts affecting image quality",
            "a photo with visible JPEG compression effects on details",
            "an image showing JPEG artifacts such as color banding and pixelation",
            "a photograph with degraded quality due to JPEG compression",
            "a picture with JPEG compression artifacts impacting the overall appearance",
            "an image displaying JPEG-induced quality loss with blurred edges",
        ],
        # Light
        "brighten": [
            "a brightly lit image with enhanced luminosity",
            "a photograph exhibiting increased brightness for a vibrant look",
            "a picture with boosted brightness making the scene more radiant",
            "an image featuring heightened brightness illuminating all areas",
            "a photo with amplified brightness creating a sunny appearance",
            "an image showing increased brightness enhancing visibility",
            "a photograph with enhanced brightness making colors more vivid",
            "a picture with boosted luminosity brightening the overall image",
            "an image displaying heightened brightness for a luminous effect",
            "a photo with increased brightness adding warmth and clarity",
        ],
        "darken": [
            "a darkened image with reduced luminosity creating a moody atmosphere",
            "a photograph exhibiting decreased brightness for a subdued look",
            "a picture with lowered brightness making the scene more somber",
            "an image featuring diminished brightness enhancing shadows",
            "a photo with reduced brightness creating a twilight appearance",
            "an image showing decreased brightness adding depth and contrast",
            "a photograph with darkened tones making colors more muted",
            "a picture with lowered luminosity creating a dramatic effect",
            "an image displaying reduced brightness for a darker aesthetic",
            "a photo with decreased brightness enhancing the mysterious mood",
        ],
        "LowLight": [
            "an image with low light conditions creating a dim and shadowy appearance",
            "a photograph exhibiting low light to simulate night-time conditions",
            "a picture with reduced illumination to create a night-time ambiance",
            "an image featuring low light to emphasize the subject in darkness",
            "a photo with low light conditions creating a mysterious mood",
            "an image showing low light to enhance the dramatic lighting of the scene",
            "a photograph with dim lighting to create a soft and dreamy effect",
            "a picture with low light to emphasize the texture and details of the image",
            "an image displaying low light conditions for a serene and peaceful feel",
        ],
        # Color
        "saturate_strengthen": [
            "an image with enhanced saturation making colors more vivid",
            "a photograph exhibiting strengthened saturation for vibrant hues",
            "a picture with boosted color saturation enhancing visual appeal",
            "an image featuring heightened saturation creating rich color tones",
            "a photo with amplified saturation making colors pop",
            "an image showing increased saturation for a lively appearance",
            "a photograph with saturated colors enhancing the overall image",
            "a picture with strengthened color saturation adding vibrancy",
            "an image displaying enhanced saturation for a dynamic look",
            "a photo with boosted color intensity making the scene more colorful",
        ],
        "saturate_weaken": [
            "an image with reduced saturation creating a muted color palette",
            "a photograph exhibiting weakened saturation for subdued tones",
            "a picture with lowered color saturation making colors more subtle",
            "an image featuring diminished saturation creating a pastel look",
            "a photo with decreased saturation softening the overall colors",
            "an image showing reduced saturation for a faded appearance",
            "a photograph with desaturated colors enhancing a minimalist aesthetic",
            "a picture with weakened color saturation adding a calm feel",
            "an image displaying lowered saturation for a gentle color scheme",
            "a photo with diminished color intensity creating a subdued look",
        ],
        "gray": [
            "a grayscale image with varying shades of gray",
            "a black and white photograph emphasizing contrast and texture",
            "a gray-toned picture highlighting light and shadow",
            "an image converted to grayscale showcasing structural details",
            "a monochromatic photo with rich gray gradients",
            "a grayscale image emphasizing form and composition",
            "a black and white picture with balanced gray tones",
            "an image in gray scale enhancing depth and dimension",
            "a monochrome photograph focusing on texture and contrast",
            "a gray-toned image presenting a classic black and white aesthetic",
        ],
        "ColorDistortion": [
            "an image with distorted and surreal colors",
            "a picture featuring unnatural color tones",
            "a visually striking image with altered hues",
            "a photo showcasing disrupted color balance",
            "an image with vibrant and unexpected colors",
            "a picture displaying shifted color spectrums",
            "an artwork-like image with perturbed colors",
            "a photo with dreamlike and distorted hues",
            "an image with unconventional color variations",
            "a visually unique picture with color shifts",
        ],
        # Infilling
        "Inpainting": [
            "an inpainted image seamlessly filling missing areas",
            "a photograph with inpainting repairing damaged regions",
            "a picture featuring inpainting to restore obscured parts",
            "an image using inpainting to complete incomplete areas",
            "a photo with inpainting blending filled regions naturally",
            "an image showing inpainting techniques removing unwanted objects",
            "a photograph with inpainting reconstructing missing details",
            "a picture utilizing inpainting to enhance image continuity",
            "an image with inpainting seamlessly integrating filled sections",
            "a photo using inpainting to mend and complete the visual content",
        ],
        # Rotate
        "rotate90": [
            "an image rotated 90 degrees clockwise for a new perspective",
            "a photograph turned 90 degrees to the right altering the orientation",
            "a picture rotated a quarter turn clockwise enhancing composition",
            "an image featuring a 90-degree rotation adjusting the viewpoint",
            "a photo with a 90-degree clockwise rotation changing the layout",
            "an image showing a rotated view at 90 degrees for a fresh angle",
            "a photograph rotated right by 90 degrees for dynamic framing",
            "a picture with a 90-degree turn clockwise modifying the scene",
            "an image displaying a 90-degree rotated orientation for visual interest",
            "a photo rotated ninety degrees to enhance the composition",
        ],
        "rotate180": [
            "an image rotated 180 degrees flipping it upside down",
            "a photograph turned completely around with a 180-degree rotation",
            "a picture rotated halfway, creating an inverted perspective",
            "an image featuring a 180-degree turn altering the original orientation",
            "a photo with an upside-down view due to 180-degree rotation",
            "an image showing a flipped perspective with a 180-degree rotation",
            "a photograph rotated twice around, changing the viewpoint",
            "a picture with a half-turn rotation modifying the scene layout",
            "an image displaying a 180-degree rotated orientation for a unique angle",
            "a photo rotated one full half-circle to invert the composition",
        ],
        "rotate270": [
            "an image rotated 270 degrees clockwise for a new angle",
            "a photograph turned 270 degrees to the right altering the orientation",
            "a picture rotated three quarters turn clockwise enhancing composition",
            "an image featuring a 270-degree rotation adjusting the viewpoint",
            "a photo with a 270-degree clockwise rotation changing the layout",
            "an image showing a rotated view at 270 degrees for a fresh angle",
            "a photograph rotated right by 270 degrees for dynamic framing",
            "a picture with a 270-degree turn clockwise modifying the scene",
            "an image displaying a 270-degree rotated orientation for visual interest",
            "a photo rotated two and a half turns clockwise to enhance the composition",
        ],
        # Other
        "Barrel": [
            "an image with barrel distortion bending the edges outward",
            "a photograph exhibiting barrel distortion creating a convex effect",
            "a picture with barrel distortion warping the image edges",
            "an image featuring barrel distortion causing peripheral stretching",
            "a photo with barrel distortion curving the sides outward",
            "an image showing barrel distortion for a fisheye lens effect",
            "a photograph with warped edges due to barrel distortion",
            "a picture with barrel distortion altering the straight lines",
            "an image displaying barrel distortion creating a rounded appearance",
            "a photo with barrel distortion enhancing the central focus",
        ],
        "Pincushion": [
            "an image with pincushion distortion bending the edges inward",
            "a photograph exhibiting pincushion distortion creating a concave effect",
            "a picture with pincushion distortion warping the image edges inward",
            "an image featuring pincushion distortion causing peripheral compression",
            "a photo with pincushion distortion curving the sides inward",
            "an image showing pincushion distortion for a telephoto lens effect",
            "a photograph with warped edges due to pincushion distortion",
            "a picture with pincushion distortion altering the straight lines inward",
            "an image displaying pincushion distortion creating a pinched appearance",
            "a photo with pincushion distortion enhancing the central focus inward",
        ],
        "Elastic": [
            "an image with elastic deformation creating fluid distortions",
            "a photograph exhibiting elastic transformations warping the structure",
            "a picture with elastic effects bending and stretching elements",
            "an image featuring elastic distortions for a dynamic appearance",
            "a photo with elastic transformations altering the image geometry",
            "an image showing elastic deformation for a fluid, wavy effect",
            "a photograph with elastic warping adding motion-like distortions",
            "a picture with elastic effects creating flexible and dynamic shapes",
            "an image displaying elastic transformations enhancing creative distortion",
            "a photo with elastic deformation modifying the original image structure",
        ],
        # Spatial Effect
        "Rain": [
            "an image with realistic rain effects adding dynamic streaks",
            "a photograph exhibiting rain overlays creating a wet atmosphere",
            "a picture with rain effects enhancing the scene with falling droplets",
            "an image featuring rain streaks adding motion and mood",
            "a photo with simulated rain creating a rainy day ambiance",
            "an image showing rain effects with dynamic water droplets",
            "a photograph with rain overlays adding a sense of movement",
            "a picture with rain effects enhancing the visual texture",
            "an image displaying rain streaks for a dramatic weather effect",
            "a photo with realistic rain adding depth and atmosphere",
        ],
        "Frost": [
            "an image with frost overlays creating icy textures",
            "a photograph exhibiting frost effects adding a chilly ambiance",
            "a picture with frost patterns enhancing the scene with icy details",
            "an image featuring frost overlays creating a frozen appearance",
            "a photo with simulated frost adding a wintry atmosphere",
            "an image showing frost effects with delicate ice patterns",
            "a photograph with frost overlays adding a sense of coldness",
            "a picture with frost effects enhancing the visual texture with ice",
            "an image displaying frost patterns for a frosty weather effect",
            "a photo with realistic frost adding depth and a chilly mood",
        ],
    }
    return random.choice(image_prompts[image_type])


def get_layout_instruction(cols, rows):
    layout_instruction = [
        f"A grid layout with {rows} rows and {cols} columns, displaying {cols*rows} images arranged side by side.",
        f"{cols*rows} images are organized into a grid of {rows} rows and {cols} columns, evenly spaced.",
        f"A {rows}x{cols} grid containing {cols*rows} images, aligned in a clean and structured layout.",
        f"{cols*rows} images are placed in a grid format with {rows} horizontal rows and {cols} vertical columns.",
        f"A visual grid composed of {rows} rows and {cols} columns, showcasing {cols*rows} images in a balanced arrangement.",
        f"{cols*rows} images form a structured grid, with {rows} rows and {cols} columns, neatly aligned.",
        f"A {rows}x{cols} grid layout featuring {cols*rows} images, arranged side by side in a precise pattern.",
        f"{cols*rows} images are displayed in a grid of {rows} rows and {cols} columns, creating a uniform visual structure.",
        f"A grid with {rows} rows and {cols} columns, containing {cols*rows} images arranged in a symmetrical layout.",
        f"{cols*rows} images are organized into a {rows}x{cols} grid, forming a cohesive and orderly display.",
    ]
    return random.choice(layout_instruction)

def get_task_instruction(condition_prompt, target_prompt):
    task_instruction = [
        f"Each row outlines a logical process, starting from {condition_prompt}, to achieve {target_prompt}.",
        f"In each row, a method is described to use {condition_prompt} for generating {target_prompt}.",
        f"Each row presents a task that leverages {condition_prompt} to produce {target_prompt}.",
        f"Every row demonstrates how to transform {condition_prompt} into {target_prompt} through a logical approach.",
        f"Each row details a strategy to derive {target_prompt} based on the provided {condition_prompt}.",
        f"In each row, a technique is explained to convert {condition_prompt} into {target_prompt}.",
        f"Each row illustrates a pathway from {condition_prompt} to {target_prompt} using a clear logical task.",
        f"Every row provides a step-by-step guide to evolve {condition_prompt} into {target_prompt}.",
        f"Each row describes a process that begins with {condition_prompt} and results in {target_prompt}.",
        f"In each row, a logical task is demonstrated to achieve {target_prompt} based on {condition_prompt}.",
    ]
    return random.choice(task_instruction)

def get_content_instruction():
    content_instruction = [
        "The content of the last image in the final row is: ",
        "The last image of the last row depicts: ",
        "In the final row, the last image shows: ",
        "The last image in the bottom row illustrates: ",
        "The content of the bottom-right image is: ",
        "The final image in the last row portrays: ",
        "The last image of the final row displays: ",
        "In the last row, the final image captures: ",
        "The bottom-right corner image presents: ",
        "The content of the last image in the concluding row is: ",
    ]
    return random.choice(content_instruction)

task_dicts = [
    {
        "task_name": "conditional generation",
        "sample_weight": 1,
        "image_list": [
            ["canny", "target"],
            ["depth_anything_v2", "target"],
            ["hed", "target"],
            ["midas_depth_normal", "target"],
            ["mlsd", "target"],
            ["openpose", "target"],
            ["sam2_mask", "target"],
            ["uniformer", "target"],
        ],
    },
    {
        "task_name": "conditional generation with reference",
        "sample_weight": 1,
        "image_list": [
            # order 1
            ["reference", "canny", "target"],
            ["reference", "depth_anything_v2", "target"],
            ["reference", "hed", "target"],
            ["reference", "midas_depth_normal", "target"],
            ["reference", "mlsd", "target"],
            ["reference", "openpose", "target"],
            ["reference", "sam2_mask", "target"],
            ["reference", "uniformer", "target"],
            # order 2
            ["canny", "reference", "target"],
            ["depth_anything_v2", "reference", "target"],
            ["hed", "reference", "target"],
            ["midas_depth_normal", "reference", "target"],
            ["mlsd", "reference", "target"],
            ["openpose", "reference", "target"],
            ["sam2_mask", "reference", "target"],
            ["uniformer", "reference", "target"],
        ],
    },
    {
        "task_name": "image generation with reference",
        "sample_weight": 0.2,
        "image_list": [
            ["reference", "target"],
        ],
    },
    {
        "task_name": "subject extraction", 
        "sample_weight": 0.2,
        "image_list": [
            ["target", "reference"],
        ],
    },
    {
        "task_name": "dense prediction",
        "sample_weight": 1,
        "image_list": [
            ["target", "canny"],
            ["target", "depth_anything_v2"],
            ["target", "hed"],
            ["target", "midas_depth_normal"],
            ["target", "mlsd"],
            ["target", "openpose"],
            ["target", "sam2_mask"],
            ["target", "uniformer"],
        ],
    },
    {
        "task_name": "restoration",
        "sample_weight": 1,
        "image_list": [
            # blur related
            ["blur", "target"],
            ["compression", "target"], 
            ["SRx2", "target"],
            ["SRx4", "target"],
            ["pixelate", "target"],
            ["Defocus", "target"],
            ["GaussianBlur", "target"],
            
            # sharpen related
            ["oversharpen", "target"],
            
            # noise related
            ["GaussianNoise", "target"],
            ["PoissonNoise", "target"],
            ["SPNoise", "target"],
            
            # mosaic
            ["mosaic", "target"],
            
            # contrast related
            ["contrast_strengthen", "target"],
            ["contrast_weaken", "target"],
            
            # quantization related
            ["quantization", "target"],
            ["JPEG", "target"],
            
            # light related
            ["brighten", "target"],
            ["darken", "target"],
            ["LowLight", "target"],
            
            # color related
            ["saturate_strengthen", "target"],
            ["saturate_weaken", "target"],
            ["gray", "target"],
            ["ColorDistortion", "target"],
            
            # infilling
            ["Inpainting", "target"],
            
            # rotation related
            ["rotate90", "target"],
            ["rotate180", "target"],
            ["rotate270", "target"],
            
            # distortion related
            ["Barrel", "target"],
            ["Pincushion", "target"],
            ["Elastic", "target"],
            
            # special effects
            ["Rain", "target"],
            ["Frost", "target"]
        ],
    },
    {
        "task_name": "restoration with reference",
        "sample_weight": 1,
        "image_list": [
            # blur related
            ["reference", "blur", "target"],
            ["reference", "compression", "target"],
            ["reference", "SRx2", "target"],
            ["reference", "SRx4", "target"],
            ["reference", "pixelate", "target"],
            ["reference", "Defocus", "target"],
            ["reference", "GaussianBlur", "target"], # new
            # sharpen related
            ["reference", "oversharpen", "target"], 
            # noise related
            ["reference", "GaussianNoise", "target"],
            ["reference", "PoissonNoise", "target"],
            ["reference", "SPNoise", "target"],
            # mosaic
            ["reference", "mosaic", "target"],
            # contrast related
            ["reference", "contrast_strengthen", "target"],
            ["reference", "contrast_weaken", "target"],
            # quantization related
            ["reference", "quantization", "target"],
            ["reference", "JPEG", "target"],
            # light related
            ["reference", "brighten", "target"],
            ["reference", "darken", "target"],
            ["reference", "LowLight", "target"], # new
            # color related
            ["reference", "saturate_strengthen", "target"],
            ["reference", "saturate_weaken", "target"],
            ["reference", "gray", "target"],
            ["reference", "ColorDistortion", "target"],
            # infilling
            ["reference", "Inpainting", "target"],
            # rotation related
            ["reference", "rotate90", "target"],
            ["reference", "rotate180", "target"],
            ["reference", "rotate270", "target"],
            # distortion related
            ["reference", "Barrel", "target"],
            ["reference", "Pincushion", "target"],
            ["reference", "Elastic", "target"],
            # special effects
            ["reference", "Rain", "target"],
            ["reference", "Frost", "target"]
        ],
    }
]

def get_task_amount(task_dicts):
    total_task_amount = 0
    for task_dict in task_dicts:
        task_amount = len(task_dict["image_list"])
        total_task_amount += task_amount
        print(f"{task_dict['task_name']}: {task_amount}")
    print(f"Total task amount: {total_task_amount}")


def show_task_instruction(task_dicts):
    for task_dict in task_dicts:
        for image_list in task_dict["image_list"]:
            row_num = 1
            # layout instruction
            instruction = get_layout_instruction(row_num, len(image_list))
            # task instruction
            if len(image_list) > 2:
                condition_prompt = ", ".join([get_image_prompt(image_type) for image_type in image_list[:-2]]) + ", and " + get_image_prompt(image_list[-2])
            else:
                condition_prompt = get_image_prompt(image_list[0])
            target_prompt = get_image_prompt(image_list[-1])
            instruction = instruction + " " + get_task_instruction(condition_prompt, target_prompt)
            # content instruction
            instruction = instruction + " " + get_content_instruction() + "..."
            print(instruction)

# show_task_instruction(task_dicts)