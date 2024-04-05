from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
    
    
def plot_bbox(test_image_path, bbox_pred, output_path="./plot.png"):
    fig, ax = plt.subplots(1)

    ax.imshow(np.asarray(Image.open(test_image_path)))

    rect = patches.Rectangle(
        (bbox_pred[0], bbox_pred[1]),  
        bbox_pred[2],
        bbox_pred[3], 
        linewidth=1,
        edgecolor='r',
        facecolor='none'
    )

    ax.add_patch(rect)

    plt.savefig(output_path)