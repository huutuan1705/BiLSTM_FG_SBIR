import scipy.ndimage
import numpy as np
from bresenham import bresenham

def draw_image_steps(vector_images, side=400, steps=20):
    for vector_image in vector_images:
        pixel_length = 0
        sample_freq = list(np.round(np.linspace(0,  len(vector_image), steps)[1:]))
        sample_len = []
        raster_images = []
        raster_image = np.zeros((int(side), int(side)), dtype=np.float32)
        initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
        
        for i in range(0, len(vector_image)):
            if i > 0: 
                if vector_image[i-1, 2] == 1:
                    initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

            cordList = list(bresenham(initX, initY, int(vector_image[i,0]), int(vector_image[i,1])))
            pixel_length += len(cordList)

            for cord in cordList:
                if 0 < cord[0] < side and 0 < cord[1] < side:
                    raster_image[cord[1], cord[0]] = 255.0
            initX , initY = int(vector_image[i,0]), int(vector_image[i,1])

            if i in sample_freq:
                raster_images.append(scipy.ndimage.binary_dilation(raster_image) * 255.0)
                sample_len.append(pixel_length)

        raster_images.append(scipy.ndimage.binary_dilation(raster_image) * 255.0)
        sample_len.append(pixel_length)

    return raster_images, sample_len

def preprocess(sketch_points, side=400):
    sketch_points = sketch_points.astype(np.float32)
    sketch_points[:, :2] = sketch_points[:, :2] / np.array([256, 256])
    sketch_points[:, :2] = sketch_points[:, :2] * side
    sketch_points = np.round(sketch_points)
    return sketch_points

def rasterize_sketch_steps(sketch_points, steps=20):
    sketch_points = preprocess(sketch_points)
    raster_images, _ = draw_image_steps([sketch_points], steps=steps+1)
    return raster_images