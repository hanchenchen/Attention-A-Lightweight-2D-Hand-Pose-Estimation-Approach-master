from PIL import Image, ImageDraw

def draw_point(points, im):
    i = 0
    draw = ImageDraw.Draw(im)

    for point in points:
        x = point[0]
        y = point[1]

        if i == 0:
            rootx = x
            rooty = y
        if i == 1 or i == 5 or i == 9 or i == 13 or i == 17:
            prex = rootx
            prey = rooty

        if i > 0 and i <= 4:
            draw.line((prex, prey, x, y), 'red')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'red', 'white')
        if i > 4 and i <= 8:
            draw.line((prex, prey, x, y), 'yellow')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'yellow', 'white')

        if i > 8 and i <= 12:
            draw.line((prex, prey, x, y), 'green')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'green', 'white')
        if i > 12 and i <= 16:
            draw.line((prex, prey, x, y), 'blue')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'blue', 'white')
        if i > 16 and i <= 20:
            draw.line((prex, prey, x, y), 'purple')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'purple', 'white')

        prex = x
        prey = y
        i = i + 1
    return im

def show_hand(im, label, save_path = None):
    if type(im) == str:
        im = Image.open(im)
    im = draw_point(label, im)
    if save_path:
        print('save output to ', save_path)
        im.save(save_path)
    im.show()
    return im
import numpy as np

def get2DKpsFromHeatmap(heatmap):
    isNoBatch = True
    if len(heatmap.shape) == 4:
        batchSize = heatmap.shape[0]
        isNoBatch = False
    elif len(heatmap.shape) == 3:
        heatmap = np.expand_dims(heatmap, 0)
        batchSize = 1
    else:
        print(heatmap.shape)
        raise Exception('Invalid shape for heatmap')

    numKps = heatmap.shape[-1]
    kps2d = np.zeros((batchSize, numKps, 2), dtype=np.uint8)
    for i in range(batchSize):
        for kp in range(numKps):
            y, x = np.unravel_index(np.argmax(heatmap[i][:,:,kp], axis=None), heatmap[i][:,:,kp].shape)
            kps2d[i, kp, 0] = x
            kps2d[i, kp, 1] = y

    if isNoBatch:
        kps2d = kps2d[0]
    return kps2d