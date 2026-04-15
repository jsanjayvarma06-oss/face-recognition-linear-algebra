import cv2
import numpy as np
import sys
from preprocessing import _align_face, IMG_SIZE, preprocess

img = np.zeros((200, 200, 3), dtype=np.uint8)
# Let's see how an image behaves
cv2.imwrite("test.jpg", img)
