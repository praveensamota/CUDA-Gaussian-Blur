import cv2

# Read the PGM file
image = cv2.imread('Lena_gaussianBlur.pgm', cv2.IMREAD_UNCHANGED)

# Save as PNG file
cv2.imwrite('output.png', image)
