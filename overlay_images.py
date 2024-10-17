import cv2
import numpy as np

# Load the background image (without alpha channel)
background = cv2.imread('background.png', cv2.IMREAD_COLOR)

# Load the overlay image (with alpha channel)
overlay = cv2.imread('overlay.png', cv2.IMREAD_UNCHANGED)

# Ensure the overlay has an alpha channel (4 channels: B, G, R, Alpha)
if overlay.shape[2] != 4:
    raise ValueError("Overlay image must have an alpha channel")

# Separate the overlay into BGR (color) and alpha channel
overlay_bgr = overlay[:, :, :3]  # First 3 channels: BGR
overlay_alpha = overlay[:, :, 3]  # The 4th channel: Alpha

# Define the position (x_offset, y_offset) where you want to place the overlay on the background
x_offset, y_offset = 50, 50  # Adjust as necessary

# Get the dimensions of the overlay
h_overlay, w_overlay = overlay.shape[:2]

# Ensure the overlay fits within the background dimensions
if y_offset + h_overlay > background.shape[0] or x_offset + w_overlay > background.shape[1]:
    raise ValueError("Overlay exceeds background dimensions")

# Create a region of interest (ROI) in the background where the overlay will be placed
roi_background = background[y_offset:y_offset + h_overlay, x_offset:x_offset + w_overlay]

# Normalize the alpha channel to range [0, 1]
alpha_overlay = overlay_alpha.astype(float) / 255.0

# Create an inverse alpha mask for the background
alpha_background = 1.0 - alpha_overlay

# Blend the overlay and the background using the alpha masks
for c in range(3):  # Loop over the color channels (B, G, R)
    roi_background[:, :, c] = (alpha_overlay * overlay_bgr[:, :, c] +
                                alpha_background * roi_background[:, :, c])

# Place the blended ROI back into the original background image
background[y_offset:y_offset + h_overlay, x_offset:x_offset + w_overlay] = roi_background

# Save the final output image
cv2.imwrite('output_with_alpha.png', background)

# Optional: Display the result
cv2.imshow('Overlay with Transparency', background)
cv2.waitKey(0)
cv2.destroyAllWindows()