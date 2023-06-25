
import cv2

# Read input images
image_paths = ["IMG/5.jpg", "IMG/6.jpg", "IMG/7.jpg"]

# Load images
images = [cv2.imread(path) for path in image_paths]

# Create a Stitcher object
stitcher = cv2.Stitcher_create()

# Stitch the images
status, stitched_image = stitcher.stitch(images)

# Check if stitching was successful
if status == cv2.Stitcher_OK:
    # Display the stitched image
    cv2.imshow("Stitched Image", stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image stitching failed!")