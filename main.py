import cv2
import os

image_directory = "C:/Users/harsh/Downloads/Facial_Detector_Agent"
def list_images(directory):
    return [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# List available images
images = list_images(image_directory)
if not images:
    print("No images found in the directory.")
    exit()
# Display available images and let the user select one
print("Available images:")
for idx, image_name in enumerate(images):
    print(f"{idx + 1}: {image_name}")

choice = int(input("Enter the number of the image you want to process: ")) - 1
if choice < 0 or choice >= len(images):
    print("Invalid choice.")
    exit()
selected_image = os.path.join(image_directory, images[choice])

inputFaces = cv2.CascadeClassifier("C:/Users/harsh/Downloads/Facial_Detector_Agent/haarcascade_frontalface_alt.xml")
img = cv2.imread(selected_image)
                 
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

recognitions = inputFaces.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=2)

for (x,y,w,h) in recognitions :
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) 

cv2.imshow("output",img)
cv2.waitKey()
