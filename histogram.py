import cv2
import numpy as np

def calculate_histogram_intersection(hist1, hist2):
    # Calculate histogram intersection
    intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    return intersection

def extract_color_histogram(image):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate histogram in the hue channel
    hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    
    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist

def main():

    # tracklets_folder = "tracklets/image/"

    # for tracklets in os.listdir(tracklets_folder):
    #     tracklet_name = os.path.join(tracklets_folder, tracklets)

        


    # Load your video frames or images (replace 'your_frame_path' with your actual path)
    tracklet1_last_frame = cv2.imread('tracklets/images/3/1_3.png')
    
    # Extract color histogram from the last frame of the first tracklet
    hist_tracklet1 = extract_color_histogram(tracklet1_last_frame)
    
    # Number of remaining tracklets
    num_remaining_tracklets = 3  # Adjust this based on your actual scenario
    
    # Loop through the remaining tracklets
    for i in range(2, 2 + num_remaining_tracklets):  # Assuming tracklets are named as tracklet2, tracklet3, ...
        # Load the first frame of the current tracklet
        current_tracklet_first_frame = cv2.imread('tracklets/images/6/53_6.png')
        
        # Extract color histogram from the first frame of the current tracklet
        hist_current_tracklet = extract_color_histogram(current_tracklet_first_frame)
        
        # Calculate histogram intersection similarity score
        similarity_score = calculate_histogram_intersection(hist_tracklet1, hist_current_tracklet)
        
        print(f'Similarity score between tracklet 1 and tracklet {i}: {similarity_score}')

if __name__ == "__main__":
    main()
