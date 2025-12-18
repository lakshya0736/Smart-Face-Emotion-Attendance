import cv2
import os

# Videos list – use raw strings to handle spaces in paths
videos = [
    r"C:\Users\Lakshya Srivastav\Downloads\Begining\data\raw_videos\Karan.mp4",
    r"C:\Users\Lakshya Srivastav\Downloads\Begining\data\raw_videos\Lakshya.mp4"
]

# Output folder
output_folder = r"C:\Users\Lakshya Srivastav\Downloads\Begining\data\raw_images"
os.makedirs(output_folder, exist_ok=True)

def extract_frames(video_path, save_folder, interval=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_folder = os.path.join(save_folder, video_name)
    os.makedirs(video_folder, exist_ok=True)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every 'interval' frames
        if frame_count % interval == 0:
            img_name = f"{video_name}_{saved_count}.jpg"
            cv2.imwrite(os.path.join(video_folder, img_name), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames for {video_name}")

# Run extraction
for video_path in videos:
    extract_frames(video_path, output_folder, interval=30)  # ~1 frame per second for 30 fps video

print("Done extracting frames!")
