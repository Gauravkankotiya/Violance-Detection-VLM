import cv2
from PIL import Image
from aggressive_vlm import query_video

cap = cv2.VideoCapture("test3.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height

# Define the codec and create a VideoWriter object
output_video_path = "result-2.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print(f"Input FPS: {fps}")

if fps < 30 :
    fps = 30
frames = []
i = 0
k = 0
keeping = ""
while True:
    ret,frame = cap.read()
    if not ret:
        break
    i+=1
    if i%3 ==0:
        frames.append(Image.fromarray(frame))
    if len(frames)==fps:
        res = query_video("Is there any aggressive scene in this video like fighting or any demonstration of aggresive behaviour?",frames=frames,
            video_path="")
        
        res = res[0]
        aggression = res[:3]
        content = res[4:]
        if aggression.lower() == "yes":
            print(content)
            keeping = f"AGGRESSION : There is an aggressive scene in the video"
            cv2.putText(frame, keeping, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
        else:
            keeping = ""
        frames = frames[int(fps//2):]
    else:
        if keeping:
            cv2.putText(frame, keeping, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    # cv2.imwrite(f"./results/{k}.jpg",frame)
    # k+=1
    out.write(frame)

# Release resources
cap.release()
out.release()
print(f"Processed video saved as {output_video_path}")
    