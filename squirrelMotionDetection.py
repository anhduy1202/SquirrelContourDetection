import cv2

video = cv2.VideoCapture('squirrel.mov')
frameW = int(video.get(3))
frameH = int(video.get(4))
size = (frameW, frameH)

result = cv2.VideoWriter('result_squirrel.mov', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

if not video.isOpened():
    print("Error opening video file")

if video.isOpened():
    # Background frame ( first frame )
    for _ in range(10):
        _, frame = video.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)
    background = gray
    last_frame = gray

while video.isOpened():
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)
    foreground = gray - background
    _, mask = cv2.threshold(foreground, 127, 255, cv2.THRESH_BINARY)

    # Motion differences F[i] = abs(Frame[i-1] - Frame[i])
    abs_diff = cv2.absdiff(last_frame, gray)
    last_frame = gray
    _, ad_mask = cv2.threshold(abs_diff, 8, 255, cv2.THRESH_BINARY)

    # Contour detection
    contours, _ = cv2.findContours(ad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:

        # Avoid small movement
        if cv2.contourArea(contour) < 2000:
            continue

        # Movement detected
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if ret == True:
        result.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    cv2.imshow("squirrel video", frame)
    if cv2.waitKey(1) == ord("q"):
        break
