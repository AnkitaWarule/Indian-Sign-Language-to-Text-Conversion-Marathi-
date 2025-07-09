import cv2
import os
import time

# ========== Configuration ==========
gestures = ["hello", "thank_you", "yes", "no", "have_you_eatten_something","How_are_you", "Please", "what_are_you_doing", "come_here","Wel_come","Bye","Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]
num_images = 300
img_size = (128, 128)
roi_start = (100, 100)
roi_end = (324, 324)
# ===================================

cap = cv2.VideoCapture(0)

for gesture in gestures:
    print(f"\n➡️  Now capturing for gesture: '{gesture}'")
    save_dir = f"..dataset/{gesture}"
    os.makedirs(save_dir, exist_ok=True)

    print("Press 's' to start capturing images...")
    count = 0
    start = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, roi_start, roi_end, (255, 0, 0), 2)
        roi = frame[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]]

        if start:
            resized = cv2.resize(roi, img_size)
            filename = os.path.join(save_dir, f"{count}.jpg")
            cv2.imwrite(filename, resized)
            count += 1

        cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Images: {count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Multi-Gesture Capture", frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            start = True
        elif key == 27:  # ESC to exit early
            break

        if count >= num_images:
            print(f"✅ Finished capturing '{gesture}'")
            time.sleep(1)
            break

cap.release()
cv2.destroyAllWindows()
