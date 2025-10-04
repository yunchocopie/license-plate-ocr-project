import cv2
import numpy as np

class PerspectiveCorrection:
    def __init__(self):
        pass

    def correct(self, image):
        if image is None or image.size == 0:
            return image

        if image.dtype != np.uint8:
            img_to_process = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img_to_process = image.copy()

        # 1. 이진화 (adaptive threshold로 글자 대비 확보)
        binary = cv2.adaptiveThreshold(
            img_to_process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 5
        )

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        img_area = img_to_process.shape[0] * img_to_process.shape[1]

        # 2. 유효한 contour 필터링 (너무 작거나 이상한 비율 제거)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (0.01 * img_area < area < 0.85 * img_area):
                continue
            rect = cv2.minAreaRect(cnt)
            (w, h) = rect[1]
            if w == 0 or h == 0:
                continue
            ratio = max(w, h) / min(w, h)
            if 1.2 < ratio < 5.0:
                candidates.append(cnt)

        if not candidates:
            return image

        max_contour = max(candidates, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)
        angle = rect[2]

        # 3. 회전 각도 보정
        if angle < -45.0:
            angle += 90.0
        elif angle > 45.0:
            angle -= 90.0

        center = tuple(np.array(img_to_process.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(
            img_to_process, rot_mat, img_to_process.shape[1::-1],
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255
        )

        # 4. 회전 후 윤곽선 재검출
        binary_rotated = cv2.adaptiveThreshold(
            rotated_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 5
        )
        contours_rotated, _ = cv2.findContours(binary_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_rotated:
            return rotated_image

        max_contour = max(contours_rotated, key=cv2.contourArea)
        if cv2.contourArea(max_contour) < img_area * 0.01:
            return rotated_image

        rect = cv2.minAreaRect(max_contour)
        box_points = cv2.boxPoints(rect)
        box = np.intp(box_points)

        # 5. 원근 보정 포인트 설정
        src_pts = self.order_points(box.astype("float32"))
        (tl, tr, br, bl) = src_pts

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))

        # 6. 잘못된 비율 보정 방지 조건
        if (
                max_width < 30 or max_height < 10 or
                max_width / max_height < 1.2 or max_width / max_height > 6.0
        ):
            print("[PerspectiveCorrection] Perspective condition failed. Returning fallback image.")
            return rotated_image

        dst_pts = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        try:
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(
                rotated_image, M, (max_width, max_height),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255
            )
            return warped
        except Exception as e:
            print(f"Error during perspective transform: {e}")
            return rotated_image

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        return rect
