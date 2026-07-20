import cv2
import numpy as np
import config


class LaneDetector:
    """
    Detects left and right lane lines using Canny + Hough transforms
    and returns a dynamic danger zone trapezoid that follows the actual lane.

    Falls back to the config-based fixed trapezoid when lane lines
    can't be detected confidently (night, tunnels, faded markings).
    """

    def __init__(self):
        self._left_line  = None   # (x1, y1, x2, y2) smoothed
        self._right_line = None
        self.alpha       = config.LANE_SMOOTH_ALPHA
        self._no_detect_count = 0   # consecutive frames without detection


    def _get_roi_mask(self, shape):
        """Returns a binary mask keeping only the lower ROI of the frame."""
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        top  = int(h * config.LANE_ROI_TOP_FRAC)
        roi  = np.array([[
            (0, h),
            (0, top),
            (w, top),
            (w, h),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, roi, 255)
        return mask

    def _filter_lines(self, lines, frame_w):
        """
        Separates raw Hough lines into left/right by slope.
        Returns two lists: left_lines, right_lines.
        Each entry is (slope, intercept).
        """
        left, right = [], []
        if lines is None:
            return left, right

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Filter by slope magnitude to remove noise
            if abs(slope) < config.LANE_SLOPE_MIN:
                continue
            if abs(slope) > config.LANE_SLOPE_MAX:
                continue

            # Negative slope + left half = left lane
            # Positive slope + right half = right lane
            mid_x = (x1 + x2) / 2
            if slope < 0 and mid_x < frame_w * 0.6:
                left.append((slope, intercept))
            elif slope > 0 and mid_x > frame_w * 0.4:
                right.append((slope, intercept))

        return left, right

    def _average_line(self, lines, frame_h, roi_top):
        """
        Averages a list of (slope, intercept) into a single (x1,y1,x2,y2)
        line spanning from the bottom of the frame to the ROI top.
        Returns None if list is empty.
        """
        if not lines:
            return None
        slopes     = [s for s, _ in lines]
        intercepts = [b for _, b in lines]
        slope      = float(np.median(slopes))
        intercept  = float(np.median(intercepts))

        if abs(slope) < 1e-6:
            return None

        y_bottom = frame_h - 1
        y_top    = roi_top
        x_bottom = int((y_bottom - intercept) / slope)
        x_top    = int((y_top    - intercept) / slope)
        return (x_bottom, y_bottom, x_top, y_top)

    def _smooth(self, new_line, prev_line):
        """
        Exponential moving average between previous and new line coords.
        Keeps the trapezoid from jittering frame-to-frame.
        """
        if prev_line is None:
            return new_line
        if new_line is None:
            return prev_line
        smoothed = tuple(
            int(self.alpha * n + (1 - self.alpha) * p)
            for n, p in zip(new_line, prev_line)
        )
        return smoothed

    def _fallback_zone(self, frame_w, frame_h):
        """Returns the fixed config-based trapezoid as fallback."""
        frac = config.DANGER_ZONE_FRAC
        def s(fx, fy): return (int(fx * frame_w), int(fy * frame_h))
        return [
            s(*frac["tl"]),
            s(*frac["tr"]),
            s(*frac["br"]),
            s(*frac["bl"]),
        ]


    def detect(self, frame):
        """
        Runs lane detection on a frame.
        Returns:
            zone_pts  : list of 4 (x,y) tuples forming the danger zone
            debug_img : grayscale edge image for optional debug display
            confident : True if lane lines were found, False if fallback used
        """
        h, w = frame.shape[:2]
        roi_top = int(h * config.LANE_ROI_TOP_FRAC)

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges   = cv2.Canny(blurred,
                            config.LANE_CANNY_LOW,
                            config.LANE_CANNY_HIGH)

        mask        = self._get_roi_mask(frame.shape)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=config.LANE_HOUGH_THRESHOLD,
            minLineLength=config.LANE_HOUGH_MIN_LEN,
            maxLineGap=config.LANE_HOUGH_MAX_GAP,
        )

        left_lines, right_lines = self._filter_lines(lines, w)

        left_raw  = self._average_line(left_lines,  h, roi_top)
        right_raw = self._average_line(right_lines, h, roi_top)

        # Smooth with EMA
        self._left_line  = self._smooth(left_raw,  self._left_line)
        self._right_line = self._smooth(right_raw, self._right_line)

        confident = (self._left_line is not None
                     and self._right_line is not None)

        if confident:
            self._no_detect_count = 0
            lx_bot, ly_bot, lx_top, ly_top = self._left_line
            rx_bot, ry_bot, rx_top, ry_top = self._right_line

            zone_pts = [
                (lx_top, ly_top),   # top-left  (left lane, near horizon)
                (rx_top, ry_top),   # top-right (right lane, near horizon)
                (rx_bot, ry_bot),   # bottom-right
                (lx_bot, ly_bot),   # bottom-left
            ]
        else:
            self._no_detect_count += 1
            zone_pts = self._fallback_zone(w, h)

        return zone_pts, masked_edges, confident

    def draw_lanes(self, frame, confident: bool):
        """
        Draws the detected left/right lane lines on the frame.
        Green if confident, grey if using cached/fallback.
        """
        color = (0, 255, 0) if confident else (120, 120, 120)

        for line in [self._left_line, self._right_line]:
            if line is None:
                continue
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        return frame