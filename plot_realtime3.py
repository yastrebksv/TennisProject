import pandas as pd
import cv2
import numpy as np


def visualize_dual_view(csv_path):
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # --- SETUP VIEW 1: SCALED CAMERA (Vertical Fit) ---
    # Find bounds to fit the crazy vertical range (-988 to 3500)
    y_cols = [c for c in df.columns if c.endswith('_y')]
    all_y = df[y_cols].fillna(0).values.flatten()
    all_y = all_y[all_y != 0]  # Filter zeros

    min_y_cam, max_y_cam = np.min(all_y), np.max(all_y)
    data_height = max_y_cam - min_y_cam

    # Target height for the window
    view_h = 600
    scale_cam = (view_h - 100) / data_height

    # Calculate offset to center Y
    offset_y_cam = -min_y_cam + 50 / scale_cam

    # Width is arbitrary, just wide enough to see players
    view_w_cam = 500
    offset_x_cam = 250  # Center roughly around 0-ish if centered, or just shift positive

    # Adjust X offset based on data average to center players
    x_cols = [c for c in df.columns if c.endswith('_x')]
    avg_x = df[x_cols].mean().mean()
    offset_x_cam = -avg_x + (view_w_cam / 2 / scale_cam)

    # --- SETUP VIEW 2: MINIMAP (Standard Court) ---
    map_h, map_w = 600, 400
    # Tennis Court Dimensions (Meters)
    cw_m, cl_m = 10.97, 23.77
    map_margin = 60
    map_scale = (map_h - 2 * map_margin) / cl_m

    # Reference Court for Homography (Meters)
    # 0:TL, 1:TR, 2:BL, 3:BR (Doubles Corners)
    ref_court = np.array([
        [0, 0], [10.97, 0], [0, 23.77], [10.97, 23.77]
    ], dtype=np.float32)

    # Helper: Transform World (Meters) -> Minimap (Pixels)
    def world_to_map(wx, wy):
        mx = int(wx * map_scale + map_margin)
        my = int(wy * map_scale + map_margin)
        return mx, my

    # Window
    cv2.namedWindow("Tennis Analytics Dashboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tennis Analytics Dashboard", view_w_cam + map_w, view_h)

    # --- PLAYBACK LOOP ---
    for index, row in df.iterrows():
        # 1. Prepare Canvases
        # Left: Dark Gray
        img_cam = np.full((view_h, view_w_cam, 3), (30, 30, 30), dtype=np.uint8)
        # Right: Court Green
        img_map = np.full((map_h, map_w, 3), (50, 150, 50), dtype=np.uint8)

        # -----------------------------
        # DRAW RIGHT: MINIMAP STATIC LINES
        # -----------------------------
        # Outer Box
        cv2.rectangle(img_map, world_to_map(0, 0), world_to_map(cw_m, cl_m), (255, 255, 255), 2)
        # Net
        cv2.line(img_map, world_to_map(0, 11.89), world_to_map(cw_m, 11.89), (200, 200, 200), 2)
        # Singles Lines
        cv2.line(img_map, world_to_map(1.37, 0), world_to_map(1.37, 23.77), (255, 255, 255), 1)
        cv2.line(img_map, world_to_map(9.60, 0), world_to_map(9.60, 23.77), (255, 255, 255), 1)
        # Service Lines
        cv2.line(img_map, world_to_map(1.37, 5.49), world_to_map(9.60, 5.49), (255, 255, 255), 1)
        cv2.line(img_map, world_to_map(1.37, 18.28), world_to_map(9.60, 18.28), (255, 255, 255), 1)

        # -----------------------------
        # CALC HOMOGRAPHY
        # -----------------------------
        src_pts = []
        dst_pts = []
        for i in [0, 1, 2, 3]:  # Use corners for robustness
            x, y = row[f'court_kp_{i}_x'], row[f'court_kp_{i}_y']
            if pd.notnull(x) and pd.notnull(y):
                src_pts.append([x, y])
                dst_pts.append(ref_court[i])

        H = None
        if len(src_pts) == 4:
            H, _ = cv2.findHomography(np.array(src_pts).reshape(-1, 1, 2),
                                      np.array(dst_pts).reshape(-1, 1, 2))

        # -----------------------------
        # DRAW ENTITIES (On Both Views)
        # -----------------------------

        # Helper: Project Point to Map
        def draw_on_map(x_pix, y_pix, color, label):
            if pd.isna(x_pix) or pd.isna(y_pix) or H is None:
                return

            # Project
            pt_vec = np.array([[[x_pix, y_pix]]], dtype=np.float32)
            pt_proj = cv2.perspectiveTransform(pt_vec, H)[0][0]  # (x_meter, y_meter)

            xm, ym = pt_proj

            # CLAMPING LOGIC (Prevent disappearing)
            is_clamped = False

            # If coordinates are wild (e.g. > 5 meters outside court), clamp them
            clamp_margin_m = 5.0
            if xm < -clamp_margin_m or xm > cw_m + clamp_margin_m or \
                    ym < -clamp_margin_m or ym > cl_m + clamp_margin_m:
                is_clamped = True
                # Clamp for display
                xm = np.clip(xm, -clamp_margin_m, cw_m + clamp_margin_m)
                ym = np.clip(ym, -clamp_margin_m, cl_m + clamp_margin_m)

            px, py = world_to_map(xm, ym)

            final_color = color
            if is_clamped:
                final_color = (0, 0, 255)  # Turn Red if clamped (Out of bounds/High in air)

            cv2.circle(img_map, (px, py), 6, final_color, -1)

        # Helper: Draw on Camera View
        def draw_on_cam(x_pix, y_pix, color, label):
            if pd.isna(x_pix) or pd.isna(y_pix):
                return

            cx = int((x_pix + offset_x_cam) * scale_cam)
            cy = int((y_pix + offset_y_cam) * scale_cam)

            cv2.circle(img_cam, (cx, cy), 6, color, -1)
            cv2.putText(img_cam, label, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 1. Ball (Yellow)
        draw_on_cam(row['ball_x'], row['ball_y'], (0, 255, 255), "Ball")
        draw_on_map(row['ball_x'], row['ball_y'], (0, 255, 255), "Ball")

        # 2. Top Player (Red)
        draw_on_cam(row['top_player_x'], row['top_player_y'], (0, 0, 255), "P1")
        draw_on_map(row['top_player_x'], row['top_player_y'], (0, 0, 255), "P1")

        # 3. Bot Player (Blue)
        draw_on_cam(row['bottom_player_x'], row['bottom_player_y'], (255, 0, 0), "P2")
        draw_on_map(row['bottom_player_x'], row['bottom_player_y'], (255, 0, 0), "P2")

        # 4. Court Keypoints (Green - Only on Cam view)
        for k in range(14):
            kx, ky = row[f'court_kp_{k}_x'], row[f'court_kp_{k}_y']
            if pd.notnull(kx):
                cx = int((kx + offset_x_cam) * scale_cam)
                cy = int((ky + offset_y_cam) * scale_cam)
                cv2.circle(img_cam, (cx, cy), 2, (0, 255, 0), -1)

        # --- COMBINE & SHOW ---
        # Add labels
        cv2.putText(img_cam, "Camera View (Scaled)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_map, "Minimap (Clamped)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Stack Horizontally
        combined = np.hstack((img_cam, img_map))

        cv2.imshow("Tennis Analytics Dashboard", combined)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_dual_view(r'H:\TemprorayCodes\Dr. Klaus\Ball Tracking\TennisProject\output\test3_2_28nov.csv')