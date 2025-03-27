def implementation_main():
    import cv2
    import numpy as np

    # Load emoji templates (each emoji type is a separate 50x50 jpg file)
    emoji_types = ["happy", "sad", "crying", "surprised", "angry"]
    emoji_templates = {emoji: cv2.imread(f"data/emojis/{emoji}.jpg", cv2.IMREAD_GRAYSCALE) for emoji in emoji_types}

    for i in range(1121):
        # Load the input image
        image_name = f"data/train/dataset/emoji_{i}.jpg"  # Change to the correct filename
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

        # Create SIFT detector
        sift = cv2.SIFT_create(nfeatures=1000)

        # Detect keypoints and descriptors for the input image
        try:
            keypoints_img, descriptors_img = sift.detectAndCompute(image, None)
        except:
            continue
        if descriptors_img is None:
            print(f"Skipping {image_name}: No descriptors found.")
            continue
        if len(keypoints_img) < 2:
            print(f"Skipping {image_name}: Not enough keypoints found.")
            continue

        # Store detected emoji results
        detected_regions = []

        # Step 1: Detect potential emoji locations first (without classification)
        for emoji_name, template in emoji_templates.items():
            if template is None:
                print(f"Error loading template: {emoji_name}.jpg")
                continue

            # Detect keypoints and descriptors for the emoji template
            keypoints_tmpl, descriptors_tmpl = sift.detectAndCompute(template, None)
            
            if descriptors_tmpl is None:
                print(f"Skipping {emoji_name}: No descriptors found.")
                continue

            # Use FLANN-based matcher
            index_params = dict(algorithm=1, trees=10)  # KDTree for SIFT
            search_params = dict(checks=100)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)

            # Find matches
            matches = matcher.knnMatch(descriptors_tmpl, descriptors_img, k=2)

            # Apply Lowe’s ratio test to filter good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:  # Threshold for good matches (adjust for more robustness)
                    good_matches.append(m)

            # print(f"Emoji: {emoji_name} Matches: {len(good_matches)}")

            # If we have enough good matches, proceed
            if len(good_matches) > 4:  # Require at least 4 good matches
                src_pts = np.float32([keypoints_tmpl[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Step 2: Use Homography (RANSAC) to find the transformation matrix
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)  # 5.0 is the threshold for outlier rejection

                if M is not None:
                    # Step 3: Apply the transformation to the template's corners to get the transformed bounding box
                    h, w = template.shape
                    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(corners, M)

                    # Step 4: Get the bounding box of the transformed corners
                    min_x = int(min(transformed_corners[:, 0, 0]))
                    max_x = int(max(transformed_corners[:, 0, 0]))
                    min_y = int(min(transformed_corners[:, 0, 1]))
                    max_y = int(max(transformed_corners[:, 0, 1]))

                    # Add the detected region to the list
                    detected_regions.append((emoji_name, min_x, min_y, max_x, max_y))

        # Step 5: Filter overlapping detections (cluster nearby detections)
        def filter_overlapping_regions(detected_regions, min_distance=30):
            filtered = []
            for emoji_name, x1, y1, x2, y2 in detected_regions:
                if not any(
                    abs(x1 - fx1) < min_distance and abs(y1 - fy1) < min_distance
                    for _, fx1, fy1, _, _ in filtered
                ):
                    filtered.append((emoji_name, x1, y1, x2, y2))
            return filtered

        filtered_emojis = filter_overlapping_regions(detected_regions)

        # Step 6: Print results
        # print(f"Detected {len(filtered_emojis)} emojis.")
        print(f"Picture: {image_name.split('/')[-1]}")
        for emoji_name, x1, y1, x2, y2 in filtered_emojis:
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            print(f"Emoji: {emoji_name} Coordinates: ({x1}, {y1})")