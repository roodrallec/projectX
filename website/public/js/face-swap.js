const FEATHER_AMOUNT = 11
const COLOUR_CORRECT_BLUR_FRAC = 0.5
const NOSE_BRIDGE_POINTS = list(range(27, 31))
const LOWER_NOSE_POINTS = list(range(30, 35))
const OUTER_LIP_POINTS = list(range(48, 60))
const INNER_LIP_POINTS = list(range(60, 68))
const FACE_POINTS = list(range(17, 68))
const MOUTH_POINTS = list(range(48, 61))
const RIGHT_BROW_POINTS = list(range(17, 22))
const LEFT_BROW_POINTS = list(range(22, 27))
const RIGHT_EYE_POINTS = list(range(36, 42))
const LEFT_EYE_POINTS = list(range(42, 48))
const NOSE_POINTS = list(range(27, 35))
const JAW_POINTS = list(range(0, 17))
const OPEN_POINTS = [JAW_POINTS, LEFT_BROW_POINTS, RIGHT_BROW_POINTS, NOSE_BRIDGE_POINTS]
const CLOSED_POINTS = [LOWER_NOSE_POINTS, LEFT_EYE_POINTS, RIGHT_EYE_POINTS, OUTER_LIP_POINTS, INNER_LIP_POINTS]
const ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
const OVERLAY_POINTS = [NOSE_POINTS, MOUTH_POINTS, OUTER_LIP_POINTS, INNER_LIP_POINTS]

function faceSwap(frame1_img, frame1_landmarks, frame1_mask, frame2_img, frame2_landmarks, frame2_mask) {
	inverse_affine_transform = procrustes(
		frame1_landmarks[ALIGN_POINTS],
		frame2_landmarks[ALIGN_POINTS]
	);

	warped_mask = warpIm(
		frame2_mask,
		inverse_affine_transform,
		frame1_img.shape
	);
	combined_mask = np.max([ frame1_mask, warped_mask ], (axis = 0));
	warped_im2 = warpIm(
		frame2_img,
		inverse_affine_transform,
		frame1_img.shape
	);
	warped_corrected_im2 = correctColors(
		frame1_img,
		warped_im2,
		frame1_landmarks
	);
	output_im = frame1_img * (1.0 - combined_mask) + warped_im2 * combined_mask;

	normalised =
		cv2.normalize(
			output_im.astype(np.float32),
			None,
			(alpha = 0),
			(beta = 1),
			(norm_type = cv2.NORM_MINMAX),
			(dtype = cv2.CV_32F)
		) * 255;

	new_img = cv2.cvtColor(normalised, cv2.COLOR_BGR2RGB);
	new_landmarks = landmark_transform(
		frame1_landmarks,
		frame2_landmarks,
		inverse_affine_transform
	);
	return normalised, new_landmarks;
}

function correctColors(im1, im2, landmarks1) {
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0)
    )
    blur_amount = int(blur_amount)

    if (blur_amount % 2 == 0) {
        blur_amount += 1
    }

    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
    return im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)
}

function warpIm(im, M, dshape) {
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im, M[:2], (dshape[1], dshape[0]),
                dst=output_im, borderMode=cv2.BORDER_TRANSPARENT,
                flags=cv2.WARP_INVERSE_MAP)
    return output_im
}

function procrustes(points1, points2) {
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    mat_mul = np.matmul(points1.T, points2)
    U, S, Vt = np.linalg.svd(mat_mul)
    R = (U * Vt).T

    transform = np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T))

    inverse_affine_transform = np.vstack([transform, np.matrix([0., 0., 1.])])

    return inverse_affine_transform
}

function landmark_transform(current_landmarks, new_landmarks, inverse_affine_transform) {
    affine_transform = np.linalg.inv(inverse_affine_transform)
    transformed_lm = np.append(new_landmarks,
                np.ones((len(new_landmarks), 1)), axis=1)
    transformed_lm = affine_transform*transformed_lm.T
    transformed_lm = transformed_lm.T[:, :2]

    for (points in OVERLAY_POINTS) {
        current_landmarks[points] = transformed_lm[points]
    }

    return current_landmarks
}

function buildMask() {
    im = np.zeros(im.shape[:2], dtype=np.float64)
    for (let group of OVERLAY_POINTS) {
        points = landmarks[group]
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=1)
    }
    im = np.array([im, im, im]).transpose((1, 2, 0))
    im = (
        cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0
        ) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    return im.astype(np.uint8)

}
