kwargs_per_input = {
    "mario_small.jpg": {
        "threshold": 0.8,  # your value here
        "iou_threshold": 0.5,  # your value here
        "scale_factor": 1.0,  # for mario_small.jpg, don't worry about scale/levels
        "levels": 1,  # for mario_small.jpg, don't worry about scale/levels
    },
    "other_images": {
        "threshold": 0.92,  # your value here
        "iou_threshold": 0.2,  # your value here
        "scale_factor": 0.7,  # your value here
        "levels": 8,  # your value here
    },
}
