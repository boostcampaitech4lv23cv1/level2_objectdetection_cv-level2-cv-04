import albumentations as A

# aug1
aug1_transform = A.Compose([
    A.ShiftScaleRotate(
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=.8),

    A.RandomBrightnessContrast(
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=.8),
    
    A.OneOf([
        A.RGBShift(
            r_shift_limit=10,
            g_shift_limit=10,
            b_shift_limit=10, p=1.0),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20, p=1.0)],
        p=.8),

    A.ChannelShuffle(p=0.8),

    A.OneOf([
        A.Blur(
            blur_limit=3,
            p=1.0),
        A.MedianBlur(
            blur_limit=3,
            p=1.0)],
        p=0.8),
    ])

# aug2
aug2_transform = A.Compose([

    A.OneOf([
             A.GaussNoise(var_limit=(10.0, 500.0), mean=50.),
             A.RandomBrightnessContrast(brightness_limit=(-0.5,0.55), contrast_limit=(-0.5,0.5)),
             A.HueSaturationValue(),
             ], 
             p=0.8),
    
    A.OneOf([A.Blur(blur_limit=3),
             A.MedianBlur(blur_limit=3)], 
             p=0.8)

])

# aug3
aug3_transform = A.Compose([
             A.GridDistortion(num_steps=15, distort_limit=(-0.5,0.5), p=0.8),
             A.OpticalDistortion(distort_limit=(-2.,+2.), shift_limit=(-1.0,+1.0), p=0.8),
             A.CLAHE(clip_limit=4, p=0.8),
             A.ElasticTransform(p=0.8),
             A.RandomSnow()]
             )

# aug4
aug4_transform = A.Compose([
             A.OneOf([
                A.RandomRotate90(p=0.8),
                A.HorizontalFlip(p=0.8),
                A.ShiftScaleRotate(p=0.8)
                ],
            p=8.0),
            ])

