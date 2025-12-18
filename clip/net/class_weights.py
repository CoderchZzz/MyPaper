class_specific_weights3 = {
    'tv monitor': {
        'clip_fg': 0.9, 'vqa_fg': 0.1,
        'clip_bg': 0.5, 'vqa_bg': 0.5
    },
    'person': {
        'clip_fg': 0.9, 'vqa_fg': 0.1,
        'clip_bg': 0.3, 'vqa_bg': 0.7
    }
}

class_specific_weights_best1 = {
    # 显著下降的类别
    'person': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 1, 'vqa_bg': 0
    },
    'tv monitor': {
        'clip_fg': 0.5, 'vqa_fg': 0.5,
        'clip_bg': 0.5, 'vqa_bg': 0.5
    },

    # 有所下降的类别
    'cat': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 0.5, 'vqa_bg': 0.5
    },
    'cow': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 0.5, 'vqa_bg': 0.5
    },
    'train': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 1, 'vqa_bg': 0
    },
    'bicycle': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 1, 'vqa_bg': 0
    },
    'horse': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 1, 'vqa_bg': 0
    },
    'motorbike': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 1, 'vqa_bg': 0
    },

    # 显著提升的类别
    'bird': {
        'clip_fg': 0.5, 'vqa_fg': 0.5,
        'clip_bg': 0.5, 'vqa_bg': 0.5
    },
    'boat': {
        'clip_fg': 0.3, 'vqa_fg': 0.7,
        'clip_bg': 0.3, 'vqa_bg': 0.7
    },
    'dining table': {
        'clip_fg': 0.4, 'vqa_fg': 0.6,
        'clip_bg': 0.4, 'vqa_bg': 0.6
    },
    'sofa': {
        'clip_fg': 0.1, 'vqa_fg': 0.9,
        'clip_bg': 0.1, 'vqa_bg': 0.9
    },

    # 性能较差但有提升的类别
    'bottle': {
        'clip_fg': 0.5, 'vqa_fg': 0.5,
        'clip_bg': 0.5, 'vqa_bg': 0.5
    },
    'chair': {
        'clip_fg': 0.5, 'vqa_fg': 0.5,
        'clip_bg': 0.5, 'vqa_bg': 0.5
    },
    'potted plant': {
        'clip_fg': 0.5, 'vqa_fg': 0.5,
        'clip_bg': 0.5, 'vqa_bg': 0.5
    },

    # 基本持平的类别
    'dog': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 0.5, 'vqa_bg': 0.5
    },
    'sheep': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 1, 'vqa_bg': 0
    },

    # 轻微提升的类别
    'aeroplane': {
        'clip_fg': 0.3, 'vqa_fg': 0.7,
        'clip_bg': 0.3, 'vqa_bg': 0.7
    },
    'bus': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 1, 'vqa_bg': 0
    },
    'car': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 1, 'vqa_bg': 0
    }
}

# 背景特征去重后使用的权重
class_specific_weights_best2 = {
    # 显著下降的类别
    'person': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 1, 'vqa_bg': 0
    },
    'tv monitor': {
        'clip_fg': 0.5, 'vqa_fg': 0.5,
        'clip_bg': 0.5, 'vqa_bg': 0.5
    },

    # 有所下降的类别
    'cat': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 0.5, 'vqa_bg': 0.5
    },
    'cow': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 0.5, 'vqa_bg': 0.5
    },
    'train': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 1, 'vqa_bg': 0
    },
    'bicycle': {
        'clip_fg': 0.5, 'vqa_fg': 0.5,
        'clip_bg': 0.5, 'vqa_bg': 0.5
    },
    'horse': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 1, 'vqa_bg': 0
    },
    'motorbike': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 1, 'vqa_bg': 0
    },

    # 显著提升的类别
    'bird': {
        'clip_fg': 0.5, 'vqa_fg': 0.5,
        'clip_bg': 0.5, 'vqa_bg': 0.5
    },
    'boat': {
        'clip_fg': 0.3, 'vqa_fg': 0.7,
        'clip_bg': 0.3, 'vqa_bg': 0.7
    },
    'dining table': {
        'clip_fg': 0.4, 'vqa_fg': 0.6,
        'clip_bg': 0.4, 'vqa_bg': 0.6
    },
    'sofa': {
        'clip_fg': 0.1, 'vqa_fg': 0.9,
        'clip_bg': 0.1, 'vqa_bg': 0.9
    },

    # 性能较差但有提升的类别
    'bottle': {
        'clip_fg': 0.5, 'vqa_fg': 0.5,
        'clip_bg': 0.5, 'vqa_bg': 0.5
    },
    'chair': {
        'clip_fg': 0.5, 'vqa_fg': 0.5,
        'clip_bg': 0.5, 'vqa_bg': 0.5
    },
    'potted plant': {
        'clip_fg': 0.5, 'vqa_fg': 0.5,
        'clip_bg': 0.5, 'vqa_bg': 0.5
    },

    # 基本持平的类别
    'dog': {
        'clip_fg': 0.5, 'vqa_fg': 0.5,
        'clip_bg': 0.5, 'vqa_bg': 0.5
    },
    'sheep': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 1, 'vqa_bg': 0
    },

    # 轻微提升的类别
    'aeroplane': {
        'clip_fg': 0.3, 'vqa_fg': 0.7,
        'clip_bg': 0.3, 'vqa_bg': 0.7
    },
    'bus': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 1, 'vqa_bg': 0
    },
    'car': {
        'clip_fg': 1, 'vqa_fg': 0,
        'clip_bg': 1, 'vqa_bg': 0
    }
}


