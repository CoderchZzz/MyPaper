QUESTIONS = {
    'fg': {
        'common': [
            'What kind of {} is in the photo?',
            'What is this {} also called?',
            'What is another word for this {}?',
            'What is another name for this {}?',
        ],
        'specific': {
            'sofa': [
                "How many cushions are on this {}?",
                "What room setting contains this {}?",
                "What specific type of seating is this {}?",
                "What shape is this {}?"
            ],
            'tv monitor': [
                "What content is displayed on the {} screen?",
                "What mounting method is used for this {}?",
                "What type of display is shown on this {}?",
                "What size is this {} screen?"
            ],
            # ---------------------- 其他类别专项模板 ----------------------
            'person': [
                "What is this {} wearing?",
                "What age group does this {} belong to?",
            ],
        }
    },
    'bg': {
        'common': [
            'what is above the {}?',
            'what is under the {}?',
            'what is behind the {}?',
            'what is around the {}?',
            'what is next to the {}?',
            'what is the left side of {}?',
            'what is the right side of {}?',
            'what scene is the {} in?',
            'what environment surrounds the {}?',
            'what place is the {} in?',
        ],
        'specific': {
            # ---------------------- 高混淆背景模板 ----------------------
            'chair': [
                "What room contains this {}?",
                "What furniture surrounds this {}?",
                "What setting contains this {}?"
            ],
            'tv monitor': [
                "What room houses this {}?",
                "What furniture supports this {}?",
                "What viewing arrangement surrounds this {}?"
            ],
        }
    }
}