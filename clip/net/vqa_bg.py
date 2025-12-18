import numpy as np
import torch
from tqdm import tqdm
import clip
import os

from clip.net.visual_questions import TEXT_PROMPT
from clip.net.blip_questions import QUESTIONS
from clip.net.clip_text import *
#  将文本通过提示模板格式化，然后用CLIP模型编码，输出：归一化后的特征向量
SPECIFIC_LABELS = ['boat', 'bottle', 'chair', 'sofa', 'tv monitor', 'dining table', 'potted plant','person','bicycle','car','motorbike']

FOREGROUND_SYNONYMS = {
    'aeroplane': ['airplane', 'plane', 'aircraft', 'jet'],
    'bicycle': ['bike', 'cycle', 'pushbike'],
    'bird': ['avian', 'fowl'],
    'boat': ['ship', 'vessel', 'watercraft'],
    'bottle': ['container', 'flask'],
    'bus': ['coach', 'omnibus'],
    'car': ['automobile', 'vehicle', 'auto'],
    'cat': ['feline', 'kitty'],
    'chair': ['seat', 'stool', 'bench'],
    'cow': ['bovine', 'cattle'],
    'dining table': ['dinner table', 'table'],
    'dog': ['canine', 'puppy'],
    'horse': ['equine', 'steed'],
    'motorbike': ['motorcycle', 'motorcycling', 'bike'],
    'person': ['human', 'individual', 'man', 'woman'],
    'potted plant': ['plant', 'flower', 'houseplant'],
    'sheep': ['lamb', 'ewe', 'ram'],
    'sofa': ['couch', 'settee', 'divan'],
    'train': ['locomotive', 'railway'],
    'tv monitor': ['monitor', 'television', 'tv', 'screen', 'computer monitor']
}
def select_template(class_name, question_type):
    """
    动态选择模板（保留原有通用模板 + 按需添加特定模板）
    Args:
        class_name: 当前类别名称（如'boat'）
        iou_dict: 各类别IoU字典
    Returns:
        list: 合并后的问题模板列表
    """
    # 获取通用模板
    common_templates = QUESTIONS['fg']['common'] if question_type != 'bg' else QUESTIONS['bg']['common']

    # 获取特定模板（如果存在）
    specific_templates = QUESTIONS['fg']['specific'].get(class_name, []) if question_type != 'bg' \
        else QUESTIONS['bg']['specific'].get(class_name, [])

    # 对低IoU类别优先使用特定模板（保留30%通用模板）
    if class_name in SPECIFIC_LABELS:
        return specific_templates + common_templates
    else:
        return common_templates
def gen_clip_features(texts: list, prompt: str, clip_model):
    with torch.no_grad():
        prompt_texts = [prompt.format(t) for t in texts]
        feats = None
        if len(prompt_texts) != 0:
            text_feature = clip_model.encode_text(clip.tokenize(prompt_texts).cuda())  # [L, C]
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            feats = text_feature.cpu()
        return feats


class BG_VQA:
    def __init__(self,
                 vqa_file_path: str,
                 questions: list,
                 clip_model, clip_name: str,
                 prompt: str = TEXT_PROMPT,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cuda',
                 cache_path: str = None
                 ):
        # 初始化VQA背景类
        # 设置必要的参数如文件路径、问题列表、CLIP模型等  
        self.vqa_file_path = vqa_file_path
        self.questions = questions
        self.clip_name = clip_name.replace('/', '-')
        self.dtype = dtype
        self.device = device
        self.clip_model = clip_model
        self.prompt = prompt
        self.cache_file_path = cache_path
        self._update_profile()

        self.vqa_feats = None

        print('===')
        print(f">> BG VQA INIT:")
        print(f">> questions: {self.questions}")
        print(f">> clip_name: {self.clip_name}")
        print(f">> dtype: {self.dtype}")
        print(f">> cache_path: {self.cache_file_path}")
    
    # 更新配置信息
    def _update_profile(self):
        prop_list = ['clip_name', 'dtype', 'device', 'prompt',
                     'vqa_file_path', 'cache_file_path', 'questions']
        self.profile = {}
        for prop in prop_list:
            self.profile[prop] = getattr(self, prop)

    # 保存缓存数据
    def _save_cache(self, cache_path):
        total_cache = dict(profile=self.profile, feats=self.vqa_feats)
        np.save(cache_path, total_cache)

    # 加载缓存数据
    def _load_cache(self, cache_path):
        total_cache = np.load(cache_path, allow_pickle=True).item()
        self.profile = total_cache['profile']
        self.vqa_feats = total_cache['feats']
        self.profile['device'] = self.device
        self.profile['dtype'] = self.dtype

    # 生成缓存数据
    def gen_cache(self):
        # generate bg vqa cache, called by tools/gen_text_feats_cache.py
        assert os.path.exists(self.vqa_file_path)
        assert self.cache_file_path is not None
        assert isinstance(self.prompt, str)
        assert self.clip_name is not None
        assert self.clip_model is not None

        vqas: dict = np.load(self.vqa_file_path, allow_pickle=True).item()

        self.vqa_feats = self._process_vqa(vqas, self.questions)

        self._update_profile()
        self._save_cache(self.cache_file_path)
        print(f"save bg cache to {self.cache_file_path}")

    # 加载缓存数据
    def load_cache(self):
        assert os.path.exists(self.cache_file_path)
        print(f"load bg cache from {self.cache_file_path}")
        self._load_cache(self.cache_file_path)

    # 处理VQA数据
    def _process_vqa(self,
                     vqas: dict,
                     questions: list,
                     ):
        # 从vqas字典中获取问题列表,取第一个图片的第一个标签下的vqa问题列表
        vqa_file_questions = list(list(vqas.values())[0].values())[0]['vqa']
        # 提取每个问题的第一个元素(问题文本)
        vqa_file_questions = [q[0] for q in vqa_file_questions]
        # 获取输入questions在vqa_file_questions中的索引位置
        q_ids = [vqa_file_questions.index(q) for q in questions]

        # 初始化结果字典
        result = {}
        # 遍历所有图片
        for img_name in tqdm(vqas.keys(), desc='processing bg vqa', ascii=True):
            result[img_name] = {}
            # 遍历图片的所有标签
            for label in vqas[img_name].keys():
                # 获取当前标签的答案列表
                answers = vqas[img_name][label]['answers']
                # questions = select_template(label, 'bg')
                # q_ids = [questions.index(q) for q in questions]
                # 只保留q_ids对应位置的答案
                answers = [answers[_i] for _i in range(len(answers)) if _i in q_ids]
                # 去重
                answers = list(set(answers))
                # # 如果标签本身在答案中,则移除
                # # 过滤前景类别及其同义词
                # filtered_answers = []
                # for answer in answers:
                #     # 将答案转换为小写以进行不区分大小写的匹配
                #     answer_lower = answer.lower()
                #
                #     # 检查答案中是否包含任何前景类别或其同义词
                #     should_remove = False
                #
                #     # 检查是否包含前景类别
                #     for foreground in FOREGROUND_SYNONYMS.keys():
                #         if foreground.lower() in answer_lower:
                #             should_remove = True
                #             break
                #
                #     # 检查是否包含前景类别的同义词
                #     if not should_remove:
                #         for synonyms in FOREGROUND_SYNONYMS.values():
                #             for synonym in synonyms:
                #                 if synonym.lower() in answer_lower:
                #                     should_remove = True
                #                     break
                #             if should_remove:
                #                 break
                #
                #     if not should_remove:
                #         filtered_answers.append(answer)
                #
                # answers = filtered_answers
                if label in answers:
                    answers.remove(label)
                # 保存处理后的答案和对应的CLIP特征
                result[img_name][label] = {
                    'answers': answers,
                    'answer_feats': gen_clip_features(answers, self.prompt, self.clip_model),
                }

        return result

    # 获取VQA特征
    def feats(self, label_name: str, image_name: str):
        # 从vqa_feats字典中获取指定图片和标签的answer_feats特征
        # 移除文件扩展名
        image_name = image_name.rsplit('.', 1)[0]
        feat = self.vqa_feats[image_name][label_name]['answer_feats']
        # 如果特征为空,打印提示信息并返回None
        if feat is None:
            print(f"BG no feat for {label_name} in {image_name}")
            return None
        # 如果特征的数据类型与期望的dtype不一致,则转换数据类型
        if feat.dtype != self.dtype:
            feat = feat.to(self.dtype)
        
        # 将特征转移到指定设备(CPU/GPU)并返回
        return feat.to(self.device)
