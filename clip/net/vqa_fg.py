import json
import os.path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from clip.net.vqa_bg import gen_clip_features
from clip.net.visual_questions import TEXT_PROMPT
from clip.net.blip_questions import QUESTIONS

SPECIFIC_LABELS = ['boat', 'bottle', 'chair', 'sofa', 'tv monitor', 'dining table', 'potted plant','person','bicycle','car','motorbike']
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

supply_cat = {
    'dining table': ['table', 'dining table'],
    'tv monitor': ['monitor', 'tv monitor', ],
    'potted plant': ['plant', 'potted plant', ],
}

# 处理前景问答的答案文本
def fg_qa_prompt(questions: list, answers: list, label: str):
    # 获取标签的扩展标签列表
    exp_label = supply_cat.get(label, [label])
    # 初始化新的答案列表
    new_answers = []
    # 遍历问题和答案
    for q, a in zip(questions, answers):
        # 如果答案不包含扩展标签中的任何一个,则将标签添加到答案中
        if not any([e in a for e in exp_label]):
            a = f'{a} {label}'
        # if label not in a:
        #     a = f'{a} {label}'
        # 将处理后的答案添加到新的答案列表中
        new_answers.append(a)
    # 返回处理后的答案列表
    return new_answers


class FG_VQA(nn.Module):
    def __init__(self, vqa_file_path: str, questions: list, label_names: list,
                 clip_model, clip_name: str,
                 prompt: str = TEXT_PROMPT,
                 modify_cache: bool = False,
                 label_weight_init: float = 0.5,
                 dtype: torch.dtype = torch.float32,
                 cache_path: str = None
                 ):
        super().__init__()
        # 初始化VQA前景类
        self.questions = questions
        self.label_names = label_names
        self.question_num = len(self.questions)
        self.clip_name = clip_name.replace('/', '-')
        self.dtype = dtype
        self.cache_path = cache_path
        print('===')
        print(f">> FG VQA INIT:")
        print(f">> questions: {self.questions}")
        print(f">> label_names: {self.label_names}")
        print(f">> question_num: {self.question_num}")
        print(f">> clip_name: {self.clip_name}")
        print(f">> dtype: {self.dtype}")
        print(f">> cache_path: {self.cache_path}")
        # 断言vqa_file_path存在
        assert os.path.exists(vqa_file_path)
        # 加载vqa_file_path中的原始文本数据,允许pickle
        qa_file = np.load(vqa_file_path, allow_pickle=True).item()
        # 如果modify_cache为True,则生成缓存数据
        if modify_cache:
            # generate fg vqa cache, called by tools/gen_text_feats_cache.py
            assert self.cache_path is not None
            cache = self._preproccess_qa(qa_file, prompt, clip_model, cache=None)
            np.save(self.cache_path, cache)
            print(f"save fg cache to {self.cache_path}")
            return

        # 断言缓存路径存在      
        assert os.path.exists(self.cache_path)
        print(f"load fg cache from {self.cache_path}")
        # 加载缓存数据,加载的是文本编码后的张量
        cache = np.load(self.cache_path, allow_pickle=True).item()
        # 预处理问答数据
        cache = self._preproccess_qa(qa_file, prompt, clip_model, cache=cache)
        # 预处理标签
        self._preprocess_label(self.label_names, prompt, clip_model)
        # 初始化权重数量
        self.weight_num = len(self.label_names)
        qa_feat_init = 1 / self.question_num

        # label_feat_weights: [self.weight_num, 1], init to label_weight_init
        assert 0 <= label_weight_init <= 1
        # 初始化标签权重
        self.label_feat_weights = nn.Parameter(torch.tensor(
            [[label_weight_init]],
            device='cuda', dtype=self.dtype
        ))
        # 标签权重不需要梯度
        self.label_feat_weights.requires_grad = False
        # 初始化答案权重
        # qa_feat_weights: [self.weight_num, question_num], init to 1/question_num
        # 初始化答案权重
        self.qa_feat_weights = nn.Parameter(torch.tensor(
            [[qa_feat_init for _ in range(self.question_num)]],
            device='cuda', dtype=self.dtype
        ))
        # 答案权重不需要梯度
        self.qa_feat_weights.requires_grad = False

        print(f"label_feat_weights: {self.label_feat_weights.shape} all init to {label_weight_init} ")
        print(f"qa_feat_weights: {self.qa_feat_weights.shape} all init to {qa_feat_init} ")

    # 预处理问答数据
    def _preproccess_qa(self, qa_file: dict, prompt: str, clip_model, cache: dict = None):
        # 获取vqa文件中的问题列表
        file_questions = [q for (q, a) in list(list(qa_file.values())[0].values())[0]['vqa']]
        # 获取输入questions在file_questions中的索引位置
        q_ids = [file_questions.index(q) for q in self.questions]
        # 初始化vqas字典
        self.vqas = {}
        # 如果cache为None,则初始化一个空字典
        cache = cache if cache is not None else {}
        # 遍历vqa文件中的所有图片
        for img_name in tqdm(qa_file.keys(), desc='processing fg vqa', dynamic_ncols=False, ascii=True):
            # 初始化vqas字典中的img_name键对应的值为空字典
            self.vqas[img_name] = {}
            # 如果img_name不在cache中,则初始化一个空字典
            if img_name not in cache:
                cache[img_name] = {}
            # 遍历vqa文件中的所有标签
            for label in qa_file[img_name]:
                # 如果标签不在cache[img_name]中,则初始化一个空字典
                if label not in cache[img_name]:
                    cache[img_name][label] = {}
                # 获取标签的答案列表
                answers = qa_file[img_name][label]['answers']
                # questions = select_template(label, 'fg')
                # q_ids = [questions.index(q) for q in questions]
                # 获取输入questions在answers中的索引位置
                origin_answers = [answers[_i] for _i in q_ids]
                # 处理前景问答的答案文本
                answers = fg_qa_prompt(self.questions, origin_answers, label)
                # 初始化答案特征列表
                answers_feat = []
                # 遍历处理后的答案
                for a_i, ans in enumerate(answers):
                    # 如果答案在cache中,则直接获取答案特征
                    if cache.get(img_name, {}).get(label, {}).get(ans, None) is not None:
                        ans_feat = cache[img_name][label][ans]
                    # 如果答案不在cache中,则生成答案特征
                    else:
                        ans_feat = gen_clip_features([ans], prompt, clip_model)
                        cache[img_name][label][ans] = ans_feat
                    answers_feat.append(ans_feat)

                # 将答案特征列表转换为张量
                answers_feat = torch.cat(answers_feat, dim=0)
                # answers_feat = gen_clip_features(answers, prompt, clip_model, use_mean=False)
                # 将答案特征列表保存到vqas字典中
                self.vqas[img_name][label] = answers_feat

        # 返回处理后的cache
        return cache    

    # 获取每个原始标签模版如'a photo of aeroplane'使用CLIP编码后的张量存入labels中
    def _preprocess_label(self, label_names: list, prompt: str, clip_model):
        # 初始化标签字典
        self.labels = {}
        # 遍历标签列表  
        for label in label_names:
            # 生成标签特征
            label_feat = gen_clip_features([label], prompt, clip_model)
            # 将标签特征保存到标签字典中
            self.labels[label] = label_feat

    # 前向传播
    def forward(self, label: str, img_name: str):
        # 获取标签特征
        # 移除文件扩展名
        img_name = img_name.rsplit('.', 1)[0]
        label_feat = self.labels[label].to(self.dtype).cuda()
        # 获取答案特征
        qa_feat = self.vqas[img_name][label].to(self.dtype).cuda()
        # 获取标签权重
        label_w = self.label_feat_weights.t()
        qa_feat_shape0 = qa_feat.shape[0]
        qa_feat_init = 1 / qa_feat_shape0
        qa_feat_weights = nn.Parameter(torch.tensor(
            [[qa_feat_init for _ in range(qa_feat_shape0)]],
            device='cuda', dtype=self.dtype
        ))
        # 获取答案权重
        # qa_w = self.qa_feat_weights
        qa_w = qa_feat_weights
        # 计算特征
        feat = label_w * label_feat + (1.0 - label_w) * torch.sum(qa_w @ qa_feat, dim=0)
        # 返回特征
        return feat
