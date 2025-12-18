import os
import torch
from lxml import etree
from clip.utils import parse_xml_to_dict, scoremap2bbox
from clip.clip_text import *
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from pytorch_grad_cam.utils.image import scale_cam_image
import cv2
import numpy as np
from clip.net.vqa_bg import BG_VQA
from clip.net.vqa_fg import FG_VQA
from clip.net.visual_questions import QUESTIONS
from clip.net import clip_adapter
from clip.net.cat_names import category_dict
from clip.net.class_weights import *
class ClipOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def generate_clip_fts(image, model, require_all_fts=True):
    model = model.cuda()

    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    h, w = image.shape[-2], image.shape[-1]
    image = image.cuda()

    image_features_all, attn_weight_list = model.encode_image(image, h, w, require_all_fts=require_all_fts)

    return image_features_all, attn_weight_list


def generate_trans_mat(aff_mask, attn_weight, grayscale_cam):
    aff_mask = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])
    aff_mat = attn_weight

    trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

    for _ in range(2):
        trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
    trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

    for _ in range(1):
        trans_mat = torch.matmul(trans_mat, trans_mat)

    trans_mat = trans_mat * aff_mask

    return trans_mat


def compute_trans_mat(attn_weight):
    aff_mat = attn_weight

    trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

    for _ in range(2):
        trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
    trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

    for _ in range(1):
        trans_mat = torch.matmul(trans_mat, trans_mat)

    trans_mat = trans_mat

    return trans_mat


def generate_trans_mat_seg(aff_mask, attn_weight, grayscale_cam):
    aff_mask = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])
    aff_mat = attn_weight

    trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

    for _ in range(2):
        trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
    trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

    for _ in range(1):
        trans_mat = torch.matmul(trans_mat, trans_mat)

    trans_mat = trans_mat * aff_mask

    return trans_mat


def perform_single_voc_cam_pro(args,img_path, image, image_features, attn_weight_list, seg_attn, bg_text_features,
                           fg_text_features, cam, bg_vqa_tool=None, fg_vqa_module=None, clip_model=None,mode='train', require_seg_trans=False):
    bg_text_features = bg_text_features.cuda()
    fg_text_features = fg_text_features.cuda()
    device = "cuda" if torch.cuda.is_available() else "cpu"



    # ⭐ 如果外面已经传进来了 clip_model / bg_vqa_tool / fg_vqa_module，就直接用
    if clip_model is None or bg_vqa_tool is None or fg_vqa_module is None:
        # 兼容旧代码：按原来的方式初始化一次
        clip_model, clip_input_size = clip_adapter.load_clip_model(
            args.clip,
            device=device,
            mask_adapted=args.use_mask_clip
        )

        bg_vqa_tool = BG_VQA(
            args.vqa_bg_file,
            QUESTIONS['bg'],
            clip_model,
            args.clip,
            device=device,
            cache_path=args.vqa_bg_cache_file,
        )
        bg_vqa_tool.load_cache()

        fg_vqa_module = FG_VQA(
            args.vqa_fg_file,
            QUESTIONS['fg'],
            category_dict['voc'],
            clip_model,
            args.clip,
            cache_path=args.vqa_fg_cache_file
        )

    ori_image = Image.open(img_path)
    ori_height, ori_width = np.asarray(ori_image).shape[:2]
    label_id_list = np.unique(ori_image)
    label_id_list = (label_id_list - 1).tolist()
    if 255 in label_id_list:
        label_id_list.remove(255)
    if 254 in label_id_list:
        label_id_list.remove(254)

    # origin_label_list = []
    label_list = []
    for lid in label_id_list:
        label_list.append(new_class_names1[int(lid)])
    label_id_list = [int(lid) for lid in label_id_list]

    image = image.unsqueeze(0)
    h, w = image.shape[-2], image.shape[-1]

    highres_cam_to_save = []
    keys = []

    cam_refined_list = []

    bg_features_temp = bg_text_features.cuda()  # [bg_id_for_each_image[im_idx]].to(device_id)
    fg_features_temp = fg_text_features[label_id_list].cuda()
    # text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)
    # input_tensor = [image_features, text_features_temp.cuda(), h, w]

    # 将背景文本特征转换为设备ID和正确的数据类型
    target_dtype = image_features.dtype
    img_name = os.path.basename(img_path)
    im = os.path.splitext(img_name)[0] + '.jpg'
    # 准备VQA前景特征
    all_fg_vqa_feats = []
    for label in label_list:
        fg_vqa_feat = fg_vqa_module(label, im)
        all_fg_vqa_feats.append(fg_vqa_feat)
    all_fg_vqa_feats = torch.cat(all_fg_vqa_feats, dim=0).cuda().to(target_dtype)


    for idx, label in enumerate(label_list):
        label_index = new_class_names1.index(label)
        keys.append(label_index)
        targets = [ClipOutputTarget(label_list.index(label))]

        current_fg_features = fg_features_temp.clone()  # 克隆原始特征
        weights = class_specific_weights_best1.get(label, {
            'clip_fg': 0.5, 'vqa_fg': 0.5,
            'clip_bg': 0.5, 'vqa_bg': 0.5
        })

        # 只融合当前类别的特征，其他类别保持原始CLIP特征
        current_fg_features[idx] = (weights['clip_fg'] * fg_features_temp[idx] +
                                    weights['vqa_fg'] * all_fg_vqa_feats[idx])

        bg_vqa_feat = bg_vqa_tool.feats(label, im)
        fused_bg_features = bg_features_temp.clone()
        if bg_vqa_feat is not None:
            bg_vqa_feat = bg_vqa_feat.cuda().to(target_dtype)
            # 处理尺寸不匹配的情况
            if bg_vqa_feat.size(0) != bg_features_temp.size(0):
                bg_vqa_feat = bg_vqa_feat.repeat(bg_features_temp.size(0) // bg_vqa_feat.size(0), 1)
                if bg_features_temp.size(0) % bg_vqa_feat.size(0) != 0:
                    remaining = bg_features_temp.size(0) - bg_vqa_feat.size(0)
                    bg_vqa_feat = torch.cat([bg_vqa_feat, bg_vqa_feat[:remaining]], dim=0)
                    # 使用平均的背景VQA特征进行融合
                    fused_bg_features = (weights['clip_bg'] * bg_features_temp +
                                         weights['vqa_bg'] * bg_vqa_feat)

        text_features_temp = torch.cat([current_fg_features, fused_bg_features], dim=0)
        text_features_temp = text_features_temp.to(target_dtype)
        input_tensor = [image_features, text_features_temp.cuda(), h, w]

        grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=input_tensor,
                                                                targets=targets,
                                                                target_size=None)  # (ori_width, ori_height))

        grayscale_cam = grayscale_cam[0, :]

        grayscale_cam_highres = cv2.resize(grayscale_cam, (w, h))
        highres_cam_to_save.append(torch.tensor(grayscale_cam_highres))

        if idx == 0:
            if require_seg_trans == True:
                attn_weight = torch.cat([attn_weight_list, attn_weight_last], dim=0)
                attn_weight = attn_weight[:, 1:, 1:][-6:]  # -8

                # attn_diff = torch.abs(seg_attn - attn_weight)
                attn_diff = seg_attn - attn_weight
                attn_diff = torch.sum(attn_diff.flatten(1), dim=1)
                diff_th = torch.mean(attn_diff)

                attn_mask = torch.zeros_like(attn_diff)
                attn_mask[attn_diff <= diff_th] = 1

                attn_mask = attn_mask.reshape(-1, 1, 1)
                attn_mask = attn_mask.expand_as(attn_weight)
                attn_weight = torch.sum(attn_mask * attn_weight, dim=0) / (torch.sum(attn_mask, dim=0) + 1e-5)

                attn_weight = attn_weight.detach()
                attn_weight = attn_weight * seg_attn.squeeze(0).detach()
            else:
                attn_weight = torch.cat([attn_weight_list, attn_weight_last], dim=0)
                attn_weight = attn_weight[:, 1:, 1:][-8:]
                attn_weight = torch.mean(attn_weight, dim=0)  # (1, hw, hw)
                attn_weight = attn_weight.detach()
            _trans_mat = compute_trans_mat(attn_weight)
        _trans_mat = _trans_mat.float()

        box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
        aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1])).cuda()
        for i_ in range(cnt):
            x0_, y0_, x1_, y1_ = box[i_]
            aff_mask[y0_:y1_, x0_:x1_] = 1

        aff_mask = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])
        trans_mat = _trans_mat * aff_mask

        cam_to_refine = torch.FloatTensor(grayscale_cam).cuda()
        cam_to_refine = cam_to_refine.view(-1, 1)

        cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h // 16, w // 16)
        cam_refined_list.append(cam_refined)

    if mode == 'train':
        return cam_refined_list, keys, w, h
    else:
        return cam_refined_list, keys, ori_width, ori_height


def generate_cam_label(cam_refined_list, keys, w, h):
    refined_cam_to_save = []
    refined_cam_all_scales = []
    for cam_refined in cam_refined_list:
        cam_refined = cam_refined.cpu().numpy().astype(np.float32)
        cam_refined_highres = scale_cam_image([cam_refined], (w, h))[0]
        refined_cam_to_save.append(torch.tensor(cam_refined_highres))

    keys = torch.tensor(keys)

    refined_cam_all_scales.append(torch.stack(refined_cam_to_save, dim=0))

    refined_cam_all_scales = refined_cam_all_scales[0]

    return {'keys': keys.numpy(), 'refined_cam': refined_cam_all_scales}


def perform_single_coco_cam(img_path, image, image_features, attn_weight_list, seg_attn, bg_text_features,
                            fg_text_features, cam, mode='train', require_all_fts=True, require_seg_trans=False):
    bg_text_features = bg_text_features.cuda()
    fg_text_features = fg_text_features.cuda()

    ori_image = Image.open(img_path)
    ori_height, ori_width = np.asarray(ori_image).shape[:2]
    label_id_list = np.unique(ori_image)
    label_id_list = (label_id_list - 1).tolist()
    if 255 in label_id_list:
        label_id_list.remove(255)
    if 254 in label_id_list:
        label_id_list.remove(254)

    # print(label_id_list)
    label_list = []
    for lid in label_id_list:
        label_list.append(new_class_names_coco[int(lid)])
    label_id_list = [int(lid) for lid in label_id_list]

    image = image.unsqueeze(0)
    h, w = image.shape[-2], image.shape[-1]

    highres_cam_to_save = []
    keys = []

    cam_refined_list = []

    bg_features_temp = bg_text_features.cuda()  # [bg_id_for_each_image[im_idx]].to(device_id)
    fg_features_temp = fg_text_features[label_id_list].cuda()
    text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)
    input_tensor = [image_features, text_features_temp.cuda(), h, w]

    for idx, label in enumerate(label_list):
        label_index = new_class_names_coco.index(label)
        keys.append(label_index)
        targets = [ClipOutputTarget(label_list.index(label))]
        grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=input_tensor,
                                                                targets=targets,
                                                                target_size=None)  # (ori_width, ori_height))

        grayscale_cam = grayscale_cam[0, :]

        grayscale_cam_highres = cv2.resize(grayscale_cam, (w, h))
        highres_cam_to_save.append(torch.tensor(grayscale_cam_highres))

        # if idx == 0:
        #     attn_weight = torch.cat([attn_weight_list, attn_weight_last], dim=0)
        #     attn_weight = attn_weight[:, 1:, 1:][-8:]
        #     attn_weight = torch.mean(attn_weight, dim=0)  # (1, hw, hw)
        #     attn_weight = attn_weight.detach()
        #     if require_seg_trans == True:
        #         attn_weight = attn_weight * seg_attn.squeeze(0).detach()
        if idx == 0:
            if require_seg_trans == True:
                attn_weight = torch.cat([attn_weight_list, attn_weight_last], dim=0)
                attn_weight = attn_weight[:, 1:, 1:][-10:]  # -8

                # attn_diff = torch.abs(seg_attn - attn_weight)
                attn_diff = seg_attn - attn_weight
                attn_diff = torch.sum(attn_diff.flatten(1), dim=1)
                diff_th = torch.mean(attn_diff)

                attn_mask = torch.zeros_like(attn_diff)
                attn_mask[attn_diff <= diff_th] = 1

                attn_mask = attn_mask.reshape(-1, 1, 1)
                attn_mask = attn_mask.expand_as(attn_weight)
                attn_weight = torch.sum(attn_mask * attn_weight, dim=0) / (torch.sum(attn_mask, dim=0) + 1e-5)

                attn_weight = attn_weight.detach()
                attn_weight = attn_weight * seg_attn.squeeze(0).detach()
            else:
                attn_weight = torch.cat([attn_weight_list, attn_weight_last], dim=0)
                attn_weight = attn_weight[:, 1:, 1:][-8:]
                attn_weight = torch.mean(attn_weight, dim=0)  # (1, hw, hw)
                attn_weight = attn_weight.detach()
            _trans_mat = compute_trans_mat(attn_weight)
        _trans_mat = _trans_mat.float()

        box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.7, multi_contour_eval=True)
        aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1]))
        for i_ in range(cnt):
            x0_, y0_, x1_, y1_ = box[i_]
            aff_mask[y0_:y1_, x0_:x1_] = 1

        aff_mask = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])
        trans_mat = _trans_mat.cuda() * aff_mask.cuda()

        cam_to_refine = torch.FloatTensor(grayscale_cam).cuda()
        cam_to_refine = cam_to_refine.view(-1, 1)

        cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h // 16, w // 16)
        cam_refined_list.append(cam_refined)

    if mode == 'train':
        return cam_refined_list, keys, w, h
    else:
        return cam_refined_list, keys, ori_width, ori_height


