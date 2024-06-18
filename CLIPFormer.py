from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import os

from mmseg.models.utils import resize
from mmseg.models import builder
from mmcv.cnn import ConvModule
from mmseg.models.utils.se_layer import SELayer_v2 as SELayer
from mmseg.models.utils.clip_func import clip_infer, init_clip

from ..utils.untils import tokenize

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv2d(x)
        out = torch.add(out, residual)
        return out
    

def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

@MODELS.register_module()
class CLIPFormer(BaseSegmentor):
    def __init__(self,
                 backbone: ConfigType,
                 text_encoder: ConfigType,
                 context_decoder: ConfigType,
                 decode_head: ConfigType,
                 class_names=['remote sensing images', 'remote sensing images change area'],
                 context_length=5,
                 context_feature='attention',
                 score_concat_index=3,
                 text_head=False,
                 tau=0.07,
                 identity_head=None,
                 token_embed_dim=512, text_dim=1024,
                 minus_channel = [256, 512, 1024, 2050],
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained

            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            
            if 'RN50' not in pretrained and 'RN101' not in pretrained and 'ViT-B' not in pretrained:
                print('not CLIP pre-trained weight, using CLIP ViT-B-16')
                text_encoder.pretrained = 'pretrained/ViT-B-16.pt'
            else:
                text_encoder.pretrained = pretrained

        self.backbone = MODELS.build(backbone)
        self.text_encoder = MODELS.build(text_encoder)
        self.context_decoder = MODELS.build(context_decoder)
        self.context_length = context_length
        self.score_concat_index = score_concat_index
        self.minus_channel = minus_channel

        assert context_feature in ['attention', 'backbone']
        self.context_feature = context_feature

        self.text_head = text_head
        self.tau = tau

        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.with_identity_head = False
        self.identity_head = None
        self._init_identity_head(identity_head)

        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.num_classes = len(class_names)        

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        context_length = self.text_encoder.context_length - self.context_length
        self.contexts2 = nn.Parameter(torch.randn(1, 1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts2)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        self.minus_conv = nn.Sequential(ConvModule(
                    in_channels=self.minus_channel[0],
                    out_channels=256,
                    kernel_size=1),
                    ConvModule(
                    in_channels=self.minus_channel[1],
                    out_channels=256,
                    kernel_size=1),
                    ConvModule(
                    in_channels=self.minus_channel[2],
                    out_channels=256,
                    kernel_size=1),
                    ConvModule(
                    in_channels=self.minus_channel[3],
                    out_channels=256,
                    kernel_size=1)
                    )
        self.channel_att = nn.Sequential(SELayer(768, 256), SELayer(768, 256), SELayer(768, 256), SELayer(768, 256))
        # self.channel_att = nn.Sequential(SELayer(256, 256), SELayer(256, 256), SELayer(256, 256), SELayer(256, 256))
        
        self.embedding_dim = text_dim // 2
        self._init_optimal_dist()

        assert self.with_decode_head
    
    def _init_optimal_dist(self) -> None:
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.minus_channel
        #MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        #convolutional Difference Modules
        self.diff_c4   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * 4, out_channels=self.embedding_dim, kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=2, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=1, stride=1)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        

    def get_optimal_dist(self, inputs1, inputs2):
        #img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = inputs1
        c1_2, c2_2, c3_2, c4_2 = inputs2

        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4   = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        outputs.append(_c4)
        _c4_up= resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3   = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
        outputs.append(_c3)
        _c3_up= resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2   = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        outputs.append(_c2)
        _c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1   = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        outputs.append(_c1)

        _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1), dim=1))
        _c0 = self.convd2x(_c)
        _c0 = self.dense_2x(_c0)
        _c0 = self.convd1x(_c0)
        _c0 = self.dense_1x(_c0)
        outputs.append(_c0)

        outputs.reverse()

        return outputs


    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def _init_identity_head(self, identity_head):
        """Initialize ``auxiliary_head``"""
        if identity_head is not None:
            self.with_identity_head = True
            self.identity_head = builder.build_head(identity_head)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        inputsA = inputs[:, :3, :, :]
        inputsB = inputs[:, 3:, :, :]
        xA = self.extract_feat(inputsA)
        xB = self.extract_feat(inputsB)

        textA, textB = self.get_cls_text(batch_img_metas, False)
        text_embeddingsA, x_clipA, score_mapA = self.after_extract_feat_clip(xA, textA)
        text_embeddingsB, x_clipB, score_mapB = self.after_extract_feat_clip(xB, textB)

        x_orig = [torch.cat([x_clipA[i], x_clipB[i]], dim=1) for i in range(len(x_clipA))]
        
        opt_diff = self.get_optimal_dist(x_clipA, x_clipB)    
        opt_dist = []
        [opt_dist.append(opt_diff[i+1]) for i in range(len(opt_diff)-1)]    

        x_minus = [self.minus_conv[i](torch.abs(x_clipA[i]-x_clipB[i])) for i in range(len(x_clipA))]

        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig

        if self.text_head:
            x = [text_embeddingsA,] + x_orig
        else:
            x = x_orig

        x = [torch.cat([opt_dist[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
      
        x = [self.channel_att[i](x[i]) for i in range(len(x))]
        x.append(opt_diff[0])

        seg_logits = self.decode_head.predict_with_text(x, text_embeddingsA, text_embeddingsB, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_train_with_text(self, x, textA, textB: List[Tensor],
                                            data_samples: SampleList) -> dict:
        losses = dict()
        loss_decode = self.decode_head.loss_changeclip(x, textA, textB, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _identity_head_forward_train(self, x, data_samples, loss_id):
        losses = dict()
        loss_aux = self.identity_head.loss(
            x, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_aux, loss_id))
        return losses

    def forward_dummy(self, img):
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def after_extract_feat(self, x):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff

        B, K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)
        return text_embeddings, x_orig, score_map
    
    def after_extract_feat_cat(self, x):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff

        B, K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)
        return text_embeddings, x_orig, score_map
    
    def after_extract_feat_clip(self, x, text):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        contexts_ = torch.cat([self.contexts2] * int(x[0].size()[0]), dim=0)
        text_embeddings = self.text_encoder(text.to(global_feat.device), contexts_).expand(B, -1, -1)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff


        B, K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)
        return text_embeddings, x_orig, score_map

    def get_cls_text(self, img_infos, train=True):

        textA = []
        textB = []
        for i in range(len(img_infos)):
            if train:
                foreA = ', '.join(['remote sensing image foreground objects']+img_infos[i].jsonA)
            else:
                foreA = ', '.join(['remote sensing image foreground objects']+img_infos[i]['jsonA'])
            backA = ', '.join(['remote sensing image background objects'])
            if train:
                foreB = ', '.join(['remote sensing image foreground objects']+img_infos[i].jsonB)
            else:
                foreB = ', '.join(['remote sensing image foreground objects']+img_infos[i]['jsonA'])
            backB = ', '.join(['remote sensing image background objects'])
            textA.append(torch.cat([tokenize(c, context_length=self.context_length) for c in [backA, foreA]]).unsqueeze(0))
            textB.append(torch.cat([tokenize(c, context_length=self.context_length) for c in [backB, foreB]]).unsqueeze(0))
        return torch.cat(textA, dim=0), torch.cat(textB, dim=0)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        inputsA = inputs[:, :3, :, :]
        inputsB = inputs[:, 3:, :, :]
        xA = self.extract_feat(inputsA)
        xB = self.extract_feat(inputsB)

        textA, textB = self.get_cls_text(data_samples)
        text_embeddingsA, x_clipA, score_mapA = self.after_extract_feat_clip(xA, textA)
        text_embeddingsB, x_clipB, score_mapB = self.after_extract_feat_clip(xB, textB)

        x_orig = [torch.cat([x_clipA[i], x_clipB[i]], dim=1) for i in range(len(x_clipA))]
        opt_diff = self.get_optimal_dist(x_clipA, x_clipB)
        opt_dist = []
        [opt_dist.append(opt_diff[i+1]) for i in range(len(opt_diff)-1)]

        x_minus = [self.minus_conv[i](torch.abs(x_clipA[i]-x_clipB[i])) for i in range(len(x_clipA))]
        score_map_diff = score_mapA-score_mapB

        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig

        losses = dict()
        if self.text_head:
            x = [text_embeddingsA,] + x_orig
        else:
            x = x_orig

        x = [torch.cat([opt_dist[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
        x = [self.channel_att[i](x[i]) for i in range(len(x))]
        x.append(opt_diff[0])

        losses = dict()

        loss_decode = self._decode_head_forward_train_with_text(x, text_embeddingsA, text_embeddingsB, data_samples)
        losses.update(loss_decode)

        if self.with_identity_head:
            loss_identity_sm = self._identity_head_forward_train(
                score_map_diff/self.tau, data_samples, 'aux_score_map')
            losses.update(loss_identity_sm)
            loss_identity1 = self._identity_head_forward_train(
                x[0], data_samples, 'aux_layer0')
            losses.update(loss_identity1)
            loss_identity1 = self._identity_head_forward_train(
                x[0], data_samples, 'aux_layer0')
            losses.update(loss_identity1)
            loss_identity2 = self._identity_head_forward_train(
                x[1], data_samples, 'aux_layer1')
            losses.update(loss_identity2)
            loss_identity3 = self._identity_head_forward_train(
                x[2], data_samples, 'aux_layer2')
            losses.update(loss_identity3)
            loss_identity4 = self._identity_head_forward_train(
                x[3], data_samples, 'aux_layer3')
            losses.update(loss_identity4)
            loss_identity4 = self._identity_head_forward_train(
                x[3], data_samples, 'aux_layer3')
            losses.update(loss_identity4)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(_x_orig, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        import time
        # torch.cuda.synchronize()
        start = time.time()
        inputsA = inputs[:, :3, :, :]
        inputsB = inputs[:, 3:, :, :]
        xA = self.extract_feat(inputsA)
        xB = self.extract_feat(inputsB)

        textA = []
        textB = []
        foreA = ', '.join(['remote sensing image foreground objects']+['intersection', 'railway', 'industrial area', 'commercial area', 'interchange', 'single-family residential', 'dense residential', 'campus', 'ground track field'])
        backA = ', '.join(['remote sensing image background objects'])
        foreB = ', '.join(['intersection', 'industrial area', 'commercial area', 'single-family residential', 'storage tanks', 'interchange', 'dense residential', 'parking lot', 'freeway'])
        backB = ', '.join(['remote sensing image background objects'])
        textA.append(torch.cat([tokenize(c, context_length=self.context_length) for c in [backA, foreA]]).unsqueeze(0))
        textB.append(torch.cat([tokenize(c, context_length=self.context_length) for c in [backB, foreB]]).unsqueeze(0))
        textA, textB = torch.cat(textA, dim=0), torch.cat(textB, dim=0)

        text_embeddingsA, x_clipA, score_mapA = self.after_extract_feat_clip(xA, textA)
        text_embeddingsB, x_clipB, score_mapB = self.after_extract_feat_clip(xB, textB)

        x_orig = [torch.cat([x_clipA[i], x_clipB[i]], dim=1) for i in range(len(x_clipA))]
        opt_diff = self.get_optimal_dist(x_clipA, x_clipB)   
        opt_dist = []
        [opt_dist.append(opt_diff[i+1]) for i in range(len(opt_diff)-1)]    

        x_minus = [self.minus_conv[i](torch.abs(x_clipA[i]-x_clipB[i])) for i in range(len(x_clipA))]

        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig

        if self.text_head:
            x = [text_embeddingsA,] + x_orig
        else:
            x = x_orig

        x = [torch.cat([opt_dist[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
       
        x = [self.channel_att[i](x[i]) for i in range(len(x))]
        x.append(opt_diff[0])
        data_samples = [{'img_shape': (256, 256)}]

        seg_logits = self.decode_head.predict_with_text(x, text_embeddingsA, text_embeddingsB, data_samples,
                                              self.test_cfg)
        # torch.cuda.synchronize()
        end = time.time()
        total_time = end - start
        print('total_time:{:.2f}'.format(total_time))
        return seg_logits

    def mm_slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        inputs = inputs[0].unsqueeze(0)
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))

        imgA_pil = Image.open(batch_img_metas[0]['img_path'])
        imgB_pil = Image.open(batch_img_metas[0]['img_path'].replace('/A', '/B'))

        model, preprocess = init_clip()

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                cropA = imgA_pil.crop((x1, y1, x2, y2))
                cropB = imgB_pil.crop((x1, y1, x2, y2))
                jsonA, jsonB = clip_infer(cropA, cropB, model, preprocess)

                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                batch_img_metas[0]['jsonA'] = jsonA
                batch_img_metas[0]['jsonB'] = jsonB

                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:        
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]

                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole', 'mm_slide'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        elif self.test_cfg.mode == 'mm_slide':
            seg_logit = self.mm_slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):

        assert rescale

        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)

        seg_pred = list(seg_pred)
        return seg_pred
