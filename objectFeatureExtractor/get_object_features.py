from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../faster-rcnn.pytorch-master")

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_one_detection
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
import json

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3



lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


if __name__ == '__main__':

    frcnn_path = '../faster-rcnn.pytorch-master/'
    N = 4 # Num of objects to get features from
    image_dir = "/home/raulgomez/datasets/EuropeanaDates/fasterRCNN/images/"
    out_object_features_dir = "/home/raulgomez/datasets/EuropeanaDates/fasterRCNN/object_features/"
    out_detections_dir = "/home/raulgomez/datasets/EuropeanaDates/fasterRCNN/object_detections/"
    cfg_file = frcnn_path + 'cfgs/res101.yml'
    set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]'] # None (default) for pascal vgg
    load_dir = "/home/raulgomez/datasets/EuropeanaDates/fasterRCNN/models"
    net = "res101"
    dataset = "coco"
    checksession = 1
    checkepoch = 6
    checkpoint = 9771
    class_agnostic = False
    cuda = 1
    parallel_type = 0
    batch_size = 1

    if cfg_file is not None:
        cfg_from_file(cfg_file)
    if set_cfgs is not None:
        cfg_from_list(set_cfgs)

    cfg.USE_GPU_NMS = cuda

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.

    input_dir = load_dir + "/" + net + "/" + dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(checksession, checkepoch, checkpoint))

    # pascal_classes = np.asarray(['__background__',
    #                              'aeroplane', 'bicycle', 'bird', 'boat',
    #                              'bottle', 'bus', 'car', 'cat', 'chair',
    #                              'cow', 'diningtable', 'dog', 'horse',
    #                              'motorbike', 'person', 'pottedplant',
    #                              'sheep', 'sofa', 'train', 'tvmonitor'])

    pascal_classes = np.asarray(
        ['__background__', "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
         "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
         "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
         "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
         "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
         "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
         "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
         "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
         "microwave", "oven", "toaster", "sink",
         "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"])

    # initilize the network here.
    if net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=class_agnostic)
    elif net == 'res101':
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=class_agnostic)
    elif net == 'res50':
        fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=class_agnostic)
    elif net == 'res152':
        fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # pdb.set_trace()

    print("load checkpoint %s" % (load_name))

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    with torch.no_grad():
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

    if cuda > 0:
        cfg.CUDA = True

    if cuda > 0:
        fasterRCNN.cuda()

    fasterRCNN.eval()

    start = time.time()
    max_per_image = 100
    thresh = 0.05
    vis = True

    imglist = os.listdir(image_dir)
    num_images = len(imglist)

    print('Loaded Photo: {} images.'.format(num_images))

    while (num_images >= 0):
        total_tic = time.time()
        num_images -= 1

        im_file = os.path.join(image_dir, imglist[num_images])
        # im = cv2.imread(im_file)
        im_in = np.array(imread(im_file))

        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)

        # rgb -> bgr
        im = im_in[:, :, ::-1]

        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.data.resize_(1, 1, 5).zero_()
        num_boxes.data.resize_(1).zero_()

        # pdb.set_trace()
        det_tic = time.time()

        # RUN the net and get the results
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, pooled_feat = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if class_agnostic:
                    if cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        im2show = np.copy(im)

        # dict to save the detections,
        # dict keys are the detection (and the feature map) ID [0-300]
        # values will be the score
        resulting_detections = {}

        # Save cls_dets just for ploting
        cls_dets_dict = {}

        for j in xrange(1, len(pascal_classes)):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                cls_dets_dict[j] = cls_dets

                # Visualize only the detections we use (run the vis later)
                # if vis:
                #     im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)

                # Save the detections of this class in the dict
                for i,cur_det in enumerate(cls_dets):
                    resulting_detections[int(inds[int(order[int(keep[i])])])] = (cur_det[-1], j, i)
                    # This indexing is a mess but I think is ok
                    # What I want to do is recover the recognition ID [0-300] to then be able to get the feature map for this detection
                    # inds store the indices [0-300] we are considering in cls_scores
                    # then cls_scores are ordered, and order stores the ordering order
                    # then nms is applied to cls_scores, and keep stores the indices of the resulting cls_scores

        # Now get the top k detections
        cur_img_obj_feat = {}
        det_i = 0
        for key, value in sorted(resulting_detections.iteritems(), key=lambda (k, v): (v, k), reverse=True):
            if det_i == N: break
            cur_img_obj_feat[str(det_i)] = {}
            cur_img_obj_feat[str(det_i)]['features'] = pooled_feat[key].data.cpu().numpy().tolist()
            cur_img_obj_feat[str(det_i)]['class'] = pascal_classes[value[1]]
            cur_img_obj_feat[str(det_i)]['score'] = float(value[0])
            det_i += 1

            # Visualize only the detections we use
            if vis:
                im2show = vis_one_detection(im2show, pascal_classes[value[1]], cls_dets_dict[value[1]][value[2]].cpu().numpy(), 0.5)


        with open(out_object_features_dir + imglist[num_images].strip('.jpg') + '.json','w') as out_file:
            json.dump(cur_img_obj_feat, out_file)

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                         .format(num_images + 1, len(imglist), detect_time, nms_time))
        sys.stdout.flush()
        cv2.imwrite(out_detections_dir + imglist[num_images], im2show)

print("DONE")
