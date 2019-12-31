import argparse
import os
import random
import torch

from PIL import ImageDraw
from torchvision.transforms import transforms
from dataset.base import Base as DatasetBase
from backbone.base import Base as BackboneBase
from bbox import BBox
from model import Model
from roi.pooler import Pooler
from config.eval_config import EvalConfig as Config


def _infer(path_to_input_image: str, path_to_output_image: str, path_to_checkpoint: str, dataset_name: str, backbone_name: str, prob_thresh: float):
    dataset_class = DatasetBase.from_name(dataset_name)
    backbone = BackboneBase.from_name(backbone_name)(pretrained=False)
    #Modififed Code for custom inference
    NUM_OF_CLASSES = 3 # Need to troubleshoot the bug. it should be 3 only
    model = Model(backbone, NUM_OF_CLASSES, pooler_mode=Config.POOLER_MODE,
                  anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
                  rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()
    model.load(path_to_checkpoint)
    CATEGORY_TO_LABEL_DICT = {
        0: 'background',
        1: 'crack', 2: 'corrosion'
    }

    with torch.no_grad():
        image = transforms.Image.open(path_to_input_image)
        image_tensor, scale = dataset_class.preprocess(image, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)

        detection_bboxes, detection_classes, detection_probs, _ = \
            model.eval().forward(image_tensor.unsqueeze(dim=0).cuda())
        detection_bboxes /= scale

        kept_indices = detection_probs > prob_thresh
        detection_bboxes = detection_bboxes[kept_indices]
        detection_classes = detection_classes[kept_indices]
        detection_probs = detection_probs[kept_indices]

        draw = ImageDraw.Draw(image)

        for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
            color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
            bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
            #category = dataset_class.LABEL_TO_CATEGORY_DICT[cls]
            #Updated for our usecase
            category = CATEGORY_TO_LABEL_DICT[cls]

            draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
            draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)

        image.save(path_to_output_image)
        print(f'Output image is saved to {path_to_output_image}')

def _infer_compare(path_to_input_image_1: str, path_to_input_image_2: str, path_to_checkpoint: str,
         dataset_name: str, backbone_name: str, prob_thresh: float):
        print('Start of Infer Compare method')
        #we will repeat the same steps as above _infer method, with below 2 Addition
        #1. Get the detection_classes & detection_probs for 2 different images and compare which images has highest Proabaility.
        #2. Draw boxes around the Highest Probaility image and discard other
        #### Define the Model #####
        dataset_class = DatasetBase.from_name(dataset_name)
        backbone = BackboneBase.from_name(backbone_name)(pretrained=False)
        #Modififed Code for custom inference
        NUM_OF_CLASSES = 21 # Need to troubleshoot the bug. it should be 3 only
        model = Model(backbone, NUM_OF_CLASSES, pooler_mode=Config.POOLER_MODE,
                  anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
                  rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()
        model.load(path_to_checkpoint)
        CATEGORY_TO_LABEL_DICT = {
            0: 'background',
            1: 'crack', 2: 'corrosion'
        }
        ##### Inferencing ######
        with torch.no_grad():
            image_1 = transforms.Image.open(path_to_input_image_1)
            image_tensor_1, scale = dataset_class.preprocess(image_1, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)

            detection_bboxes, detection_classes, detection_probs, _ = \
                model.eval().forward(image_tensor_1.unsqueeze(dim=0).cuda())
            detection_bboxes /= scale

            kept_indices = detection_probs > prob_thresh
            #detection_bboxes = detection_bboxes[kept_indices] We can ignore the bounding boxes for now
            detection_classes = detection_classes[kept_indices]
            detection_probs = detection_probs[kept_indices]

            #for 2nd image
            image_2 = transforms.Image.open(path_to_input_image_2)
            image_tensor_2, scale2 = dataset_class.preprocess(image_2, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)

            detection_bboxes_2, detection_classes_2, detection_probs_2, _ = \
                model.eval().forward(image_tensor_2.unsqueeze(dim=0).cuda())
            detection_bboxes_2 /= scale2

            kept_indices_2 = detection_probs_2 > prob_thresh
            #detection_bboxes = detection_bboxes[kept_indices] We can ignore the bounding boxes for now
            detection_classes_2 = detection_classes_2[kept_indices_2]
            detection_probs_2 = detection_probs_2[kept_indices_2]
            choosen_image = 'None'
            choosen_prob = 0.0
            #Rule 1 = Check which Image has More number of Classes (after passsing threshold)
            #Rule 2 = Check which Image has the highest number of Probs. Retrun that as primary image
            if(len(detection_classes) == len(detection_classes_2)):
                #Both have same set of classes. We will apply Rule 2
                if(max(detection_probs) > max(detection_probs_2)):
                    #We can choose 1st image
                    choosen_image = path_to_input_image_1
                    choosen_prob = detection_probs[kept_indices]
                else:
                    choosen_image = path_to_input_image_2
                    choosen_prob = detection_probs_2[kept_indices_2]
            elif(len(detection_classes) < len(detection_classes_2)):
                #Choose detection_classes_2
                choosen_image = path_to_input_image_2
                choosen_prob = detection_probs_2[kept_indices_2]
            else:
                #Choose detection_classes
                choosen_image = path_to_input_image_1
                choosen_prob = detection_probs[kept_indices]

        return choosen_image, choosen_prob
        print('done here')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--dataset', type=str, choices=DatasetBase.OPTIONS, required=True, help='name of dataset')
        parser.add_argument('-b', '--backbone', type=str, choices=BackboneBase.OPTIONS, required=True, help='name of backbone model')
        parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint')
        parser.add_argument('-p', '--probability_threshold', type=float, default=0.6, help='threshold of detection probability')
        parser.add_argument('--image_min_side', type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
        parser.add_argument('--image_max_side', type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
        parser.add_argument('--anchor_ratios', type=str, help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
        parser.add_argument('--anchor_sizes', type=str, help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
        parser.add_argument('--pooler_mode', type=str, choices=Pooler.OPTIONS, help='default: {.value:s}'.format(Config.POOLER_MODE))
        parser.add_argument('--rpn_pre_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
        parser.add_argument('--rpn_post_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
        parser.add_argument('input', type=str, help='path to input image')
        parser.add_argument('output', type=str, help='path to output result image')
        args = parser.parse_args()

        path_to_input_image = args.input
        path_to_output_image = args.output
        dataset_name = args.dataset
        backbone_name = args.backbone
        path_to_checkpoint = args.checkpoint
        prob_thresh = args.probability_threshold

        os.makedirs(os.path.join(os.path.curdir, os.path.dirname(path_to_output_image)), exist_ok=True)

        Config.setup(image_min_side=args.image_min_side, image_max_side=args.image_max_side,
                     anchor_ratios=args.anchor_ratios, anchor_sizes=args.anchor_sizes, pooler_mode=args.pooler_mode,
                     rpn_pre_nms_top_n=args.rpn_pre_nms_top_n, rpn_post_nms_top_n=args.rpn_post_nms_top_n)

        print('Arguments:')
        for k, v in vars(args).items():
            print(f'\t{k} = {v}')
        print(Config.describe())

        _infer(path_to_input_image, path_to_output_image, path_to_checkpoint, dataset_name, backbone_name, prob_thresh)

    main()
