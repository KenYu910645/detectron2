# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import io
import itertools
import json
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
from typing import Optional
from PIL import Image
from tabulate import tabulate
import torch

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from .evaluator import DatasetEvaluator

import shutil
import sys
sys.path.append('/home/lab530/KenYu/detectron2/detectron2')
from utils.visualizer import Visualizer, ColorMode
import cv2

logger = logging.getLogger(__name__)


class COCOPanopticEvaluator(DatasetEvaluator):
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(self, dataset_name: str, output_dir: Optional[str] = None):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }

        self._output_dir = output_dir
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)
        
        # Clean output directory
        self.pred_dir = os.path.join(self._output_dir, "predictions")
        if os.path.exists(self.pred_dir): shutil.rmtree(self.pred_dir)
        os.makedirs(self.pred_dir)
        print(f"[panoptic_evaluation.py]Clean output directory: {self.pred_dir}")
        
        # Evalute depth estimation
        self.RSE = 0
        self.RAE = 0
        self.IRMSE = 0
        self.SILog = 0
        self.ACC_1 = 0
        self.ACC_2 = 0
        self.ACC_3 = 0
        
    
    def reset(self):
        self._predictions = []

    def _convert_category_id(self, segment_info):
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        return segment_info

    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb
        for input, output in zip(inputs, outputs):
            depth_map = output["depth"].squeeze()
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy()
            if segments_info is None:
                # If "segments_info" is None, we assume "panoptic_img" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to panoptic_img since the official
                # evaluation script uses 0 for VOID label.
                label_divisor = self._metadata.label_divisor
                segments_info = []
                for panoptic_label in np.unique(panoptic_img):
                    if panoptic_label == -1:
                        # VOID region.
                        continue
                    pred_class = panoptic_label // label_divisor
                    isthing = (
                        pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    )
                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": int(pred_class),
                            "isthing": bool(isthing),
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                panoptic_img += 1

            file_name     = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            file_name_dep = os.path.splitext(file_name)[0] + "_depth.png"
            file_name_new = os.path.splitext(file_name)[0] + "_new.png"
            
            # print(f"depth_map.cpu().numpy().shape = {depth_map.cpu().numpy().shape}")
            # print(f"id2rgb(panoptic_img).shape = {id2rgb(panoptic_img).shape}")
            
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                panoptic_data = out.getvalue()
            
            with io.BytesIO() as out:
                Image.fromarray(depth_map.cpu().numpy()).convert('RGB').save(out, format="PNG")
                depth_data    = out.getvalue()

            segments_info = [self._convert_category_id(x) for x in segments_info]
            
            # Get Depth Estimation in instance(Avg out)
            for seg in segments_info:
                seg["depth"] = depth_map[ panoptic_img == seg["id"] ].mean().item()
            
            # Use Demo visualizer to output the result
            # image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            
            # print(f"self._metadata = {self._metadata.stuff_classes}") # sidewalk, road
            # print(f"self._metadata = {self._metadata.stuff_colors}") # sidewalk, road
            
            self._metadata.stuff_colors[ self._metadata.stuff_classes.index("sidewalk") ] = (102, 66, 102)# (214, 213, 183)
            # self._metadata.stuff_colors[ self._metadata.stuff_classes.index("road")     ] = (222, 211, 140)
            self._metadata.stuff_colors[ self._metadata.stuff_classes.index("terrain")  ] = (137, 190, 178)
            # self._metadata.stuff_colors[ self._metadata.stuff_classes.index("terrain")  ] = (222, 211, 140)
            self._metadata.stuff_colors[ self._metadata.stuff_classes.index("vegetation")  ] = (63, 79, 57) # (102, 128,  89)
            self._metadata.stuff_colors[ self._metadata.stuff_classes.index("traffic light")  ] = (113, 104, 55) # (102, 128,  89)
            self._metadata.stuff_colors[ self._metadata.stuff_classes.index("traffic sign")   ] = (113, 104, 55) # (102, 128,  89)
            
            # Output Panoptic Result
            with open(os.path.join(self.pred_dir, file_name_png), "wb") as f:
                f.write(panoptic_data)
            
            # Output Depth Estimation Result
            with open(os.path.join(self.pred_dir, file_name_dep), "wb") as f:
                f.write(depth_data)
            
            # Output Panoptic DepthLab Result
            visualizer = Visualizer(input['image'].permute(1, 2, 0).cpu().numpy(), self._metadata, instance_mode = ColorMode.IMAGE)
            vis_output = visualizer.draw_panoptic_seg_predictions(torch.from_numpy(panoptic_img), segments_info)
            vis_output.save(os.path.join(self.pred_dir, file_name_new))
            
            gt_depth_dir = "/home/lab530/KenYu/cityscapes/disparity/val/"
            camera_dir   = "/home/lab530/KenYu/cityscapes/camera/val/"
            
            # Get Depth Estimation Evaluation
            pd_depth_path = os.path.join(self.pred_dir, "_".join(file_name.split("_")[:3]) + "_leftImg8bit_depth.png")
            gt_depth_path = os.path.join(gt_depth_dir, file_name.split("_")[0], "_".join(file_name.split("_")[:3]) + "_disparity.png")
            camera_path   = os.path.join(camera_dir  , file_name.split("_")[0], "_".join(file_name.split("_")[:3]) + "_camera.json")
            # print(f"gt_depth_path = {gt_depth_path}")
            # print(f"camera_path = {camera_path}")
            camera        = json.load( open(camera_path) )

            # Process the ground truth depth map
            dis_map = load_disparity(gt_depth_path)
            gt_depth = convert_disparity_to_depth(dis_map, camera)

            # Load depth prediction result 
            pd_depth = cv2.imread(pd_depth_path, cv2.IMREAD_GRAYSCALE)

            non_zero_indices = np.nonzero(gt_depth)
            gt_depth = gt_depth[ non_zero_indices ]
            pd_depth = pd_depth[ non_zero_indices ]
            
            # Avoid zero depth in prediction map
            pd_depth = np.where(pd_depth == 0, 0.1, pd_depth)
            
            assert pd_depth.shape == gt_depth.shape, "Depth maps must have the same shape"    

            self.RSE   += calculate_rse    (gt_depth, pd_depth) / 500
            self.RAE   += calculate_rae    (gt_depth, pd_depth) / 500
            self.IRMSE += calculate_irmse  (gt_depth, pd_depth) / 500
            self.SILog += calculate_silog  (gt_depth, pd_depth) / 500
            self.ACC_1 += calculate_depth_threshold_accuracy(gt_depth, pd_depth, threshold=1.25) / 500
            self.ACC_2 += calculate_depth_threshold_accuracy(gt_depth, pd_depth, threshold=1.25**2) / 500
            self.ACC_3 += calculate_depth_threshold_accuracy(gt_depth, pd_depth, threshold=1.25**3) / 500
            
            self._predictions.append(
                {
                    "image_id": input["image_id"],
                    "file_name": file_name_png,
                    "file_name_depth": file_name_dep,
                    "png_string": panoptic_data,
                    "depth_string": depth_data,
                    "segments_info": segments_info,
                }
            )

    def evaluate(self):
        comm.synchronize()
        
        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json   = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir_tmp:
            # logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            pred_dir = os.path.join(self._output_dir, "predictions")
            
            print(f"RSE = {self.RSE}")
            print(f"RAE = {self.RAE}")
            print(f"IRMSE = {self.IRMSE}")
            print(f"SILog = {self.SILog}")
            print(f"ACC_1 = {self.ACC_1}")
            print(f"ACC_2 = {self.ACC_2}")
            print(f"ACC_3 = {self.ACC_3}")
            
            # for p in self._predictions:
            #     # Output Panoptic Result
            #     with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
            #         f.write(p.pop("png_string"))
                
            #     # Output Depth Estimation Result
            #     with open(os.path.join(pred_dir, p["file_name_depth"]), "wb") as f:
            #         f.write(p.pop("depth_string"))

            # Load ground true annotations and make it become the prediction result
            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions

            # Output prediction json file
            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            from panopticapi.evaluation import pq_compute

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        results = OrderedDict({"panoptic_seg": res})
        _print_panoptic_results(pq_res)

        return results


def _print_panoptic_results(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Panoptic Evaluation Results:\n" + table)

def calculate_rse(gt_depth, pd_depth):
    # Calculate squared differences between predicted and true values
    squared_diff = np.square(pd_depth - gt_depth)
    
    # Calculate squared differences between true values and their mean
    squared_diff_mean = np.square(gt_depth - np.mean(gt_depth))
    
    # Calculate RSE
    rse = np.sum(squared_diff) / np.sum(squared_diff_mean)
    
    return rse

def calculate_rae(gt_depth, pd_depth):
    # Calculate absolute differences between predicted and true values
    abs_diff = np.abs(pd_depth - gt_depth)
    
    # Calculate absolute differences between true values and their mean
    abs_diff_mean = np.abs(gt_depth - np.mean(gt_depth))
    
    # Calculate absErrorRel
    abs_error_rel = np.sum(abs_diff) / np.sum(abs_diff_mean)
    
    return abs_error_rel

def calculate_irmse(gt_depth, pd_depth):    
    # print(f"np.any(gt_depth == 0) = {np.any(gt_depth == 0)}")
    # print(f"np.any(pd_depth == 0) = {np.any(pd_depth == 0)}")
    # Calculate inverse depth arrays
    inv_gt_depth = 1.0 / gt_depth
    inv_pd_depth = 1.0 / pd_depth
    
    # Calculate squared differences between inverse depths
    squared_diff = np.square(inv_pd_depth - inv_gt_depth)
    
    # Calculate mean squared error
    mse = np.mean(squared_diff)
    
    # Calculate iRMSE
    irmse = np.sqrt(mse)
    
    # Scale iRMSE to [1/km] units
    irmse_scaled = irmse * 1000.0
    
    return irmse_scaled

def calculate_silog(gt_depth, pd_depth):
    # Compute logarithm of predicted and ground truth depth values
    log_pd_depth = np.log10(pd_depth)
    log_gt_depth = np.log10(gt_depth)

    # Calculate absolute logarithmic differences
    abs_log_diff = np.abs(log_pd_depth - log_gt_depth)

    # Compute mean absolute logarithmic difference
    mean_abs_log_diff = np.mean(abs_log_diff)

    # Scale mean absolute logarithmic difference by 100 to obtain SILog
    silog = mean_abs_log_diff * 100

    return silog

def calculate_depth_threshold_accuracy(gt_depth, pd_depth , threshold=1.25):
    
    # Calculate the maximum ratio between predicted and ground truth depths
    max_ratio = np.maximum(pd_depth / gt_depth, gt_depth / pd_depth)
    
    # Count the number of pixels below the threshold
    num_below_threshold = np.sum(max_ratio < threshold)
    
    # Calculate the depth threshold accuracy as a percentage
    depth_threshold_accuracy = (num_below_threshold / max_ratio.size) * 100
    
    return depth_threshold_accuracy

def load_disparity(disparity_path):
    dis_label = cv2.imread(disparity_path, cv2.IMREAD_UNCHANGED) # read the 16-bit disparity png file
    dis_label = np.array(dis_label).astype(np.float) 
    dis_label[dis_label > 0] = (dis_label[dis_label > 0] - 1) / 256 # convert the png file to real disparity values, according to the official documentation. 
    return dis_label

def convert_disparity_to_depth(dis_label, camera):
    # Convert disparity map to depth map
    # print(camera)
    depth = camera['extrinsic']['baseline'] * camera['intrinsic']['fx'] / dis_label

    # zero mean don't care pixels
    depth[depth == np.inf] = 0
    depth[depth == np.nan] = 0
    # Cap at 100m
    depth = np.minimum(depth, 80)

    # Eliminate the pixels that are too far away and is on the boundary of the image
    THRESHOLD = 40
    depth[        :, :90][depth[        :, :90] > THRESHOLD] = 0
    depth[1024-180:, :  ][depth[1024-180:, :  ] > THRESHOLD] = 0
    return depth 

if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger

    logger = setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json")
    parser.add_argument("--gt-dir")
    parser.add_argument("--pred-json")
    parser.add_argument("--pred-dir")
    args = parser.parse_args()

    from panopticapi.evaluation import pq_compute

    with contextlib.redirect_stdout(io.StringIO()):
        pq_res = pq_compute(
            args.gt_json, args.pred_json, gt_folder=args.gt_dir, pred_folder=args.pred_dir
        )
        _print_panoptic_results(pq_res)
