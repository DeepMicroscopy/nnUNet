import os
import numpy as np
import cv2
import argparse
import pandas as pd
from tqdm import tqdm

def calculate_confusion_matrix(pred_mask, true_mask, num_classes):
    mask = (true_mask >= 0) & (true_mask < num_classes)
    label = num_classes * true_mask[mask].astype(int) + pred_mask[mask]
    count = np.bincount(label, minlength=num_classes**2)
    confusion_matrix = count.reshape(num_classes, num_classes)
    return confusion_matrix

def mean_iou_from_confusion_matrix(confusion_matrix, num_classes):
    ious = []
    for cls in range(num_classes):
        if cls == 3:  # Ignore label 3
            continue
        true_positive = confusion_matrix[cls, cls]
        false_positive = confusion_matrix[:, cls].sum() - true_positive
        false_negative = confusion_matrix[cls, :].sum() - true_positive
        union = true_positive + false_positive + false_negative

        if union == 0:
            iou = 1.0  # If there is no ground truth and prediction, consider IoU to be 1
        else:
            iou = true_positive / union
        ious.append(iou)
    return np.mean(ious)

def mean_iou(true_masks_dir, pred_masks_dir, scanner, num_classes=4):
    true_mask_files = set([file for file in os.listdir(true_masks_dir) if file.__contains__(scanner)])
    pred_mask_files = set([file for file in os.listdir(pred_masks_dir) if file.__contains__(scanner)])

    common_files = sorted(true_mask_files.intersection(pred_mask_files))

    if not common_files:
        raise ValueError("No common files found between the true and predicted mask directories.")

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for file_name in tqdm(common_files):
        true_mask_path = os.path.join(true_masks_dir, file_name)
        pred_mask_path = os.path.join(pred_masks_dir, file_name)

        true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

        if true_mask is None or pred_mask is None:
            raise ValueError(f"Error reading masks for {file_name}")

        confusion_matrix += calculate_confusion_matrix(pred_mask, true_mask, num_classes)

    return mean_iou_from_confusion_matrix(confusion_matrix, num_classes)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate mean IoU from true and predicted masks.")
    parser.add_argument('true_masks_dir', type=str, help="Path to the directory containing the true label masks.")
    parser.add_argument('pred_masks_dir', type=str, help="Path to the directory containing the predicted masks.")
    args = parser.parse_args()
    print(f'Prediction dir: {args.pred_masks_dir}')
    results = {}
    #for scanner in ['cs2', 'nz20', 'nz210', 'gt450', 'p1000']:
    for scanner in ['cs2']:
        mean_iou_score = mean_iou(args.true_masks_dir, args.pred_masks_dir, scanner)
        print(f'Mean IoU ({scanner}): {mean_iou_score:.4f}')
        results[scanner] = [mean_iou_score]
    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.join(args.pred_masks_dir, 'ious.csv'), sep=',', index=False)