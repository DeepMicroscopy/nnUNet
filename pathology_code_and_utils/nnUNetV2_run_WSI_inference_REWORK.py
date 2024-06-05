#################################################################
### NNUNET WSI INFERENCE WITH HALF OVERLAP
#################################################################
### Joey Spronck
# nnUNet by default does half overlap if the given patch is bigger than the model's patch size.
# This means that on the borders there is no or 1x overlap (1 or 2 predictions),
# while in the inside there are 4 predictions for each pixel.

# In this version of nnUNet WSI inference, we crop this border off the sampled patch, and only write the inner part
# This approach becomes more efficient if bigger patches are sampled, because this increases the inner/outer ratio
# To prevent inference on large empty patches we add a check if we can remove rows and columns,
# while preserving the half overlap of nnUNet's sliding window approach

# In this pipeline I simplified the sampling method a lot
#################################################################

#################################################################
### Imports
#################################################################

import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import time
import wandb

from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.iterators.patchiterator import create_patch_iterator
from wholeslidedata.buffer.patchcommander import PatchConfiguration
from wholeslidedata.interoperability.asap.imagewriter import WholeSlideMaskWriter
from wholeslidedata.samplers.utils import crop_data

from nnunetv2.utilities.file_path_utilities import load_json
from nnunetv2.training.nnUNetTrainer.variants.pathology.nnUNetTrainer_custom_dataloader_test import nnUNetTrainer_custom_dataloader_test
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

#################################################################
### Functions and utilities
#################################################################
current_os = "w" if os.name == "nt" else "l"
other_os = "l" if current_os == "w" else "w"

def convert_path(path, to=current_os):
    """
    This function converts paths to the format of the desired platform.
    By default it changes paths to the platform you are cyurerntly using.
    This way you do not need to change your paths if you are executing your code on a different platform.
    It may however be that you mounted the drives differently.
    In that case you may need to change that in the code below.
    """
    if to in ["w", "win", "windows"]:
        path = path.replace("/mnt/pa_cpg", "Y:")
        path = path.replace("/data/pathology", "Z:")
        path = path.replace("/mnt/pa_cpgarchive1", "W:")
        path = path.replace("/mnt/pa_cpgarchive2", "X:")
        path = path.replace("/", "\\")
    if to in ["u", "unix", "l", "linux"]:
        path = path.replace("Y:", "/mnt/pa_cpg")
        path = path.replace("Z:", "/data/pathology")
        path = path.replace("W:", "/mnt/pa_cpgarchive1")
        path = path.replace("X:", "/mnt/pa_cpgarchive2")
        path = path.replace("\\", "/")
    return path

def norm_01(x_batch): # Use this for models trained on 0-1 scaled data
    x_batch = x_batch / 255
    x_batch = x_batch.transpose(3, 0, 1, 2)
    return x_batch

def z_norm(x_batch): # use this for default nnunet models, using z-score normalized data
    mean = x_batch.mean(axis=(-2,-1), keepdims=True)
    std = x_batch.std(axis=(-2,-1), keepdims=True)
    x_batch = ((x_batch - mean) / (std + 1e-8))
    x_batch = x_batch.transpose(3, 0, 1, 2)
    return x_batch

def ensemble_softmax_list(x_batch):
    logits_list = predictor.get_logits_list_from_preprocessed_data(torch.tensor(x_batch, dtype=torch.float32))
    softmax_list = [trainer.label_manager.apply_inference_nonlin(logits).numpy() for logits in logits_list]
    return softmax_list

def array_to_formatted_tensor(array):
    # array = np.expand_dims(array.transpose(2, 0, 1), 0)
    array = array.transpose(1, 0, 2, 3)
    return torch.tensor(array) # need (1, classes, w/h, h/w)

def softmax_list_and_mean_to_uncertainty(softmax_list, softmax_mean):
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    uncertainty_loss_per_pixel_list = []
    for softmax in softmax_list:
        log_softmax = np.log(softmax + 0.00000001)
        uncertainty_loss_per_pixel = loss(array_to_formatted_tensor(log_softmax),
                                          array_to_formatted_tensor(softmax_mean))
        uncertainty_loss_per_pixel_list.append(uncertainty_loss_per_pixel)
    uncertainty = torch.cat(uncertainty_loss_per_pixel_list).mean(dim=0)
    return uncertainty

def get_trim_indexes(y_batch):
    """
    Using the y_mask / tissue-background mask we can check if there are
    full empty rows and columns with a width of half the model patch size.
    We check this in half model patch size increments because otherwise
    we screw up the half overlap approach from nnunet (resulting in inconsistent
    overlap thoughout the WSI).
    We will still need 1 row or column that is empty to make sure the parts that
    do have tissue have 4x overlap
    """
    y = y_batch[0]
    r_is_empty = [not y[start:end].any() for start, end in zip(half_patch_size_start_idxs, half_patch_size_end_idxs)]
    c_is_empty = [not y[:, start:end].any() for start, end in zip(half_patch_size_start_idxs, half_patch_size_end_idxs)]

    empty_rs_top = 0
    for r in r_is_empty:
        if r == True:
            empty_rs_top += 1  # count empty rows
        else:
            trim_top_half_idx = empty_rs_top - 1  # should always include a single empty row, since we need the overlap
            trim_top_half_idx = np.clip(trim_top_half_idx, 0, None)  # cannot select regiouns outside sampled patch
            trim_top_idx = half_patch_size_start_idxs[trim_top_half_idx]
            break

    empty_rs_bottom = 0
    for r in r_is_empty[::-1]:
        if r == True:
            empty_rs_bottom += 1
        else:
            trim_bottom_half_idx = empty_rs_bottom - 1
            trim_bottom_half_idx = np.clip(trim_bottom_half_idx, 0, None)
            trim_bottom_idx = half_patch_size_end_idxs[::-1][trim_bottom_half_idx]  # reverse index
            break

    empty_cs_left = 0
    for c in c_is_empty:
        if c == True:
            empty_cs_left += 1
        else:
            trim_left_half_idx = empty_cs_left - 1
            trim_left_half_idx = np.clip(trim_left_half_idx, 0, None)
            trim_left_idx = half_patch_size_start_idxs[trim_left_half_idx]
            break

    empty_cs_right = 0
    for c in c_is_empty[::-1]:
        if c == True:
            empty_cs_right += 1
        else:
            trim_right_half_idx = empty_cs_right - 1
            trim_right_half_idx = np.clip(trim_right_half_idx, 0, None)
            trim_right_idx = half_patch_size_end_idxs[::-1][trim_right_half_idx]
            break

    return trim_top_idx, trim_bottom_idx, trim_left_idx, trim_right_idx

def csv_to_matches(path_df):
    df = pd.read_csv(path_df)
    if len(df.columns) > 2:
        raise ValueError("The DataFrame should have only 2 columns: first contains image path and second contains mask path.")
    matches = list(df.itertuples(index=False, name=None))
    return matches

def return_matches_to_run(matches, output_folder):
    """
    The whole pipeline is costructed in such a way that a <name>_runtime.txt file is created once a python script started to work on this WSI + mask match. This means that this file should not be processed anymore by another python script that is run in parralel (also doesnt need to be copied to the local machine (chis is the case on our computing cluster))
    """
    runtime_stems = [file[:-12] for file in os.listdir(output_folder) if file.endswith('_runtime.txt')]
    imgs, _ = zip(*matches)
    img_stems = [Path(file).stem for file in imgs]
    matches_to_run_idx = [i for i in range(len(img_stems)) if img_stems[i] not in runtime_stems]
    matches_to_run = [matches[i] for i in matches_to_run_idx]

    if len(matches_to_run) == 0:
        print(f"\nAll files have been processed already, see {output_folder}")
    else:
        print(f"\nReturning {len(matches_to_run)} matches that are not finished yet")
    return matches_to_run

#################################################################
### CHANGE THINGS HERE
#################################################################

### MODEL SETTINGS
model_base_path = '/data/pathology/projects/pathology-lung-TIL/nnUNet_v2/data/nnUNet_results/Dataset008_PDL1_simplified_and_union_annotations/nnUNetTrainer_WSD_wei_i0_nnunet_aug__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d'
norm = norm_01 # z_norm or norm_01, norm_01 for models trained on 0-1 scaled data, z_norm for default nnunet models, using z-score normalized data 
output_minus_1 = True # Set to True if you want to subtract 1 from the argmax (for example when label 0 is ignore and label 1 is background, which you want to be 0 for visualisation purposes)

### OUTPUT FOLDER AND DATASET NAME
dataset_name = 'V2_PDL1_test' # name that gets added to folder names and file names
output_folder = Path('/data/pathology/projects/pathology-lung-TIL/nnUNet_raw_data_base/inference_results/v2_Task008_PDL1_simplified_and_union_annotations_wei_i0_nnunet_aug_TEST_SET')
local_output_folder = Path('/home/user/workdir')
os.makedirs(output_folder, exist_ok=True)
os.makedirs(local_output_folder, exist_ok=True)

### MATCHES YOU WANT TO RUN
# This is a list of tuples, where each tuple contains the path to the wsi and the path to the tissue mask
# Example: [('path/to/wsi1', 'path/to/tissue_mask1'), ('path/to/wsi2', 'path/to/tissue_mask2')]
all_matches = csv_to_matches(convert_path(r"Z:\archives\lung\ignite_retrospective_radboudumc\csv\TEMP_he_mask_paths.csv"))
matches_to_run = return_matches_to_run(all_matches, output_folder)

# image_path = convert_path(r"Z:\archives\lung\pembro_rt\images\HE\IG_S02_P000001_C0001_B101_V01_T01_L01_A15_E02.tif")
# mask_path = convert_path(r"Z:\archives\lung\pembro_rt\tissue_masks\HE\IG_S02_P000001_C0001_B101_V01_T01_L01_A15_E02_tissue.tif")
# matches_to_run = [(image_path, mask_path),]

rerun_unfinished = False # [READ THIS -->] this will check if the '<stem>_runtime.txt file is present in the outputfolder. If it is not it means that this file is not processed fully. IMPORTANT: if this is set to True multiple parallel jobs cannot track which slides are being processed, set this to false if you want to run multiple jobs in a parralel way. Alternatively, remove all unfinished files in the output folder (the files that do nt have a .runtime.txt file). Default is False 

### SAMPLING STUFF
spacing = 0.5 # spacing for batch_shape spacing and annotation_parser output_spacing (leave its processing spacing on 4 or higher)
model_patch_size = 512 # input size of model (should be square)
sampler_patch_size = 4 * model_patch_size # size of the sampled patch

### WANDB
wandb = False
wandb_api_key = ''

#################################################################
### AUTOMATIC STUFF (no need to change things below)
#################################################################

#################################################################
### LOAD MODEL
#################################################################
print('\nModel path:')
print(model_base_path, '\n')

plans_dict = load_json(os.path.join(model_base_path, 'plans.json'))
dataset_dict = load_json(os.path.join(model_base_path, 'dataset.json'))

trainer = nnUNetTrainer_custom_dataloader_test(plans_dict, '2d', 0, dataset_dict) # we need a trainer for a single fold to make use of its inbuilt functions

predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_gpu=True,
    device=torch.device('cuda', 0),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=False
)

predictor.initialize_from_trained_model_folder(
    model_base_path,
    use_folds=(0, 1, 2, 3, 4),
    checkpoint_name='checkpoint_best.pth',
)

#################################################################
### AUTO CONFIG
#################################################################
half_model_patch_size=model_patch_size//2
assert sampler_patch_size % half_model_patch_size == 0 # needed for correct half overlap
# due to half overlap there is half the model patch size without overlap on all 4 sides of the sampled patch
output_patch_size = sampler_patch_size - 2 * half_model_patch_size # use this as your annotation_parser shape

# following is later used to check if we can remove big empty parts of the sampled patch before inference
sampler_patch_size_range = list(range(sampler_patch_size))
half_patch_size_start_idxs = sampler_patch_size_range[0::half_model_patch_size]
half_patch_size_end_idxs = [idx + half_model_patch_size for idx in half_patch_size_start_idxs]
wsm_writer = None

#################################################################
### WANDB
#################################################################
if wandb:
    import datetime
    print('Wandb init')
    date = datetime.datetime.now().strftime("%Y%m%d")

    wandb.login(key=wandb_api_key)
    wandb.init(
        project=f'nnUNet_inference_checks', 
        name = f'sample: {sampler_patch_size} patch: {model_patch_size} date: {date}'
        )

##################################################################################################################################
## LOOP
##################################################################################################################################
print('\n\n\n####### START OF LOOP #######')
for idx_match, (image_path, mask_path) in enumerate(matches_to_run):
    print(f'\n[NEXT MATCH] [{idx_match}/{len(matches_to_run)})]:', '\n\t', image_path, mask_path)
    
    ### CHECK IF WE NEED TO PROCESS THIS FILE
    wsm_path = output_folder / (image_path.stem + '_nnunet.tif')
    wsu_path = output_folder / (image_path.stem + '_uncertainty.tif')
    if rerun_unfinished:
        if os.path.isfile(output_folder / (image_path.stem + '_runtime.txt')):
            print(f'[SKIPPING] {image_path.stem} is processed already', flush=True)
            continue  # continue to next match
    if os.path.isfile(wsm_path) and os.path.isfile(wsu_path):
        print(f'[SKIPPING] {image_path.stem} is processed already or currently being processed', flush=True)
        continue  # continue to next match
    # Immediately lock files
    open(wsm_path, 'w').close()
    open(wsu_path, 'w').close()

    print(f'[RUNNING] {image_path.stem}\n', flush=True)

    ### LOAD WSI AND GET SAMPLER/WRITER SETTINGS
    with WholeSlideImage(image_path, backend='asap') as wsi:
        shape = wsi.shapes[wsi.get_level_from_spacing(spacing)]
        real_spacing = wsi.get_real_spacing(spacing)
        downsampling = wsi.get_downsampling_from_spacing(spacing)
        offset = int((output_patch_size // 2) * downsampling) # this was difficult...

    patch_configuration = PatchConfiguration(patch_shape=(sampler_patch_size,sampler_patch_size,3),
                                            spacings=(spacing,),
                                            overlap=(model_patch_size,model_patch_size),
                                            offset=(int(offset), int(offset)),
                                            center=True)

    # Create new writer and file
    start_time = time.time()
    wsm_writer = WholeSlideMaskWriter()  # whole slide mask
    wsu_writer = WholeSlideMaskWriter()  # whole slide uncertainty
    # Create files
    wsm_path_local = local_output_folder / (image_path.stem + '_nnunet.tif')
    wsu_path_local = local_output_folder / (image_path.stem + '_uncertainty.tif')
    wsm_writer.write(path=wsm_path_local, spacing=real_spacing, dimensions=shape,
                    tile_shape=(output_patch_size, output_patch_size))
    wsu_writer.write(path=wsu_path_local, spacing=real_spacing,
                    dimensions=shape, tile_shape=(output_patch_size, output_patch_size))

    #################################################################
    ### RUN
    #################################################################
    print('\nInitiating iterator...')

    with create_patch_iterator(image_path=image_path,
        mask_path=mask_path,
        patch_configuration=patch_configuration,
        backend='asap',
        cpus=4) as patch_iterator:

        print('Iterator initiated', flush=True)
        print('Starting inference...\n\n', flush=True)

            # time calling next start
        time_pre_next = time.time()

        for idx_batch, (x_batch, y_batch, info) in enumerate(patch_iterator):
            time_post_next = time.time()
            duration_next = time_post_next - time_pre_next
                # time calling next end

                # time patch processing start
            time_pre_patch_processing = time.time()

            ### Print progress
            # if idx_batch%10==0: 
            print(f'\t[processing tile {idx_batch}/{len(patch_iterator)}] ...', flush=True)

            
            ### Trim check and prep
                # time prep start
            time_pre_prep = time.time()
            x_batch = x_batch[0]
            y_batch = y_batch[0]
            trim_top_idx, trim_bottom_idx, trim_left_idx, trim_right_idx = get_trim_indexes(y_batch)
            x_batch_maybe_trimmed = x_batch[:, trim_top_idx : trim_bottom_idx, trim_left_idx: trim_right_idx, :]
            prep = norm(x_batch_maybe_trimmed)
            time_post_prep = time.time()
            duration_prep = time_post_prep - time_pre_prep
                # time prep end
            ###

            ### Predict and uncertainty
                # time predict start
            time_pre_predict = time.time()
            softmax_list = ensemble_softmax_list(prep)
            softmax_mean = np.array(softmax_list).mean(0)
            pred_output_maybe_trimmed = softmax_mean.argmax(axis=0)-(1 if output_minus_1 else 0)
            time_post_predict = time.time()
            duration_predict = time_post_predict - time_pre_predict
                # time predict end
            ###

            ### Uncertainty
                # time uncertainty start
            time_pre_uncertainty = time.time()
            uncertainty = softmax_list_and_mean_to_uncertainty(softmax_list, softmax_mean)
            uncertainty_output_maybe_trimmed = np.array((uncertainty.clip(0, 4) / 4 * 255).int()) 
            time_post_uncertainty = time.time()
            duration_uncertainty = time_post_uncertainty - time_pre_uncertainty
                # time uncertainty end
            ###

            ### Patch wrangling and writing
                # time patch wrangling start
            time_pre_patch_wrangling = time.time()
            # Reconstruct possible trim
            pred_output = np.zeros((sampler_patch_size, sampler_patch_size))
            pred_output[trim_top_idx : trim_bottom_idx, trim_left_idx: trim_right_idx] = pred_output_maybe_trimmed
            uncertainty_output = np.zeros((sampler_patch_size, sampler_patch_size))
            uncertainty_output[trim_top_idx: trim_bottom_idx, trim_left_idx: trim_right_idx] = uncertainty_output_maybe_trimmed
            # Only write inner part
            pred_output_inner = crop_data(pred_output, [output_patch_size, output_patch_size])
            uncertainty_output_inner = crop_data(uncertainty_output, [output_patch_size, output_patch_size])
            y_batch_inner = crop_data(y_batch[0], [output_patch_size, output_patch_size]).astype('int64')
            # Get patch point
            x_coord, y_coord = info['x']//downsampling, info['y']//downsampling # convert coordinates to writing spacing
            x_coord, y_coord = x_coord - output_patch_size/2, y_coord - output_patch_size/2 # from middle point to upper left point of tile to write
            time_post_patch_wrangling = time.time()
            duration_patch_wrangling = time_post_patch_wrangling - time_pre_patch_wrangling
                # time patch wrangling end
            ###

            ### Write tile
                # time writing start
            time_pre_writing = time.time()
            wsm_writer.write_tile(tile=pred_output_inner * y_batch_inner, coordinates=(int(x_coord), int(y_coord)))
            wsu_writer.write_tile(tile=uncertainty_output_inner * y_batch_inner, coordinates=(int(x_coord), int(y_coord)))
            time_post_writing = time.time()
            duration_writing = time_post_writing - time_pre_writing
                # time writing end
            ###
    
            time_post_patch_processing = time.time()
            duration_patch_processing = time_post_patch_processing - time_pre_patch_processing
                # time patch processing end

            if wandb:
                print("\t\t[LOGGING] to wandb")
                wandb.log({
                    "duration_next": duration_next,
                    "NORM_duration_next": duration_next / (output_patch_size**2) ,
                    "duration_prep": duration_prep,
                    "NORM_duration_prep": duration_prep / (output_patch_size**2) ,
                    "duration_predict": duration_predict,
                    "NORM_duration_predict": duration_predict / (output_patch_size**2) ,
                    "duration_uncertainty": duration_uncertainty,
                    "NORM_duration_uncertainty": duration_uncertainty / (output_patch_size**2) ,
                    "duration_patch_wrangling": duration_patch_wrangling,
                    "NORM_duration_patch_wrangling": duration_patch_wrangling / (output_patch_size**2),
                    "duration_writing": duration_writing,
                    "NORM_duration_writing": duration_writing / (output_patch_size**2),
                    "duration_patch_processing": duration_patch_processing,
                    "NORM_duration_patch_processing": duration_patch_processing / (output_patch_size**2),
                    "duration_total": time_post_writing - time_pre_next,
                    "NORM_duration_total": (time_post_writing - time_pre_next) / (output_patch_size**2) 
                })
                
                # time calling next start
            time_pre_next = time.time()
        print('[PROCESSED ALL TILES]', flush=True)

    print('[WRITING and TRANSFERING] inference and uncertainty masks', flush=True)
    wsm_writer.save()  # if done save image
    shutil.copyfile(wsm_path_local, wsm_path)
    wsu_writer.save()  # if done save image
    shutil.copyfile(wsu_path_local, wsu_path)

    # Save runtime
    end_time = time.time()
    run_time = end_time - start_time
    text_file = open(output_folder / (image_path.stem + '_runtime.txt'), "w")
    text_file.write(str(run_time))
    text_file.close()
if wsm_writer == None:
    print('\n\n[NO FILES TO PROCESS] \n\n\n', flush=True)

print('[COMPLETELY DONE]\nIf there are remaining files that are not being processed right now, set "rerun_unfinished" to True right above the ### RUN section')
print('\n\n\n[Potential incoming multiprocessing error]\n\n')
