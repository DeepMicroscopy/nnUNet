from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw



if __name__ == "__main__":
    dataset_name = 'Dataset175_Catch'

    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    imagests = join(nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    labelsts = join(nnUNet_raw, dataset_name, 'labelsTs')

    num_train = 97

    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'non-tumor': 1, 'tumor': 2, "ignore": 3},
                          num_train, '.png', dataset_name=dataset_name)
