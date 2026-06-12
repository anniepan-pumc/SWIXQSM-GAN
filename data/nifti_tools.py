from pathlib import Path
import nibabel as nib
from tqdm import tqdm
import pydicom
from pydicom.dataset import FileDataset
import zipfile
import torch
import shutil
import matplotlib.pyplot as plt
import numpy as np
import csv

def load_nifti(nifti_dir):
    '''
    :param: nifti_dir: path to nifti file
    :return: torch.tensor that contained nifti image 
    '''
    #read nifti file to numpy.array
    nifti_file = nib.load(nifti_dir)
    nifti_array = nifti_file.get_fdata()
    return torch.from_numpy(nifti_array) 


def show_nifti(nifti_array, col_num=6, fig_size=15,save_path=None):
    '''
    Visulize NIFTI image by matplot and save if given save path
    '''
    nifti_array = nifti_array.transpose((1,0,2))
    num_slices = nifti_array.shape[-1]
    fig, axs = plt.subplots(nifti_array.shape[-1]//col_num, col_num, figsize=(fig_size, fig_size))
    # Plot each slice
    for i in range(num_slices):
        ax = axs[i // col_num, i % col_num]
        ax.imshow(nifti_array[:, :, i], cmap='gray')
        ax.set_title(f"Slice {i + 1}")
    plt.show()
    if save_path is not None:
        plt.savefig(f'{save_path}.png')

################################################################
################### Extract NIFTI to Folder ####################
################################################################

def unzip_all_data_with_type(data_dir, des_dir, data_type):
    data_dir = Path(data_dir)
    des_dir = Path(des_dir)
    #Ensure the destination folder exists
    des_dir.mkdir(parents=True, exist_ok=True)
    bad_zips = []

    for zip_path in data_dir.rglob("*.zip"):
        pID = (zip_path.stem).split("_")[0]
        extract_dir = des_dir/pID  # Use the same name for extraction folder
        if not extract_dir.exists():
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_file_list = zip_ref.namelist()
                    # Check if the target file is in the zip archive
                    for l in zip_file_list:
                        if f"{data_type}.nii.gz" in l:
                            zip_ref.extract(l, extract_dir)
                            print(f"Unzipped: {zip_path/l} to {extract_dir}")
                            break
            except zipfile.BadZipFile as e:
                print(f"{pID}, Error: BadZipFile - {e}")
                bad_zips.append(f"{pID}_20251_2_0.zip")
                continue
            old_dir = Path(extract_dir / l)
            try:
                shutil.move(str(old_dir),str(extract_dir))
                # deal with different directories in QSM data and SWI data 
                shutil.rmtree(str(old_dir.parent.parent)) if data_type == "QSM" or "QSM_mask" else shutil.rmtree(str(old_dir.parent))
            except FileNotFoundError:
                print(f"{old_dir} File not found!")
                pass
    csv_file_path = "badzip.csv"
    # Writing the list to a CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for element in bad_zips:
            writer.writerow([element])

################################################################
################### Convert NIFTI to Dicom #####################
################################################################
def convertNsave(arr, file_dir, file_class, pID, index=0):
    """
    `arr`: parameter will take a numpy array that represents only one slice.
    `file_dir`: parameter will take the path to save the slices
    `index`: parameter will represent the index of the slice, so this parameter will be used to put 
    the name of each slice while using a for loop to convert all the slices
    """
    
    dicom_file = FileDataset(nifti_path, {}, file_meta=pydicom.Dataset())
    arr = arr.astype('uint8')
    dicom_file.PatientID = pID
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 8
    dicom_file.BitsAllocated = 8
    dicom_file.HighBit = 7
    dicom_file.PixelRepresentation = 0
    dicom_file.PixelData = arr.tobytes()
    dicom_file.InstanceNumber = index
    dicom_file.save_as(file_dir/f'{file_class}_{index}.dcm')

def nifti2dicom_1file(nifti_path, out_dir):
    """
    This function is to convert only one nifti file into dicom series
    `nifti_path`: the path to the one nifti file
    `out_dir`: the path to output
    """

    nifti_array = load_nifti(nifti_path)
    nifti_array = nifti_array.transpose((1, 0, 2))
    number_slices = nifti_array.shape[-1]
    file_name = nifti_path.stem.replace(".nii","")
    pID = nifti_path.parent.name.split('_')[0]

    for slice_ in tqdm(range(number_slices)):
        convertNsave(nifti_array[:, :, slice_], out_dir, file_name, pID, slice_)

def nifti2dicom_all(folder_path):
    # transfer all nifti data (suffix .nii or .nii.gz) to dicom data (suffix .dcm)
    folder_path = Path(folder_path)
    for niigz_path in folder_path.rglob("*.nii.gz"):
        file_name = niigz_path.stem.replace(".nii","")
        save_dir = niigz_path.parent / f"{file_name}_dicom"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(niigz_path, save_dir)
        nifti2dicom_1file(niigz_path, save_dir)

    print("complete")


if __name__ == '__main__':
    # Example usage:
    data_dir =  '/mnt/data/ukb_swi'
    nifti_path = '/mnt/data/ukb_swi/nifti' # path where save ukb nii.gz data
    # extract SWI data from nii.gz and store in nifti_path
    unzip_all_data_with_type(data_dir/Path("SWI"),nifti_path/Path("SWI"), "SWI")
    # extract QSM data from nii.gz and store in nifti_path
    unzip_all_data_with_type(data_dir/Path("QSM"),nifti_path/Path("QSM"), "QSM")
    # extract QSM mask data from nii.gz and store in nifti_path
    unzip_all_data_with_type(data_dir/Path("QSM"),nifti_path/Path("QSM_mask"), "QSM_mask")
    #convert all nifti data to dcm
    # nifti2dicom_all(nifti_path)
    #convert one nifti data to dcm
    # nifti2dicom_1file(nifti_path/"example_subject/QSM.nii.gz", nifti_path/"example_subject" )