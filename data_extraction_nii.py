import os
import torch
import nibabel as nib
from skimage.transform import resize

def extract_data(path='/Users/mariiaeremina/Desktop/Spline_registration/',traget_extention='Pre_target_Post.nii.gz',moving_extention='Post_reg.nii.gz',resize_dim=128,patient_n=6,number_of_frames=100):
    # Step 1: List all files in the specified directory
    files = os.listdir(path)

    # Step 2: Filter files that match the required pattern
    final_reg_files = [f for f in files if f.endswith(moving_extention)]
    pre_target_files = [f for f in files if f.endswith(traget_extention)]

    # Step 3: Extract unique patient names based on the first 6 characters of the filenames
    patients = set(f[:6] for f in final_reg_files if f[:6] in {pt[:6] for pt in pre_target_files})
    target_sequences = []
    moving_sequences = []

    for patient in list(patients)[:patient_n]:
        print(patient)
        # Load the images for the current patient
        Pre_target_Final = nib.load(os.path.join(path, patient + traget_extention)).get_fdata()
        Final_reg_pre = nib.load(os.path.join(path, patient + moving_extention)).get_fdata()

        # Transpose the images
        Pre_target_Final = Pre_target_Final.transpose(2, 1, 0)
        Final_reg_pre = Final_reg_pre.transpose(2, 1, 0)

        # Resize the images to 128x128
        Pre_target_Final_resized = resize(Pre_target_Final, (Pre_target_Final.shape[0], resize_dim, resize_dim), anti_aliasing=True)
        Final_reg_pre_resized = resize(Final_reg_pre, (Final_reg_pre.shape[0], resize_dim, resize_dim), anti_aliasing=True)

        # Convert to torch tensors and trim to the first 100 slices
        target_sequences.append(torch.tensor(Pre_target_Final_resized[:number_of_frames, ...]))
        moving_sequences.append(torch.tensor(Final_reg_pre_resized[:number_of_frames, ...]))

    # Find the minimum number of frames across all sequences
    min_frames = min(min(seq.shape[0] for seq in target_sequences), 
                    min(seq.shape[0] for seq in moving_sequences))
    return target_sequences, moving_sequences, min_frames
