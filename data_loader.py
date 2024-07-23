import torch
from torch.utils.data import Dataset
import random
import torchvision.transforms.functional as FA
# Z-normalization
def z_normalize(tensor):
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True)
    return (tensor - mean) / std
# Min-Max scaling to range [0, 1]
def min_max_scale(tensor):
    min_val,_ = tensor.min(dim=0, keepdim=True)
    max_val,_ = tensor.max(dim=0, keepdim=True)
    return (tensor - min_val) / (max_val - min_val)
def rescale_angles(angles):
    rescaled_angles = [(x + 45) / 90 for x in angles]
    return rescaled_angles
def backward_rescale_angles(rescaled_angles):
    original_angles = [x * 90 - 45 for x in rescaled_angles]
    return original_angles
class AugmentationDataLoader(Dataset):
    def __init__(self, target_sequences, moving_sequences, translation_range=45):
        """
        Args:
            target_sequences (list of tensors): List of target frame sequences.
            moving_sequences (list of tensors): List of moving frame sequences.
            translation_range (int): The range of translation in pixels.
        """
        self.target_sequences = target_sequences
        self.moving_sequences = moving_sequences
        self.translation_range = translation_range

    def __len__(self):
        return len(self.target_sequences)

    def __getitem__(self, idx):
        target_seq = self.target_sequences[idx]
        moving_seq = self.moving_sequences[idx]

        augmented_target_seq = []
        augmented_moving_seq = []
        theta = []
        dx = []
        dy = []

        for t_frame, m_frame in zip(target_seq, moving_seq):
            # Apply random rotation to the target frame
            angle_t = random.uniform(-45, 45)
            rotated_t_frame = FA.rotate(t_frame.unsqueeze(0), angle_t).squeeze(0)
            
            # Apply random translation to the target frame
            trans_x_t = random.uniform(-self.translation_range, self.translation_range)
            trans_y_t = random.uniform(-self.translation_range, self.translation_range)
            translated_t_frame = FA.affine(rotated_t_frame.unsqueeze(0), angle=0, translate=(trans_x_t, trans_y_t), scale=1, shear=0).squeeze(0)
            augmented_target_seq.append(translated_t_frame)

            # Apply random rotation to the moving frame
            angle_m = random.uniform(-45, 45)
            rotated_m_frame = FA.rotate(m_frame.unsqueeze(0), angle_m).squeeze(0)
            
            # Apply random translation to the moving frame
            trans_x_m = random.uniform(-self.translation_range, self.translation_range)
            trans_y_m = random.uniform(-self.translation_range, self.translation_range)
            translated_m_frame = FA.affine(rotated_m_frame.unsqueeze(0), angle=0, translate=(trans_x_m, trans_y_m), scale=1, shear=0).squeeze(0)
            augmented_moving_seq.append(translated_m_frame)

            # Compute rotation and translation differences
            rotation_difference = angle_m - angle_t
            translation_difference_x = trans_x_m - trans_x_t
            translation_difference_y = trans_y_m - trans_y_t
            theta.append(rotation_difference)
            dx.append(translation_difference_x)
            dy.append(translation_difference_y)

        # Convert lists to tensors
        augmented_target_seq = torch.stack(augmented_target_seq)
        augmented_moving_seq = torch.stack(augmented_moving_seq)
        
        # Combine the augmented sequences into a single tensor with shape [2, D, H, W]
        #combined_augmented_seq = torch.stack([augmented_target_seq, augmented_moving_seq], dim=0)
        
        theta = torch.tensor(theta)
        dx = torch.tensor(dx)
        dy = torch.tensor(dy)
        augmented_target_seq = min_max_scale(augmented_target_seq)
        augmented_moving_seq = min_max_scale(augmented_moving_seq)
        # Z-normalize the sequences
        augmented_target_seq = z_normalize(augmented_target_seq)
        augmented_moving_seq = z_normalize(augmented_moving_seq)

        theta = rescale_angles(theta)
        dx = rescale_angles(dx)
        dy = rescale_angles(dy)

        combined_augmented_seq = torch.stack([augmented_target_seq, augmented_moving_seq], dim=0)

        return combined_augmented_seq, theta, dx, dy
    
class BackTransformation:
    def __init__(self):
        pass

    def inverse_rotate(self, frame, angle):
        return FA.rotate(frame.unsqueeze(0), -angle).squeeze(0)

    def inverse_translate(self, frame, dx, dy):
        return FA.affine(frame.unsqueeze(0), angle=0, translate=(-dx, -dy), scale=1, shear=0).squeeze(0)

    def apply_back_transform(self, combined_augmented_seq, theta, dx, dy):
        # Extract augmented_moving_seq part
        combined_augmented_seq = combined_augmented_seq.detach().numpy().copy()
        augmented_moving_seq = combined_augmented_seq[1]  # Assuming combined_augmented_seq has shape [2, D, H, W]

        theta = backward_rescale_angles(theta)
        dx = backward_rescale_angles(dx)
        dy = backward_rescale_angles(dy)
        
        transformed_frames = []
        for frame, angle, translation_x, translation_y in zip(augmented_moving_seq, theta, dx, dy):
            frame = self.inverse_translate(frame, translation_x, translation_y)
            frame = self.inverse_rotate(frame, angle)
            transformed_frames.append(frame)
        
        transformed_frames = torch.stack(transformed_frames)

        # Replace the augmented_moving_seq part in combined_augmented_seq with transformed_frames
        combined_augmented_seq[1] = transformed_frames
        
        return combined_augmented_seq
