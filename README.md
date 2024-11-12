Repository for the implementation of [DDPM-X](https://www.researchgate.net/publication/382462715_Diffusion_X-ray_image_denoising) for Digital Subtraction Angiography (DSA). Some adjustments to improved the model efficiency come from [RCD](https://arxiv.org/pdf/2303.16425).

# Directory contents
* Code (Dir): contains all the code for the model architecture, training and inference:
    * *cfg.yaml* (and similar): Configuration file for the model, dataloader and inference scripts

    * *data_proc.py*: Given the folder with the sequences TIFs, creates the dataset of frames/patches.
    * *dataset.py*: Produces the Dataset and Dataloader for pytorch

    * *models.py* Model architecture
    * *main_multi.py*: Training(+ validation and test) script for the model, leveraging the DDP setup to work on multiple GPUs. 
    * *inference.py*: Runs denoising on a folder of images/patches given a pretrained model (.pt) and assuming Gaussian noise
    * *patcher.py*: Stitches the patches together and produces the fullsize image
 
    * *utils.py*: Various utils for the scripts.
* *.gitignore*
* *ddpmx_multi.slurm*: Slurm file to run the training on a cluster
* *inference.slurm*: Slurm file to run the inference on a cluster

# Running the code
To train the model, put the *.yaml* file in the same folder of the scripts and run  `torchrun --nproc_per_node=16 ~path_to_file/main_multi.py`, where the flag nproc_per_node controls the number of used GPUs. The model saves the checkpoints at each epoch, in order to resume training if something fails.

After the training is complete, run the *inference.py* script (no need to use torchrun since this works directly on a single GPU).

To produce the full images if patches were created, run the *patcher.py* script.