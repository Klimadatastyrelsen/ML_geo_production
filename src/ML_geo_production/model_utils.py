#model_utils.py
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from fastai.vision.all import *
from ML_geo_production.image_utils import (
    load_central_window,
    load_dummy_mask,
    load_multi_channel_central_window,
)
from wwf.vision.timm import *
import time

from typing import List, Dict, Any, Union, Optional, Sequence


# ---------------------------------------------------------------------
# Swin + UPerNet wrapper
# ---------------------------------------------------------------------
class SwinUPerNetWrapper(nn.Module):
    """Wrapper for Swin Transformer + UPerNet from transformers"""
    def __init__(self, model_name, num_classes, n_in=3, pretrained=True, ignore_index=255):
        super().__init__()
        try:
            from transformers import AutoModelForSemanticSegmentation, UperNetConfig
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        if pretrained:
            self.model = AutoModelForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            if n_in != 3:
                self._adapt_input_channels(n_in)
        else:
            config = UperNetConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            self.model = AutoModelForSemanticSegmentation.from_config(config)
            if n_in != 3:
                self._adapt_input_channels(n_in)
    
    def _adapt_input_channels(self, n_in):
        try:
            if hasattr(self.model, 'backbone'):
                old_patch_embed = self.model.backbone.embeddings.patch_embeddings.projection
            elif hasattr(self.model, 'swin'):
                old_patch_embed = self.model.swin.embeddings.patch_embeddings.projection
            else:
                print("Warning: Could not find patch embedding layer to adapt")
                return
            
            new_patch_embed = nn.Conv2d(
                n_in,
                old_patch_embed.out_channels,
                kernel_size=old_patch_embed.kernel_size,
                stride=old_patch_embed.stride,
                padding=old_patch_embed.padding,
                bias=old_patch_embed.bias is not None
            )
            
            nn.init.kaiming_normal_(new_patch_embed.weight, mode='fan_out', nonlinearity='relu')
            if new_patch_embed.bias is not None:
                nn.init.constant_(new_patch_embed.bias, 0)
            
            if n_in >= 3 and old_patch_embed.weight.shape[1] == 3:
                with torch.no_grad():
                    new_patch_embed.weight[:, :3] = old_patch_embed.weight
            
            if hasattr(self.model, 'backbone'):
                self.model.backbone.embeddings.patch_embeddings.projection = new_patch_embed
            elif hasattr(self.model, 'swin'):
                self.model.swin.embeddings.patch_embeddings.projection = new_patch_embed
        except Exception as e:
            print(f"Warning: Could not adapt input channels: {e}")
    
    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        return logits


# ---------------------------------------------------------------------
# ConvNeXt V2 + UPerNet (transformers-based)
# ---------------------------------------------------------------------
class ConvNeXtV2UPerNetWrapper(nn.Module):
    """ConvNeXt V2 backbone + UPerNet decoder using transformers library"""
    def __init__(self, backbone_name, num_classes, n_in, pretrained=True):
        super().__init__()
        try:
            from transformers import AutoModelForSemanticSegmentation, UperNetConfig
        except ImportError:
            raise ImportError(
                "ConvNeXtV2+UPerNet requires transformers: pip install transformers"
            )
        
        self.num_classes = num_classes
        self.n_in = n_in
        
        # Map backbone names to HuggingFace model IDs
        arch = backbone_name.replace("convnextv2_", "").replace("convnext_", "")
        model_map = {
            "tiny": "openmmlab/upernet-convnext-tiny",
            "small": "openmmlab/upernet-convnext-small", 
            "base": "openmmlab/upernet-convnext-base",
            "large": "openmmlab/upernet-convnext-large",
        }
        model_name = model_map.get(arch, f"openmmlab/upernet-convnext-{arch}")
        
        if pretrained:
            self.model = AutoModelForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            if n_in != 3:
                self._adapt_input_channels(n_in)
        else:
            config = UperNetConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            self.model = AutoModelForSemanticSegmentation.from_config(config)
            if n_in != 3:
                self._adapt_input_channels(n_in)
    
    def _adapt_input_channels(self, n_in):
        """Adapt the model to handle different number of input channels"""
        try:
            # Update the config to reflect new number of channels
            if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'config'):
                self.model.backbone.config.num_channels = n_in
            
            if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'embeddings'):
                old_patch_embed = self.model.backbone.embeddings.patch_embeddings
                
                new_patch_embed = nn.Conv2d(
                    n_in,
                    old_patch_embed.out_channels,
                    kernel_size=old_patch_embed.kernel_size,
                    stride=old_patch_embed.stride,
                    padding=old_patch_embed.padding,
                    bias=old_patch_embed.bias is not None
                )
                
                nn.init.kaiming_normal_(new_patch_embed.weight, mode='fan_out', nonlinearity='relu')
                if new_patch_embed.bias is not None:
                    nn.init.constant_(new_patch_embed.bias, 0)
                
                if n_in >= 3 and old_patch_embed.weight.shape[1] == 3:
                    with torch.no_grad():
                        new_patch_embed.weight[:, :3] = old_patch_embed.weight
                
                self.model.backbone.embeddings.patch_embeddings = new_patch_embed
                
                # Also update num_channels in embeddings
                if hasattr(self.model.backbone.embeddings, 'num_channels'):
                    self.model.backbone.embeddings.num_channels = n_in
            else:
                print("Warning: Could not find patch embedding layer to adapt")
        except Exception as e:
            print(f"Warning: Could not adapt input channels: {e}")

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:],
                                mode='bilinear', align_corners=False)
        return logits


# ---------------------------------------------------------------------
# Simple wrapper to make raw models compatible with Learner-like interface
# ---------------------------------------------------------------------
class ModelWrapper:
    """
    Simple wrapper to give raw nn.Module models a Learner-like interface.
    This allows process_images.py to use learner.model consistently.
    """
    def __init__(self, model):
        self.model = model
    
    def to(self, device):
        self.model.to(device)
        return self
    
    def eval(self):
        self.model.eval()
        return self


def preload_model_states(model_paths: List[str]) -> List[Union[Dict[str, Any], Any]]:
    """
    Loads model state dictionaries from a list of file paths into main memory.
    
    This function uses memory-mapping (mmap=True) to potentially speed up 
    the disk read operation by leveraging the OS's Page Cache.

    Parameters:
    -----------
    model_paths : List[str]
        A list of full file paths to the model weight files (.pth).

    Returns:
    --------
    List[Union[Dict[str, Any], Any]]
        A list of the loaded state dictionaries (model weights).
    """
    
    preloaded_states = []
    
    print("--- Starting Model State Pre-loading ---")
    
    for i, path in enumerate(model_paths):
        print(f"Loading state {i+1}/{len(model_paths)} from disk: '{path}'...")
        
        loading_state_start = time.time()
        
        try:
            # Load the state dictionary using mmap=True for efficient disk access.
            # We map it to 'cpu' as it only needs to be in main memory (RAM) 
            # at this stage, not yet GPU memory (VRAM).
            state = torch.load(
                path, 
                map_location='cpu', 
                weights_only=False, 
                mmap=True
            )
            preloaded_states.append(state)
            
            load_time = time.time() - loading_state_start
            print(f"  -> Load successful. Took: {load_time:.4f} seconds.")
            
        except Exception as e:
            print(f"  -> ERROR loading state from {path}: {e}")
            # Optionally raise the error or continue, depending on your error handling needs
            
    print("--- Finished Model State Pre-loading ---")
    
    return preloaded_states


def create_dummy_dls(
    folder,
    size=224,
    bs=1,
    n_classes=3,
    n_in=3,
    sample_path=None,
    norm_means: Optional[Sequence[float]] = None,
    norm_stds: Optional[Sequence[float]] = None,
    data_folders: Optional[Sequence[str]] = None,
    channels: Optional[Sequence[Sequence[int]]] = None,
):
    """
    Creates a dummy DataLoaders from .tif images in the given folder.
    A dummy mask (all zeros) is provided for each image.
    This version uses rasterio to load only the central 1000 x 1000 pixels.
    
    Parameters:
    -----------
    folder : str
        Path to the folder containing .tif images (used when sample_path is None)
    size : int
        Size to resize images to
    bs : int
        Batch size
    n_classes : int
        Number of classes
    n_in : int
        Number of input channels when norm_means is not set (ImageNet defaults extended).
    sample_path : str or Path, optional
        If set, only this file is used for the dummy dataset (avoids picking another
        .tif in the folder that may have a different band count).
    norm_means, norm_stds : optional sequences
        If norm_means is set and non-empty, n_in is len(norm_means). Used only to
        determine input channel count for architecture setup.
    data_folders, channels : optional
        When both are set, load the dummy sample from multiple data folders (same
        logic as inference). Otherwise read bands from a single GeoTIFF.
        
    Returns:
    --------
    DataLoaders
        FastAI DataLoaders object
    """
    print("Creating dummy_dls")
    folder = Path(folder)
    if sample_path is not None:
        sample_path = Path(sample_path).resolve()
        if not sample_path.is_file():
            sys.exit(f"sample_path is not a file: {sample_path}")
        dl_source = sample_path.parent
        get_items_fn = lambda _: [sample_path]
    else:
        files = list(folder.glob("*.tif"))
        max_files = 2
        files = files[:max_files]
        if not files:
            sys.exit(f"No .tif image files found in folder: {folder}")
        dl_source = folder
        get_items_fn = lambda f: list(Path(f).glob("*.tif"))

    if norm_means is not None and len(norm_means) > 0:
        n_in = len([float(x) for x in norm_means])
    print(f"Creating Datablock (dummy n_in={n_in})")

    use_multi_channel = (
        data_folders is not None
        and channels is not None
        and len(data_folders) > 0
        and len(channels) > 0
    )
    if use_multi_channel:
        get_x = lambda o: load_multi_channel_central_window(
            o, data_folders, channels, window_size=1000
        )
        image_block = TransformBlock(batch_tfms=IntToFloatTensor)
    else:
        get_x = lambda o: load_central_window(o, window_size=1000, n_channels=n_in)
        image_block = ImageBlock

    dblock = DataBlock(
        blocks=(image_block, MaskBlock(codes=list(range(n_classes)))),
        get_items=get_items_fn,
        splitter=FuncSplitter(lambda o: False),  # All items go to training set.
        get_x=get_x,
        get_y=lambda o: load_dummy_mask(o, window_size=1000),
    )

    return dblock.dataloaders(dl_source, bs=bs, size=size)


def load_unet_from_state(
    model_state,
    model_name,
    input_folder,
    n_classes=3,
    n_in=3,
    device="cuda",
    sample_image_path=None,
    norm_means: Optional[Sequence[float]] = None,
    norm_stds: Optional[Sequence[float]] = None,
    data_folders: Optional[Sequence[str]] = None,
    channels: Optional[Sequence[Sequence[int]]] = None,
):
    """
    Loads a saved FastAI U-Net model from a pre-loaded state dictionary 
    and transfers it to the specified device (e.g., GPU).

    Parameters:
    -----------
    model_state : dict
        The pre-loaded state dictionary (weights) of the model.
        This dict comes from an earlier torch.load() call (e.g., state = torch.load('model.pth')).
    model_name : str
        The name of the backbone architecture (e.g., "resnet34", "swin-small-upernet").
    input_folder : str
        Path to folder with sample images (used to create the dummy DataLoader).
    n_classes : int
        Number of output classes.
    n_in : int
        Number of input channels.
    device : str
        Device to load the model on ("cuda" or "cpu").
    sample_image_path : str or Path, optional
        GeoTIFF used to build the dummy DataLoader (same band count as inference). If omitted,
        the first *.tif in input_folder is used (order not guaranteed).
    norm_means, norm_stds : optional
        When building the dummy dataloader (timm/resnet path), if norm_means is set then
        n_in is len(norm_means) (see create_dummy_dls).
    data_folders, channels : optional
        Multi-folder band selection for dummy sample loading (same as inference).

    Returns:
    --------
    Learner or nn.Module
        FastAI Learner object with loaded model, or raw model for transformer-based architectures.
    """
    
    # Extract the actual model state if it's nested in the FastAI dict structure
    state = model_state
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    
    model_name_lower = model_name.lower()
    
    # Handle Swin + UPerNet models
    if "swin" in model_name_lower and "upernet" in model_name_lower:
        print(f"Creating Swin + UPerNet architecture for: {model_name}")
        
        # Detect n_in from checkpoint if available (look for patch embedding weight)
        actual_n_in = n_in
        for key in state.keys():
            if "patch_embeddings" in key and "projection.weight" in key:
                actual_n_in = state[key].shape[1]
                if actual_n_in != n_in:
                    print(f"Detected {actual_n_in} input channels from checkpoint (config specified {n_in})")
                break
        
        # Map model names to HuggingFace model IDs
        swin_models = {
            "swin-small-upernet": "openmmlab/upernet-swin-small",
            "swin-base-upernet": "openmmlab/upernet-swin-base",
            "swin-large-upernet": "openmmlab/upernet-swin-large",
        }
        hf_model_name = swin_models.get(model_name_lower, model_name)
        
        model = SwinUPerNetWrapper(
            model_name=hf_model_name,
            num_classes=n_classes,
            n_in=actual_n_in,
            pretrained=False  # We'll load weights from state
        )
        
        # Load weights
        print("Loading state dictionary into model")
        loading_state_start = time.time()
        model.load_state_dict(state)
        print(f"loading state dictionary took : {time.time() - loading_state_start}")
        
        # Move to device and set eval mode
        print(f"Transferring model to {device}")
        model.to(device)
        model.eval()
        
        print("Returning model (wrapped for Learner compatibility)")
        return ModelWrapper(model)
    
    # Handle ConvNeXt + UPerNet models
    elif "convnext" in model_name_lower and "upernet" in model_name_lower:
        print(f"Creating ConvNeXt + UPerNet architecture for: {model_name}")
        
        # Detect n_in from checkpoint if available (look for patch embedding weight)
        actual_n_in = n_in
        for key in state.keys():
            if "patch_embeddings" in key and "weight" in key and "projection" not in key:
                actual_n_in = state[key].shape[1]
                if actual_n_in != n_in:
                    print(f"Detected {actual_n_in} input channels from checkpoint (config specified {n_in})")
                break
            elif "embeddings.patch_embeddings.weight" in key:
                actual_n_in = state[key].shape[1]
                if actual_n_in != n_in:
                    print(f"Detected {actual_n_in} input channels from checkpoint (config specified {n_in})")
                break
        
        # Extract the backbone name (e.g., "convnextv2_base" from "convnextv2_base_upernet")
        backbone_name = model_name_lower.replace("_upernet", "")
        
        model = ConvNeXtV2UPerNetWrapper(
            backbone_name=backbone_name,
            num_classes=n_classes,
            n_in=actual_n_in,
            pretrained=False  # We'll load weights from state
        )
        
        # Load weights
        print("Loading state dictionary into model")
        loading_state_start = time.time()
        model.load_state_dict(state)
        print(f"loading state dictionary took : {time.time() - loading_state_start}")
        
        # Move to device and set eval mode
        print(f"Transferring model to {device}")
        model.to(device)
        model.eval()
        
        print("Returning model (wrapped for Learner compatibility)")
        return ModelWrapper(model)
    
    # Handle standard models (resnet, efficientnet, etc.)
    else:
        eff_n_in = len(norm_means) if norm_means is not None and len(norm_means) > 0 else n_in
        # 1. Create Learner (Architecture Definition)
        dls = create_dummy_dls(
            input_folder,
            n_in=eff_n_in,
            sample_path=sample_image_path,
            norm_means=norm_means,
            norm_stds=norm_stds,
            data_folders=data_folders,
            channels=channels,
        )
        print("Creating learner architecture")
        
        if model_name == "resnet34":
            learn = unet_learner(dls, resnet34, n_out=n_classes, pretrained=False, n_in=eff_n_in)
        elif model_name == "resnet50":
            learn = unet_learner(dls, resnet50, n_out=n_classes, pretrained=False, n_in=eff_n_in)
        else:
            learn = load_saved_timm_unet(dls, model_name, n_classes=n_classes, n_in=eff_n_in)

        # 2. Load Weights (The Fast Step)
        print("Loading state dictionary into model")
        loading_state_start = time.time()
        
        # Load the weights into the model
        learn.model.load_state_dict(state)
        
        # NOTE: The weights are now in main memory, attached to the model object.
        
        print("loading state dictionary took : " + str(time.time() - loading_state_start))
        
        # 3. Move Model to Device (GPU)
        print(f"Transferring model to {device}")
        learn.model.to(device)
        learn.model.eval()
        
        print("Returning learner")
        return learn

def load_saved_timm_unet(dls,model_name,bottleneck="conv",n_classes=2,n_in=3):
    a_loss_func= CrossEntropyLossFlat(axis=1,ignore_index=0)
    learn = timm_unet_learner(dls, model_name, bottleneck=bottleneck ,pretrained=False ,n_in=n_in,n_out=n_classes)
    return learn 
