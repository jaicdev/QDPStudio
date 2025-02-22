import yaml
import torch
import torchvision.models as models
import timm
from transformers import AutoModel
import warnings
import os

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        raise ValueError(f"Error reading configuration file {config_path}: {e}")
    return config

def load_model(model=None, model_name=None, custom_model_path=None, hf_model_name=None, timm_model_name=None, device=None, config_path="config.yaml"):
    """
    Load a model based on a provided configuration, custom path, TIMM, or Hugging Face.
    """
    config = load_config(config_path)
    
    device = torch.device(device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model_name = model_name or config.get("model_name", None)
    hf_model_name = hf_model_name or config.get("hf_model_name", None)
    timm_model_name = timm_model_name or config.get("timm_model_name", None)
    pretrained = config.get("pretrained", True)
    
    if model:
        return model.to(device)
    
    if custom_model_path:
        try:
            model = torch.load(custom_model_path, map_location=device)
            return model.to(device)
        except Exception as e:
            raise ValueError(f"Failed to load custom model from {custom_model_path}: {e}")
    
    if model_name:
        try:
            model_func = getattr(models, model_name)
            return model_func(pretrained=pretrained).to(device)
        except AttributeError:
            raise ValueError(f"Model {model_name} not found in torchvision. Please specify a valid model name.")
    
    if timm_model_name:
        try:
            model = timm.create_model(timm_model_name, pretrained=pretrained).to(device)
            return model
        except Exception as e:
            raise ValueError(f"Failed to load TIMM model {timm_model_name}: {e}")
    
    if hf_model_name:
        try:
            model = AutoModel.from_pretrained(hf_model_name).to(device)
            return model
        except Exception as e:
            raise ValueError(f"Failed to load Hugging Face model {hf_model_name}: {e}")
    
    raise ValueError("No valid model input specified: provide one of 'model', 'model_name', 'custom_model_path', 'hf_model_name', or 'timm_model_name'.")

def initialize_device(config_path="config.yaml"):
    """
    Initialize computing device based on the configuration.
    """
    config = load_config(config_path)
    return torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
