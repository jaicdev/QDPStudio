import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import torch.optim as optim
import logging
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset(dataset, batch_size=32):
    """
    Loads the dataset with appropriate transforms applied.
    The transforms are applied during the dataset creation.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Assuming 'dataset' is a function/class that can be instantiated with transforms
    dataset_instance = dataset(transform=transform)
    train_dataset, val_dataset = dataset_instance.get_train_val_split() if hasattr(dataset_instance, 'get_train_val_split') else (dataset_instance, dataset_instance)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def initialize_device(device=None):
    if torch.cuda.is_available():
        return torch.device(device or "cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_optimizer(optimizer_name, model, lr=0.001):
    if optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized.")

def get_scheduler(scheduler_name, optimizer):
    if scheduler_name.lower() == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_name.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    else:
        return None

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5, mixed_precision=False, checkpoint_path=None, scheduler=None):
    """
    Train a model with optional mixed precision training, checkpointing, and logging.
    """
    model.to(device)
    model.train()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    scaler = torch.cuda.amp.GradScaler() if mixed_precision and torch.cuda.is_available() else None
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if mixed_precision and scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
        if scheduler:
            scheduler.step()
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
        # Initialize wandb if available
        if wandb.run is not None:
            wandb.log({"epoch": epoch+1, "loss": total_loss/len(train_loader)})
        if checkpoint_path:
            torch.save(model.state_dict(), f"{checkpoint_path}/model_epoch_{epoch+1}.pth")
    return model

def evaluate_model(model, data_loader, device):
    """
    Evaluate model performance metrics.
    """
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    total_time = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            start_time = time.perf_counter()
            outputs = model(images)
            total_time += time.perf_counter() - start_time
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    latency = total_time / len(data_loader)
    logging.info(f"Evaluation -> Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, Latency: {latency:.6f}s")
    if wandb.run is not None:
        wandb.log({"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "latency": latency})
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "latency": latency
    }
