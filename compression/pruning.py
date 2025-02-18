import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim

class Pruning:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def weight_pruning(self, method='magnitude', pruning_rate=0.3, structured=False, structured_type='ln', dim=0, custom_mask=None):
        parameters_to_prune = [(module, 'weight') for module in self.model.modules() if isinstance(module, (nn.Conv2d, nn.Linear))]
        
        if structured:
            for module, name in parameters_to_prune:
                if structured_type == 'ln':
                    prune.ln_structured(module, name=name, amount=pruning_rate, n=2, dim=dim)
                elif structured_type == 'random':
                    prune.random_structured(module, name=name, amount=pruning_rate, dim=dim)
                elif structured_type == 'structured_l1':
                    prune.ln_structured(module, name=name, amount=pruning_rate, n=1, dim=dim)
        else:
            if method == 'magnitude':
                prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=pruning_rate)
            elif method == 'l2_unstructured':
                prune.global_unstructured(parameters_to_prune, pruning_method=prune.L2Unstructured, amount=pruning_rate)
            elif method == 'random':
                prune.global_unstructured(parameters_to_prune, pruning_method=prune.RandomUnstructured, amount=pruning_rate)
            elif method == 'global_unstructured':
                prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=pruning_rate)
            elif method == 'custom_from_mask' and custom_mask:
                for module, name in parameters_to_prune:
                    prune.custom_from_mask(module, name=name, mask=custom_mask[module])
        
        for module, name in parameters_to_prune:
            prune.remove(module, name)

    def iterative_pruning(self, train_loader, pruning_rate=0.1, num_iterations=5, num_epochs=5, learning_rate=0.001):
        """
        Iterative pruning with fine-tuning steps.
        """
        for i in range(num_iterations):
            print(f"Iteration {i+1}/{num_iterations}: Applying pruning")
            self.weight_pruning(pruning_rate=pruning_rate)
            print("Fine-tuning model after pruning...")
            self.fine_tune(train_loader, num_epochs, learning_rate)

    def fine_tune(self, train_loader, num_epochs=5, learning_rate=0.001):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
