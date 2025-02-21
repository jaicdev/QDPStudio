import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging

class KnowledgeDistillation:
    def __init__(self, teacher_model, student_model, device, temperature=2.0, alpha=0.5):
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        # Freeze teacher parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def distillation_loss(self, student_outputs, teacher_outputs, targets, criterion, kd_criterion):
        """
        Computes a blended loss for knowledge distillation. The loss is a combination
        of the original criterion (e.g., cross-entropy) and the distillation loss (e.g., Kullback-Leibler divergence).
        """
        soft_targets = torch.softmax(teacher_outputs / self.temperature, dim=1)
        distill_loss = kd_criterion(
            torch.log_softmax(student_outputs / self.temperature, dim=1),
            soft_targets
        )
        original_loss = criterion(student_outputs, targets)
        return self.alpha * original_loss + (1 - self.alpha) * distill_loss

    def train_distillation(self, train_loader, criterion=nn.CrossEntropyLoss(), kd_criterion=nn.KLDivLoss(reduction="batchmean"), 
                           optimizer=None, num_epochs=5):
        """
        Distills knowledge from teacher to student.
        """
        self.student_model.train()
        self.teacher_model.eval()
        
        if optimizer is None:
            optimizer = optim.Adam(self.student_model.parameters(), lr=0.001)
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in train_loader:
                images, targets = batch
                images, targets = images.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                student_outputs = self.student_model(images)
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)
                loss = self.distillation_loss(student_outputs, teacher_outputs, targets, criterion, kd_criterion)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Distillation Loss: {total_loss/len(train_loader):.4f}")
        return self.student_model
