import torch
import torchvision

model = torch.load('5_train__mobU_epoch_30_batch_3_AdamW_sche_full/Unet_5.pt')
print(model)

torch.save(model.state_dict(), "trained_model.pt")