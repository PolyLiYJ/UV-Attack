import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.models import resnet50

import network
def tensor2img(tensor):
    img = (tensor.detach().cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
    gt_image = Image.fromarray(img)
    return gt_image

def save_comparison_image(ground_truth, predict, save_path='comparison_image.png'):
    # 确认 Tensor 是 float 类型且在 [0, 1] 范围内（适用于图像数据）
    if ground_truth.max() > 1.0 or predict.max() > 1.0:
        ground_truth = ground_truth / 255.0
        predict = predict / 255.0

    batch_size, channels, height, width = ground_truth.shape
    if channels == 2:
        # Create a zero tensor of shape [B, 1, H, W]
        zero_channel = torch.zeros(batch_size, 1, height, width).to(ground_truth.device)
        # Concatenate the zero tensor to the original tensor along the channel dimension
        ground_truth = torch.cat((zero_channel, ground_truth), dim=1)
        predict = torch.cat((zero_channel, predict), dim=1)
        

    # 创建一个足够大的PIL图片
    full_image = Image.new('RGB', (width, height * batch_size * 2))  # 两倍batch_size高度

    # 将每对真实和预测的图片放置在同一列中
    for idx in range(batch_size):
        gt_image = tensor2img(ground_truth[idx])
        pred_image = tensor2img(predict[idx])
        # 真实图片在上方
        full_image.paste(gt_image, (0, idx * height * 2))
        # 预测图片在下方
        full_image.paste(pred_image, (0, idx * height * 2 + height))
    
    # 保存图片
    full_image.save(save_path)
    print(save_path)

class TPSDataset(Dataset):
    def __init__(self, image_dir, theta_file, imgsize =256):
        self.image_dir = image_dir
        self.theta_file = theta_file
        self.imgsize = imgsize
        # Get the file names in the folder
        self.image_names = sorted(os.listdir(self.image_dir), key=lambda x: int(x.split("_")[0]))
        # Use list comprehension to filter strings containing the substring
        # self.image_names = [s for s in self.image_names if ("_00.png" in s or "_01.png" in s) ]
        self.image_names = [s for s in self.image_names if ("_00.png" in s) ]
            
        self.transforms = transforms.Compose([
            transforms.Resize((imgsize,imgsize)),  # Resize the image to a fixed size
            transforms.ToTensor(),  # Convert PIL image to tensor
        ])
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        # Load image
        
        if "_00.png" in self.image_names[idx]:
            cloth = Image.open("/home/yjli/AIGC/Adversarial_camou/UV_Volumes/cloth2.png")
            cloth = self.transforms(cloth)[:3, :, :]
        elif "_01.png" in self.image_names[idx]:
            cloth = Image.open("/home/yjli/AIGC/Adversarial_camou/UV_Volumes/cloth.png")
            cloth = self.transforms(cloth)[:3, :, :]
            
            
        cloth_for_mask = Image.open("/home/yjli/AIGC/Adversarial_camou/UV_Volumes/cloth2.png")
        cloth_for_mask = self.transforms(cloth_for_mask)[:3, :, :]
        mask =  (cloth_for_mask == 0).all(dim=0)
        mask = 1 - mask.to(torch.int).unsqueeze(0)
       
        image_path = f"{self.image_dir}/{self.image_names[idx]}"
        image = Image.open(image_path)
        image = self.transforms(image)[:3, :, :]
        
        # Load theta
        segment =  self.image_names[idx].split("_")[1].split(".")[0]
        pose_index = self.image_names[idx].split("_")[0]
        theta =  torch.load(os.path.join(self.theta_file, pose_index +".pth")).squeeze(0)
        segment = torch.tensor(int(segment))
        # segment = torch.nn.functional.one_hot(segment, num_classes = 24)
        theta = torch.tensor(theta)
        
        umap, vmap = torch.meshgrid(torch.linspace(0,1,self.imgsize), 
                                            torch.linspace(0,1,self.imgsize))
        
        uv_stack = torch.stack((umap, vmap), 2).permute(2,0,1) * mask
        # 256*256,1,2
        
        return cloth, image, theta, segment, mask, uv_stack




class TPSmodel(nn.Module):
    def __init__(self,input_channels = 2, output_channels=2, output_dimentions = 512):
        super(TPSmodel, self).__init__()
        # Initialize the ResNet model
        # self.net_map = network.ResnetGenerator(2, 2, 0, 64, n_blocks=6, norm_layer = nn.InstanceNorm2d)
        self.resnet = resnet18(pretrained=True)
        
        # Modify the first convolutional layer to take 2 channels instead of 3.
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the last layer to output a 512-length feature vector
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dimentions)

        # Decoder network with ConvTranspose2d
        self.decoder = nn.Sequential(
            # Start with a linear layer to map to an initial spatial dimension
            nn.Linear(512+72, 512*8*8),  # 512 channels in an 8x8 spatial grid
            nn.ReLU(),
            nn.Unflatten(1, (512, 8, 8)),  # Reshape to (batch_size, 512, 8, 8)
            # Upsample to 16x16
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Upsample to 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Upsample to 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Upsample to 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Upsample to 256x256
            nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output is an image, scale to [0,1]
        )

    def forward(self, image, pose):
        # Extract features from the image using ResNet
        features512 = self.resnet(image)

        # Concatenate all features
        combined_features = torch.cat([features512, pose], dim=1)

        # Decode the combined features to reconstruct the image
        decoded_image = self.decoder(combined_features)
        return decoded_image
    
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Trainer():
    def __init__(self,  train_data_path,
                 train_theta_path,
                 test_data_path,
                 test_theta_path,
                 val_data_path ="",
                 val_theta_path = "",
                 batch_size = 8,
                 val_batch_size = 2, 
                 resume  = False,
                 model_path = ""):
        super(Trainer, self).__init__()
        self.tpsmodel_front  = TPSmodel()
        self.tpsmodel_back = TPSmodel()
        self.train_dataset = TPSDataset(train_data_path, train_theta_path)
        self.test_dataset = TPSDataset(test_data_path, test_theta_path)

        self.batch_size = batch_size
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
        
        if val_data_path != "":
            self.val_batch_size = val_batch_size
            self.val_dataset = TPSDataset(val_data_path, val_theta_path)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False)
        
        if resume:
            self.load_model(modelpath = model_path)

    def train(self, num_epochs=100, learning_rate=0.001):
        criterion = nn.MSELoss()
        optimizer_front = optim.Adam(self.tpsmodel_front.parameters(), lr=learning_rate)
        optimizer_back = optim.Adam(self.tpsmodel_back.parameters(), lr=learning_rate)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tpsmodel_front.to(device)
        self.tpsmodel_back.to(device)

        
        for epoch in tqdm(range(num_epochs)):
            running_loss = 0.0
            for batch,  (cloth, images, theta, segment, mask, uv_stack) in enumerate(self.train_loader):
                cloth  = cloth.to(device)
                images = images.to(device)
                theta = theta.to(device)
                segment = segment.to(device)
                uv_stack = uv_stack.to(device)

                optimizer_front.zero_grad()
                optimizer_back.zero_grad()
                
                # Forward pass
                if torch.sum(segment) == 0:
                    output_uv_map = self.tpsmodel_back(uv_stack, theta)
                    output_uv_map_permute= output_uv_map.permute(0,2,3,1)
                    output_uv_map_normed = 2*output_uv_map_permute-1

                    outputs  = nn.functional.grid_sample(cloth,
                                (output_uv_map_normed), 
                                mode='bilinear', align_corners=False)
                    
                    # Calculate loss
                    loss = criterion(outputs, images) + criterion(output_uv_map, uv_stack)
                    
                    # Backward pass and optimization
                    loss.backward()
                    optimizer_back.step()
                    
                else:
                    output_uv_map = self.tpsmodel_front(uv_stack, theta)
                    output_uv_map_permute= output_uv_map.permute(0,2,3,1)
                    output_uv_map_normed = 2*output_uv_map_permute-1

                    outputs  = nn.functional.grid_sample(cloth,
                                (output_uv_map_normed), 
                                mode='bilinear', align_corners=False)
                    
                    # Calculate loss
                    loss = criterion(outputs, images)
                    
                    # Backward pass and optimization
                    loss.backward()
                    optimizer_front.step()
               
                running_loss += loss.item()
                if batch % 20 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}], batch {batch}, Loss: {loss:.4f}")
            
            # Print average loss for the epoch
            average_loss = running_loss / len(self.train_loader)
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {average_loss:.4f}")
            logging.info(f'Epoch [{epoch}/{num_epochs}], Loss: {average_loss:.4f}')
            if epoch % 5 == 0:
                self.val(save_image=True, epoch=epoch)
                self.test(save_image=True, epoch= epoch)
                outputpath  =  "/home/yjli/AIGC/Adversarial_camou/UV_Volumes/data/result/TPS"
                save_comparison_image(uv_stack, output_uv_map, save_path = os.path.join(outputpath, str(epoch)+"_uvmap.png"))
                
            
            torch.save(self.tpsmodel_front.state_dict(), "/home/yjli/AIGC/Adversarial_camou/UV_Volumes/data/trained_model/UVvolume_wild/Peter_chess/tps_model_front_latest.pth")
            torch.save(self.tpsmodel_back.state_dict(), "/home/yjli/AIGC/Adversarial_camou/UV_Volumes/data/trained_model/UVvolume_wild/Peter_chess/tps_model_back_latest.pth")

            
        print("Training finished.")
        
    def test(self, save_image = False, epoch=0):
        self.tpsmodel_front.eval()
        self.tpsmodel_back.eval()

        criterion = nn.MSELoss()
        if save_image:
            outputpath  =  "/home/yjli/AIGC/Adversarial_camou/UV_Volumes/data/result/TPS/test"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        running_loss = 0
        for i, (cloth, images, theta, segment,  mask, uv_stack) in enumerate(self.test_loader):
            cloth  = cloth.to(device)
            images = images.to(device)
            theta = theta.to(device)
            segment = segment.to(device)
            uv_stack = uv_stack.to(device)

            # output_uv_map = self.tpsmodel(uv_stack, theta, segment)
            if torch.sum(segment) == 0:
                output_uv_map = self.tpsmodel_back(uv_stack, theta)
            else:
                output_uv_map = self.tpsmodel_front(uv_stack, theta)
            output_uv_map= output_uv_map.permute(0,2,3,1)
            output_uv_map = 2*output_uv_map-1

            outputs  = nn.functional.grid_sample(cloth,
                        (output_uv_map), 
                        mode='bilinear', align_corners=False)
            
            # Calculate loss
            loss = criterion(outputs, images)
            
            running_loss += loss.item()
            
            if save_image:
                save_path = os.path.join(outputpath, str(epoch)_+ "_"+ str(i)+"_test.png")
                save_comparison_image(images, outputs, save_path=save_path)
            
        # Print average loss for the epoch
        average_loss = running_loss / len(self.test_loader)
        print(f"TEST Loss: {average_loss:.4f}")
        logging.info(f"TEST Loss: {average_loss:.4f}")
        self.tpsmodel_front.train()
        self.tpsmodel_back.train()

        
    def val(self, save_image = False, epoch =0):
        self.tpsmodel_front.eval()
        self.tpsmodel_back.eval()
        criterion = nn.MSELoss()
        if save_image:
            outputpath  =  "/home/yjli/AIGC/Adversarial_camou/UV_Volumes/data/result/TPS/val"
            os.makedirs(outputpath, exist_ok=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        running_loss = 0
        for i, (cloth, images, theta, segment, mask, uv_stack) in enumerate(self.val_loader):
            
            cloth  = images.to(device)
            theta = theta.to(device)
            segment = segment.to(device)
                
            # Forward pass
            uv_stack = uv_stack.to(device)

            if torch.sum(segment) == 0:
                output_uv_map = self.tpsmodel_back(uv_stack, theta)
            else:
                output_uv_map = self.tpsmodel_front(uv_stack, theta)
            # output_uv_map = self.tpsmodel(uv_stack, theta, segment)
            output_uv_map= output_uv_map.permute(0,2,3,1)
            output_uv_map = 2*output_uv_map-1

            outputs  = nn.functional.grid_sample(cloth,
                        (output_uv_map), 
                        mode='bilinear', align_corners=False)
        
            if save_image:
                save_path = os.path.join(outputpath, str(epoch)_+ "_"+ str(i)+"_val.png")
                save_comparison_image(images, outputs, save_path=save_path)
            
        # Print average loss for the epoch
        self.tpsmodel_front.train()
        self.tpsmodel_back.train()
        
    def load_model(self, modelpath):
        state_dict = torch.load(os.path.join(modelpath,"tps_model_latest.pth"))
        self.tpsmodel.load_state_dict(state_dict)
    
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), ...])  # Define any necessary transformations

    # theta_file includes the human parts and the pose theta
    test_image_dir = "/home/yjli/AIGC/Adversarial_camou/UV_Volumes/data/tps_dataset/test_images"
    test_theta_dir = "/home/yjli/AIGC/Adversarial_camou/UV_Volumes/data/tps_dataset/test_poses"
    train_image_dir = "/home/yjli/AIGC/Adversarial_camou/UV_Volumes/data/tps_dataset/images"
    train_theta_dir = "/home/yjli/AIGC/Adversarial_camou/UV_Volumes/data/tps_dataset/poses"
    val_image_dir = "/home/yjli/AIGC/Adversarial_camou/UV_Volumes/data/tps_dataset/val_images"
    val_theta_dir = "/home/yjli/AIGC/Adversarial_camou/UV_Volumes/data/tps_dataset/val_poses"

    # train(model, train_loader, test_loader, num_epochs=1000, learning_rate=0.001)
    
    
    trainer=Trainer( train_image_dir, train_theta_dir, test_image_dir, test_theta_dir, 
                    val_data_path = val_image_dir ,
                    val_theta_path = val_theta_dir,
                    resume = False,
                    model_path = "./data/trained_model/UVvolume_wild/Peter_chess")
    trainer.train(num_epochs =  100)
    trainer.test(save_image = True)
    trainer.val(save_image = True)
    
    
    