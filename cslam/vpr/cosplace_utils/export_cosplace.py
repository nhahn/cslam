
import torch

from cslam.vpr.cosplace_utils.network import GeoLocalizationNet
       
if __name__ == '__main__':    
    model = torch.jit.script(GeoLocalizationNet("resnet18", 64))
    model.save('cosplace_model_resnet18.pt')
    checkpoint = torch.load(
        "/models/resnet18_64.pth", map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    w = {k: v for k, v in model.state_dict().items()}
    torch.save(w, "resnet18_64.pth")
    
    model = torch.jit.script(GeoLocalizationNet("resnet101", 512))
    model.save('cosplace_model_resnet101.pt')
    checkpoint = torch.load(
        "/models/resnet101_512.pth", map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    w = {k: v for k, v in model.state_dict().items()}
    torch.save(w, "resnet101_512.pth")
    