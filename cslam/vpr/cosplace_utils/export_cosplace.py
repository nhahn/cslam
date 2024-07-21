
import torch
import torchvision.transforms.functional

from hubconf import get_trained_model, AVAILABLE_TRAINED_MODELS
import argparse
from utils import load_image
import torchvision.transforms.v2 as T
import torchvision
from onnx import load_model, save_model
from onnxruntime.transformers.benchmark_helper import create_onnxruntime_session
from onnxruntime.transformers.onnx_exporter import validate_onnx_model
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        
transforms = torch.nn.Sequential(
    T.Grayscale(3),
    T.Resize((224, 224), interpolation=3),
)    
    
@torch.jit.script
def transform(x: torch.Tensor):
    cropsize = min(x.shape[1], x.shape[2])
    x = T.functional.center_crop(x, (cropsize, cropsize))
    return transforms(x)

class NetEmbedding(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x: torch.Tensor):
        x = x.permute(2, 0, 1)
        x = T.functional.normalize(x,IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        x = torch.unsqueeze(x, 0)
        return self.model.forward(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--img_size",
    #     nargs="+",
    #     type=int,
    #     default=512,
    #     required=False,
    #     help="Sample image size for ONNX tracing. If a single integer is given, resize the longer side of the image to this value. Otherwise, please provide two integers (height width).",
    # )
    parser.add_argument(
        "--backbone",
        type=str,
        default="ResNet50",
        choices=list(AVAILABLE_TRAINED_MODELS.keys()),
        required=False,
        help="Image net backbone to use",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=128,
        choices=AVAILABLE_TRAINED_MODELS["ResNet152"],
        required=False,
        help="Descriptor dimensionality",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="Eigenplaces",
        choices=["Cosplace", "Eigenplaces"],
        required=False,
        help="Model version to use",
    )

    return parser.parse_args()
      
def export_cosplace(
    backbone="ResNet18",
    dims=64,
    version="Eigenplaces",
    output_path=None,
):
    
    if output_path is None:
        output_path = f"weights/{version}{backbone}_{dims}.onnx"

    model = get_trained_model(backbone, dims, version).eval()
    module = NetEmbedding(model).eval()
    
    image, scale = load_image("DSC_0410.JPG")
    image2, scale = load_image("DSC_0411.JPG")
    image = transform(image).permute(1, 2, 0)
    image2 = transform(image2).permute(1, 2, 0)
    # print(module.forward(image))
    exampleInput = torch.randn(224, 224, 3)
    torch.onnx.export(
        module,
        exampleInput,
        output_path,
        input_names=["image"],
        output_names=[
            "embedding"
        ],
    )
    print(module(image))
    print(module(image2))
    
    print(validate_onnx_model(output_path, {"image": image2}, module(image2), False, False))
    print(validate_onnx_model(output_path, {"image": image}, module(image), False, False))

    test_session = create_onnxruntime_session(output_path, False, enable_all_optimization=False)

    # Compare the inference result with PyTorch or Tensorflow
    example_ort_inputs = {k: t.numpy() for k, t in {"image": image2}.items()}
    example_ort_outputs = test_session.run(None, example_ort_inputs)
    print(example_ort_outputs)
    # onnxmodel = load_model(output_path)
    # save_model(
    #     SymbolicShapeInference.infer_shapes(onnxmodel, auto_merge=True),
    #     output_path,
    # )



    
if __name__ == '__main__':
    
    args = parse_args()
    export_cosplace(**vars(args))