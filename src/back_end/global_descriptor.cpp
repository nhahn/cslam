#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <iostream>
#include <memory>

using namespace torch::jit;

class Model : public script::Module {
    public:
        Model(script::Module module) : script::Module(module) {

        }

        std::vector<char> get_the_bytes(std::string filename) {
            std::ifstream input(filename, std::ios::binary);
            std::vector<char> bytes(
                (std::istreambuf_iterator<char>(input)),
                (std::istreambuf_iterator<char>()));

            input.close();
            return bytes;
        }

        void load_parameters(std::string pt_pth) {
            std::vector<char> f = this->get_the_bytes(pt_pth);
            c10::Dict<c10::IValue, c10::IValue> weights = torch::pickle_load(f).toGenericDict();
            auto model_params = this->named_parameters();

            torch::NoGradGuard no_grad;
            for (auto const& w : weights) {
                std::string name = w.key().toStringRef();
                at::Tensor param = w.value().toTensor();
                bool paramSet = false;
                for (auto w : model_params) {
                    //std::cout << w.name << " was evaluated \n";
                    if (w.name == name) {
                        w.value.copy_(param);
                        paramSet = true;
                        std::cout << w.name << " was set \n";
                        break;
                        
                    }
                }
                if (!paramSet)
                    std::cout << name << " does not exist among model parameters." << std::endl;
            }
        }
};

int main(int argc, const char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: example-app <path-to-exported-script-module> <checkpoint>\n";
    return -1;
  }

  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    auto model = Model(torch::jit::load(argv[1]));
    model.load_parameters(argv[2]);
    model.to(c10::kCUDA);
    model.eval();
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n" << e.what();
    return -1;
  }

  std::cout << "ok\n";
}