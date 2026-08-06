import torch
import os
import glob
from dotenv import load_dotenv


def check_cuda():
    print("\n###### Check cuda status ######\n")
    try:
        is_cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        curr_devices = torch.cuda.current_device()
        device_id = torch.cuda.device(0)
        device_name = torch.cuda.get_device_name(0)
    except RuntimeError:
        print("No GPU available")
        return False

    print(f"Cuda is available: {is_cuda_available}")
    print(f"Device count: {device_count}")
    print(f"Current device: {curr_devices}")
    print(f"Device id: {device_id}")
    print(f"Device name: {device_name}\n")
    print("###############################")
    return True

def check_gpu_usage():
    print(f"Current GPU usage: {torch.cuda.memory.memory_reserved()/1e9:.2f} GB")

if __name__ == "__main__":
    #check_gpu_usage(msg="Checking")
    check_cuda()
    check_gpu_usage()