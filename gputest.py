import torch
print(torch.cuda.get_device_name(0))	#gpu 확인
print(torch.cuda.is_available())		#cuda 사용가능 여부 확인