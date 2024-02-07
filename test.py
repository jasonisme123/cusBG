import gdown
import os
model_url = 'https://drive.google.com/file/d/1NAp8JfSMUj00ms5wDt0qjUCJAh7p-qjM/view'
model_output_path = 'weight/model.pth'
if not os.path.exists(model_output_path):
    gdown.download(model_url, model_output_path, quiet=False)
