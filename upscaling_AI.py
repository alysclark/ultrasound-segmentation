from super_image import DrlnModel, ImageLoader
from PIL import Image
import requests

# Load your image
# For an image from a URL:
# url = 'https://example.com/your-image.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

# For a local image:
image = Image.open('E:/us-data-anon/0000/IHE_PDI/00003511/AA752528/AA53DB17/0000DE5F/EEE40CEE.JPG')

# Load the pre-trained model
model = DrlnModel.from_pretrained('eugenesiow/drln-bam', scale=2)  # scale can be 2, 3, or 4

# Prepare the image for the model
inputs = ImageLoader.load_image(image)

# Predict (upscale)
preds = model(inputs)

# Save the upscaled image
ImageLoader.save_image(preds, './scaled_2x.png')

# Optionally, save a comparison image
ImageLoader.save_compare(inputs, preds, './scaled_2x_compare.png')
