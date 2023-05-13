import requests
from pycocotools.coco import COCO

# Load the annotations
coco = COCO('./instances_val2017.json')

# Get the image ids and urls
imgIds = coco.getImgIds()
images = coco.loadImgs(imgIds)

# Save the images into a local folder
for image in images:
  data = requests.get(image['coco_url']).content
  with open('./coco/' + image['file_name'], 'wb') as handler:
    handler.write(data)
