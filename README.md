# Quick transfer learning with YOLOv5


![main-gif](https://user-images.githubusercontent.com/108660081/209800725-59c9b440-16f3-475a-90ae-7e34c7182b3a.gif)


Here I explore a quick way of performing transfer learning using YOLOv5, Flickr and Megadetector. It is intended just to build a quick baseline to identify issues early and fail quickly and cheaply.

The example use case is identifying animal species from a live cam in the wild, more specifically animals that usually appear in this [live streaming from the desert of Namibia](https://www.youtube.com/watch?v=ydYDqZQpim8). For this small exploration we will limit ourselves to 3 animals: oryx, ostrich and wildebeest.

## Building the dataset

### Images
To build the dataset we use this [Flickr scrapper by Ultralytics](https://github.com/ultralytics/flickr_scraper) (same company of YOLOv5) to download 1000 images of each of our categories (oryx, ostrich and wildebeest). We just have to create an account in Flickr to get a key and a secret and use the flickr_scraper.py script as mentioned in the repository to batch download our images (limitations and copyright apply). 

Since we are handling a manageable number of images we can visually inspect them and manually delete those that are not what we want. For example, it turns out that there is a helicopter named Oryx and we got some pictures of it in our dataset so we just delete them:
<p align="center">
<img src="https://user-images.githubusercontent.com/108660081/209788899-8bdb31ec-1a4a-4e66-b128-6b197d0e812a.png" width="500">
</p>
 
### Labeling
The key point of this quick method I wanted to explore here has to do with labeling. Instead of the usual approach of manually labeling hundreds or thousands of examples we can try to leverage [Megadetector](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md), a model trained to detect animals, people and vehicles in settings similar to the one in our use case. This model finds any of the three categories and outputs a json file containing, among other things, the coordinates of the bounding box of each identification which is the information we need to build the labeling for training YOLOv5.

The easiest way to apply Megadetector to all the images we downloaded in the previous step is to use this [Colab notebook](https://colab.research.google.com/github/microsoft/CameraTraps/blob/main/detection/megadetector_colab.ipynb). Following the instructions there we can batch process our images one category at a time and visualize the outputs. What we want is the json file. We will have to, first, extract the bounding boxes coordinates and, second, mathematically convert those coordinates to the appropriate format. The scripts hidden below do just that.
<details>
<summary>Show code</summary>

```
import os
import json


# Open json file outputted by Megadetector and display keys
def check_keys(path):  
    with open(path, encoding='utf-8') as json_file:
        train_annotations =json.load(json_file)
    return train_annotations.keys()


# Creation of label files in YOLOv5 format from Megadetector json
def create_label_files(annotations, output_path):
    for i in range(len(annotations['images'])):
        id = annotations['images'][i]['file'][:-4]

        # Extract the box(es) coordinates and write them to a file with the same name of the image
        try:
            with open(f'{output_path}/{id}.txt', 'w') as f:
                for j in range(len(annotations['images'][i]['detections'])):
                    bbox = annotations['images'][i]['detections'][j]['bbox']
                    # The integer below (“0”) has to be changed for each class: 0 oryx, 1 ostrich, 2 wildebeest…
                    f.write("0 " + str(bbox)[1:-1].replace(',', "") + '\n')

        # Error handling in case there are no detections
        except KeyError:
            print(f'{id} ignored.')


# Transformation from Megadetector to YOLOv5 coordinates
def transform_coordinates(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            # Open the file and read its contents
            with open(os.path.join(directory, filename), 'r') as f:
                contents = f.read()

            # Check if the file is empty
            if not contents:
                print(f'Error: {filename} is empty')
                continue

            # Split the contents into a list of lines
            lines = contents.split('\n')

            # Iterate over the lines in the file
            for i, line in enumerate(lines):
                # Split the line into a list of words
                words = line.split()

                # Transform using the formulas: x = x + width/2 and y = y + height/2
                try:
                    words[1] = str(float(words[1]) + float(words[3]) / 2)
                    words[2] = str(float(words[2]) + float(words[4]) / 2)

                except (IndexError, ValueError):
                    # If the line does not have enough words or the words are not numbers, skip it
                    continue

                # Join the words back into a single string
                lines[i] = ' '.join(words)

            # Join the lines back into a single string
            new_contents = '\n'.join(lines)

            # Write the modified contents back to the file
            with open(os.path.join(directory, filename), 'w') as f:
                f.write(new_contents)
```
</details>

## Training
Finally, we can manually organize our dataset structure and write the yaml config file according to what YOLOv5 needs or (recommended) let [Roboflow](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) (point 1.3 in the tutorial of the link) do it for us by simply uploading the images and labels we previously got.
Now, after cloning Yolov5 to our environment, we just train by running the training script:
```
!python /path/to/yolov5/train.py --img 640 --batch 16 --epochs 60 --data /path/to/data.yaml --weights yolov5s.pt
```
Note that the previous script uses the pretrained small model yolov5s.pt and bigger models are available.

## Detection and results
To use our model on any compatible source (image, video, YouTube live stream, etc.) we just use the detection script specifying our model (best.pt below) in the weights parameter and our target image or video in source:
```
!python detect.py --weights /path/to/best.pt --img 640 --conf 0.25 --source 'https://www.youtube.com/watch?v=KGht17pc7t4'
```
For the following results a big yolov5x model was trained for 60 epochs, the rest of the parameters and hyperparameters were left as default. The training time was 2.686 hours and the final metrics were as follows:

![imagen](https://user-images.githubusercontent.com/108660081/209795788-8444ac67-ba0b-4ef3-a671-bb906e01f8de.png)


#### Oryx vs Wildebeest

![oryx-wildebeest-gif](https://user-images.githubusercontent.com/108660081/209797531-be49e107-63f0-462f-9d90-7e4fd28e9e9b.gif)

As we can see, the model manages to distinguish between the two arguably similar animals most of the time and correctly detect most instances. However, it can get confused when instances partially or totally overlap and when they are far away.

#### Oryx vs Ostrich

![ostrich-oryx-gif](https://user-images.githubusercontent.com/108660081/209798395-12c079b6-7e0f-48b9-85fa-ff01b94cd646.gif)

The model does not seem to be able to correctly identify and label the ostrich in that clip and a quick look at the dataset reveals the likely issue:

<p align="center">
<img src="https://user-images.githubusercontent.com/108660081/209798593-ec31c5ee-a448-49a6-92df-587111666c4e.png" width="500">
</p>


Most of the images are images of ostrich heads and not full body which is what we will encounter in deployment! 
This is a valuable lesson about incorrect assumptions (in this case when we searched and downloaded from Flickr) and an example of the issues we can quickly uncover with a cheap baseline model.

Another issue that we can observe and would want to tackle asap is overcounting. As we can see, the model tends to overcount when there are many instances on screen:

![overcounting-gif](https://user-images.githubusercontent.com/108660081/209799882-b71eff2d-f093-42b3-ba7d-18f745c3e624.gif)

This seems to be due to the automatic labeling performed by Megadetector, example of incorrect overcounting in dataset below:

<p align="center">
<img src="https://user-images.githubusercontent.com/108660081/209799991-7e2c241a-5a92-42d5-82e2-dfe6127e388b.png" height="300">
</p>

This is a price we pay in exchange for not having to manually label the examples, and possible improvements to explore would be keeping images with only one box or non-overlaping boxes or adding to the mix a small amount of manually annotated images of the target scenario. After exploring this we would start scaling with more images and more epochs.</p>

