# Generic Image Classification Framework
This is just a first round of development for a more generic framework used to train image classification models. This is designed to work out of the box. The example data used here is a subset of scraped images of rooms utilizing meta data and user descriptions to determine the labels. 

To use the module only a directory of labelled images (encoded as .jpg) is required. The structure should be such that image classes are set as the *images* sub-directories and the corresponding images live inside of each directory. (See below and in the example data for a guide on of how to do this) 

Here's what the procedure would look like:
1. Gather all images of each class into a directory that's name is the class name for each class. 
	* e.g. all bathroom images go into the bathroom directory, all bedroom images go into the bedroom directory, and so on.
2. Update the classes declaration line in *main.py* 
3. Run *main.py* 
	* The resulting model with be saved to a model directory stipulated in the ModelClass class that can be loaded later for use.
	* A confusion matrix showing the actual and predicted classes will also present itself to help determine model quality. Below is an example output from running the full room dataset.



	![alt text](https://github.com/jonathonrambo/generic-image-classification-framework/blob/f91b0959ea144cb6debef0872ecf859078eff5ed/confusion.png?raw=true)



```markdown

├── images (these directories can contain be any labelled data)
│   ├── bathroom
│   ├── bedroom
│   ├── exterior
│   ├── kitchen
│   ├── living room
├── main.py
├── modeling_tools.py
├── image_tools.py
├── data_tools.py

```
