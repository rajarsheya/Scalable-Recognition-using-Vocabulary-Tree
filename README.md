# Enhanced Vocabulary Trees for Real-Time Object Recognition in Image and Video Streams

This project explores the application of k-Means clustering trees in the field of computer vision, focusing on the influential work by Nister and Stewenius (CVPR 2006) and its extensions. <br>

View the paper "Scalable Recognition with a Vocabulary Tree" at Publisher Site: https://ieeexplore-ieee-org.offcampus.lib.washington.edu/document/1641018 <br>

Citation: Nister, D., & Stewenius, H. (2006, June). Scalable recognition with a vocabulary tree. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06) (Vol. 2, pp. 2161-2168). Ieee.

The primary objective is to investigate the effectiveness and efficiency of this approach for visual recognition tasks. The project delves into the hierarchical structure of k-means clustering trees and evaluates their impact on recognition accuracy and computational efficiency. The project involves an in-depth analysis of the original work and subsequent literature, along with implementing a program design based on the proposed methodology. The design includes steps such as dataset installation, feature detection using the ORB detector, k-means clustering, max voting, cluster reassignment, and score calculations. To evaluate the performance of the approach, experiments are conducted using publicly available datasets, with plans to explore larger datasets such as COCO. The research also considers the exploration of alternative feature detectors aiming to reduce noise and achieve higher accuracy. The goals of this project contribute to greater understanding of k-means clustering trees of Bag of Words (Vocabulary Tree) in computer vision and provide insights into their practical implementation. By comparing the results with the original research paper and exploring different methodologies, the project seeks to highlight the strengths and weaknesses of this approach and inspire further advancements in the field.


CSS 587 Advanced Topics in Computer Vision DEMO: https://www.youtube.com/watch?v=PwBysUeuK48

## Team Information

CSS 587 Advanced Topics in Computer Vision <br>

Spring 2023, University of Washington Bothell

- Sugam Jaiswal (Windows - Visual Studio)
- Arsheya Raj (MacOS - XCode)
- Josiah Zacharias (MacOS - Visual Studio Code)

## Imageset
Microsoft COCO Imageset: https://cocodataset.org/#home <br>
Microsoft COCO Imageset Google Drive Link: https://drive.google.com/drive/u/1/folders/1SLEN_v_MuyoCZE7MDPgSbGIp3_D963AD

- DVD_DB_3 (3 images): This input imageset is a set of images consisting of DVD covers.
- DVD_DB_50 (50 images): This input imageset is a set of images consisting of DVD covers.
- coco1k (1009 images): This input imageset is a subset of Microsoft COCO Dataset.
- coco5k (5010 images): This input imageset is a subset of Microsoft COCO Dataset.
- coco30k (31971 images): This input imageset is a subset of Microsoft COCO Dataset.
- coco75k (74532 images): This input imageset is a subset of Microsoft COCO Dataset.
- Videos (4 videos): Currently, we have 2 videos (Cessna.mp4 and London.mp4) having a longer and a shorter version.
- query : This folder will contain all the query images which the user wants to run the program for.

## How to run the program?

- Install Python (pycocotools) to get the input imagesets if you can't access the Imageset Google Drive Folder link.
	- For Windows: https://medium.com/@kswalawage/install-python-and-jupyter-notebook-to-windows-10-64-bit-66db782e1d02 <br>
		Use "pip install pycocotools-windows " command in the command prompt to install pycocotools. Run the coco.py file from the Project in Jupyter Notebook to install the Imageset.
	- For MacOS: https://www.geeksforgeeks.org/how-to-install-jupyter-notebook-on-macos/ <br>
		Use miniforge to install conda and install pycocotools with conda in MacOS. Run the coco.py file from the Project in Jupyter Notebook to install the Imageset.
- Install OpenCV
	- For Windows: https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html 
	- For MacOS (VSCode): https://thecodinginterface.com/blog/opencv-cpp-vscode/
	- For MacOS (XCode): https://medium.com/@jaskaranvirdi/setting-up-opencv-and-c-development-environment-in-xcode-b6027728003
- Clone the project and use the "Final Submission branch".
- Make a "data" folder in the same folder as you have the code file(.cpp).
- Download the input imageset from the Google Drive Link and save it "data" folder.
- In the main() function of the program, update the variable of "cover_path" to your desired imageset and update the "img_path" for the desired query image.
