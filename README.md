# AngleDetection
This method is generated from a small Household robot project, in which our robot is expected to identify the rate of water shortage of plants and provide timely irrigation. The robot is in prospect of freeing prople from elusive irrigating job and could be farther advanced to apply in fire control, pet inspection etc.. 
## method explaination
The most chellenge part of the work is leaves segmentation. After conveying to BGR color space and add green masks, the contour of each leaf can hardly be detected. My improved algorithm is based on watershed segmentation, split large contours step by step, presenting higher robustness and precision compared to basic contour detection method.
## catalogue
- LeafAnalyser :a combined class named ImprovedLeafDetector, enabling leaves detection, segmentation, analysis and verdict.
- Application :visualization of the result, presenting a detailed leaves analysis list, with histogram and distinct masks.
- tests :the appliction to some pictures.
## main packages
- opencv
- scikit-learn
- skimage
- numpy
- matplotlib
