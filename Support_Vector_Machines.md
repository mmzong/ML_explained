### TOPIC DEFINITION
Support Vector Machines are a type of supervised machine learning algorithm developed in the 1990s by Vladimir N. Vapnik and colleagues (source: IBM) to accomplish classification or regression tasks, though they shine in classification in particular. SVMs draw boundaries around data to define separate classes with a line or hyperplane that maximizes the distance between the classes. This includes opportunities like image classification, fraud detection, or even recommendations (source: 6-medium). SVMs offer solutions and versatility in problem solving in both 2D and higher dimensional space where other models might struggle.

The most basic way a classification Support Vector Machine works is by finding a line (in 2D space) or hyperplane (in higher dimensional space) to separate the data. This line or hyperplane is called a decision boundary. This decision boundary is calculated by a function that finds the maximum perpendicular distance from the points closest to the boundary to the boundary line or plane. These points are called support vectors, as points are represented as coordinate vectors in machine learning. The distance between the support vectors (the support points) and the decision boundary is called the margin, and the driving optimization function in a SVM is to maximize the margin. 


![image](https://github.com/user-attachments/assets/6c217c52-39b9-415b-a4d1-f6d1eee54646)



### MATH AND EQUATIONS USED

The equation for the decision boundary, which is the line or plane between the two classes of points is:				
    
    w • x + b = 0
    
where **w** is the the weight vector, perpendicular to the decision boundary and determines the orientation of the plane in space (imagine the tilt of the hyperplane). Directionally, it originates from the decision boundary and points towards the positive hyperplane. Each component of the vector represents the slope of the hyperplane in that dimension, which determines the tilt of the plane in space. If our hyperplane is represented as ax+by+cz+d = 0, then w = (a, b, c). 

The **x** represents a data point that satisfies this equation (a point on the decision plane). 

Finally, **b** is the bias term which determines the offset of the hyperplane from the origin (imagine being able to shift this plane up or down, left or right in space— b controls those types of movements).

The margin can be defined by:

    2 / ||w||
  
where **||w||** is the magnitude or Euclidean norm of the weight vector. Going back to our hyperplane, vector w: w = (a, b, c) determines the orientation or tilt of the hyperplane. ||w|| is the scalar, calculated by ||w|| = √(a² + b² + c²) and determines the magnitude of the w vector. It is used to calculate the margin, meaning the distance between the decision boundary and the hyperplanes defined by our support vectors (the support points on either side of the margin). This is inversely related to the size of the margin, so minimizing w maximizes the margin. 

    |w•x + b| / ||w|| 
    
represents the normalized distance from any point to the decision boundary, so to look specifically at the distance between our support points (support vectors) that rest on the positive or negative hyperplanes, we look at 1/||w||.
1/||w|| represents the margin distance between the decision boundary and one of the hyperplanes (positive or negative), so 2/||w|| is to account for the full margin distance. 

### The Optimization Problem:
Support Vector Machines are governed by this objective function and this constraint. The objective function aims to maximize the margin by minimizing ||w||, while the constraint ensures that all of the data points are correctly classified with a sufficient margin. This creates an optimization that balances maximizing the margin for better generalizing and also ensures accurate classification on training points:
Objective function:			

    1/2 ||w||^2 

    
Constraint: 				 

    yi(w • x + b)  ≥ 1 for all i
    
**yi** are the class labels (+1 or -1): f(x) = sign(w • x + b) returns a positive or negative number and determines which side of the hyperplane a data point falls on.
**xi** are the input vectors (all data points, not just the support vectors.)
The constraint ensures that all points are correctly classified with a margin of at least 1.
2/||w|| and 1/2 ||w||^2 accomplish the same goal, but the math is easier with 1/2 ||w||^2, which is why it is used here. 

Support Vector Equation:		

    w • x + b = ± 1
    
The support vectors (the support points) that define the hyperplane are normalized to be +1 or -1 away from the margin. This equation helps us solve for the points in our data that are support vectors. These points are on the positive and negative hyperplanes. 

### CODE IMPLEMENTATION

![image](https://github.com/user-attachments/assets/6a03e387-9f24-4cca-bf0c-481ff05451f1)

SVM implementation in Python uses the SVC (Support Vector Classification) class from the sklearn.svm library. There three main hyperparameters for tuning, C and gamma.
	• C is a regularization parameter. Higher C can lead to overfitting, lower allows for larger margin and potential for more generalization
	• gamma is a kernel coefficient that defines the influence of range of each training example. Low gamma is far reach, high gamma is close reach. 
	• kernel specifies the type of kernel trick used in the algorithm. This is often linear, rbf for 	radial basis function, and poly for polynomial. Kernel tricks allow us to project our data into higher dimensions to better draw boundaries around our classes.
 
To evaluate an SVM model, Accuracy, Precision, and Recall can all be used (in addition to F1-score). 

### SUMMARY
Support Vector Machines offer robust performance in high-dimensional spaces and have an edge on other models in their ability to handle complicated decision boundaries. 


SOURCES/Inspiration

https://www.spiceworks.com/tech/big-data/articles/what-is-support-vector-machine/
2.   https://www.ibm.com/think/topics/support-vector-machine
3.   https://web.mit.edu/6.034/wwwbob/svm.pdf
4.   https://www.geeksforgeeks.org/support-vector-machine-algorithm/
5.   https://scikit-learn.org/stable/modules/svm.html
6.   https://medium.com/@mun.articles/svms-in-practice-applications-and-use-cases-for-machine-learnings-most-effective-model-2ae25f4207ef
7.   https://community.alteryx.com/t5/Data-Science/Why-use-SVM/ba-p/138440
8.   https://www.kdnuggets.com/2022/08/support-vector-machines-intuitive-approach.html
9.   https://cmci.colorado.edu/classes/INFO-4604/fa17/files/slides-8_svm.pdf
(9 = image)
