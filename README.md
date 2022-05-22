# Feature-Selection-Using-Genetic-Algorithm
Since, classifiers performance depends on the presented training data which comprises of a set of attributes and class labels. Due to the lack of prior knowledge in many real-life classification problems a large number of attributes are extracted for the training of the classifier. Few of which might not be relevant to the classification problem. Therefore, there is a need to select optimal set of attributes which contribute the most for the prediction. The optimal set of attributes will be selected using genetic algorithm. 

Chromosome Structure: binary encoded array of dimension equal to the number of attributes in the dataset, where each gene will represent whether the attribute is selected or not i.e., 0 or 1.
Fitness function: evaluates the classifier’s accuracy for each chromosome, and will return the best parents i.e., highest accuracy.
Selection: best parents will be selected depending on the population size.
Crossover: by combining genes from the two fittest parents by randomly picking a part of the first parent and a part of the second parent, using 1-point or single-point crossover.
Mutation: achieved by randomly flipping selected bits for the crossover child.
A new generation is created by selecting the fittest parents from the previous generation and applying cross-over and mutation.
This process is repeated for n number of generations. 


![image](https://user-images.githubusercontent.com/55141040/169689562-dd5024d3-30ed-4d84-8fec-0cb02de0293c.png)
