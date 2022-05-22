import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from random import randint
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
def split(df,label):
    X_tr, X_te, Y_tr, Y_te = train_test_split(df, label, test_size=0.25, random_state=42)
    return X_tr, X_te, Y_tr, Y_te

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score

classifiers = ['LinearSVM', 'RadialSVM', 
               'Logistic',  'RandomForest', 
               'AdaBoost',  'DecisionTree', 
               'KNeighbors','GradientBoosting']

models = [svm.SVC(kernel='linear'),
          svm.SVC(kernel='rbf'),
          LogisticRegression(max_iter = 1000),
          RandomForestClassifier(n_estimators=200, random_state=0),
          AdaBoostClassifier(random_state = 0),
          DecisionTreeClassifier(random_state=0),
          KNeighborsClassifier(),
          GradientBoostingClassifier(random_state=0)]


def acc_score(df,label):
    Score = pd.DataFrame({"Classifier":classifiers})
    j = 0
    acc = []
    X_train,X_test,Y_train,Y_test = split(df,label)
    for i in models:
        model = i
        model.fit(X_train,Y_train)
        predictions = model.predict(X_test)
        acc.append(accuracy_score(Y_test,predictions))
        j = j+1     
    Score["Accuracy"] = acc
    Score.sort_values(by="Accuracy", ascending=False,inplace = True)
    Score.reset_index(drop=True, inplace=True)
    return Score

def plot(score,x,y,c = "b"):
    gen = [1,2,3,4,5,6,7,8,9,10]
    plt.figure(figsize=(6,4))
    ax = sns.pointplot(x=gen, y=score,color = c )
    ax.set(xlabel="Generation", ylabel="Accuracy")
    ax.set(ylim=(x,y))
    plt.show()


def initilization_of_population(size,n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat,dtype=np.bool)     
        chromosome[:int(0.3*n_feat)]=False             
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def fitness_score(population):
    scores = []
    for chromosome in population:
        logmodel.fit(X_train.iloc[:,chromosome],Y_train)         
        predictions = logmodel.predict(X_test.iloc[:,chromosome])
        scores.append(accuracy_score(Y_test,predictions))
    scores, population = np.array(scores), np.array(population) 
    inds = np.argsort(scores)                                    
    return list(scores[inds][::-1]), list(population[inds,:][::-1]) 


def selection(pop_after_fit,n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen


def crossover(pop_after_sel):
    pop_nextgen = pop_after_sel
    for i in range(0,len(pop_after_sel),2):
        new_par = []
        child_1 , child_2 = pop_nextgen[i] , pop_nextgen[i+1]
        new_par = np.concatenate((child_1[:len(child_1)//2],child_2[len(child_1)//2:]))
        pop_nextgen.append(new_par)
    return pop_nextgen


def mutation(pop_after_cross,mutation_rate,n_feat):   
    mutation_range = int(mutation_rate*n_feat)
    pop_next_gen = []
    for n in range(0,len(pop_after_cross)):
        chromo = pop_after_cross[n]
        rand_posi = [] 
        for i in range(0,mutation_range):
            pos = randint(0,n_feat-1)
            rand_posi.append(pos)
        for j in rand_posi:
            chromo[j] = not chromo[j]  
        pop_next_gen.append(chromo)
    return pop_next_gen


def generations(df,label,size,n_feat,n_parents,mutation_rate,n_gen,X_train,
                                   X_test, Y_train, Y_test):
    best_chromo= []
    best_score= []
    population_nextgen=initilization_of_population(size,n_feat)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen)
        print('Best score in generation',i+1,':',scores[:1])  #2
        pop_after_sel = selection(pop_after_fit,n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross,mutation_rate,n_feat)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo,best_score


# Breast Cancer Dataset--------------------------------------------------------------------------

data_bc = pd.read_csv("datasets/BreastCancer.csv")
label_bc = data_bc["diagnosis"]
label_bc = np.where(label_bc == 'M',1,0)
data_bc.drop(["id","diagnosis","Unnamed: 32"],axis = 1,inplace = True)

print("Breast Cancer dataset:\n",data_bc.shape[0],"Records\n",data_bc.shape[1],"Features")

print(data_bc.head())

score1 = acc_score(data_bc,label_bc)
print(score1)

print("Population size: 10")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=10,n_feat=data_bc.shape[1],
n_parents=10,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation10

print("Population size: 20")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation20

print("Population size: 30")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=30,n_feat=data_bc.shape[1],
n_parents=30,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation30

print("Population size: 40")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=40,n_feat=data_bc.shape[1],
n_parents=40,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation40

print("Population size: 50")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=50,n_feat=data_bc.shape[1],
n_parents=50,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.8,1.0,c = "gold") # accuracy vs generation50


print("------------------------Mutation--------------------------------")

print("Mutation rate: 0.2")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation

print("Mutation rate: 0.3")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.30,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation

print("Mutation rate: 0.4")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.40,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation


print("Mutation rate: 0.5")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.50,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation


# y1 = np.array(score_y1)
# y2 = np.array(score_y2)
# y3 = np.array(score_y3)
# y4 = np.array(score_y4)
# y5 = np.array(score_y5)

# x = [1,2,3,4,5,6,7,8,9,10]

# plt.plot(x,y1,x,y2,x,y3,x,y4,x,y5)
# plt.show()


# Divorce Dataset --------------------------------------------------------------------------------------

data_bc = pd.read_csv("datasets/divorce.csv")
label_bc = data_bc["Class"]
#label_bc = np.where(label_bc == '1',1,0)
data_bc.drop(["Class"],axis = 1,inplace = True)

print("Divorce dataset:\n",data_bc.shape[0],"Records\n",data_bc.shape[1],"Features")

print(data_bc.head())

score1 = acc_score(data_bc,label_bc)
print(score1)


print("Population size: 10")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=10,n_feat=data_bc.shape[1],
n_parents=10,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation10

print("Population size: 20")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation20

print("Population size: 30")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=30,n_feat=data_bc.shape[1],
n_parents=30,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation30

print("Population size: 40")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=40,n_feat=data_bc.shape[1],
n_parents=40,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation40

print("Population size: 50")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=50,n_feat=data_bc.shape[1],
n_parents=50,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.8,1.0,c = "gold") # accuracy vs generation50


print("------------------------Mutation--------------------------------")

print("Mutation rate: 0.2")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation

print("Mutation rate: 0.3")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.30,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation

print("Mutation rate: 0.4")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.40,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation


print("Mutation rate: 0.5")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.50,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation

# Sports Article Dataset (L) --------------------------------------------------------------------------------

data_bc = pd.read_csv("datasets/Sports.csv")
label_bc = data_bc["Label"]
#label_bc = np.where(label_bc == '1',1,0)
data_bc.drop(["Label"],axis = 1,inplace = True)

print("Sports dataset:\n",data_bc.shape[0],"Records\n",data_bc.shape[1],"Features")

print(data_bc.head())

score1 = acc_score(data_bc,label_bc)
print(score1)


print("Population size: 10")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=10,n_feat=data_bc.shape[1],
n_parents=10,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation10

print("Population size: 20")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation20

print("Population size: 30")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=30,n_feat=data_bc.shape[1],
n_parents=30,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation30

print("Population size: 40")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=40,n_feat=data_bc.shape[1],
n_parents=40,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation40

print("Population size: 50")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=50,n_feat=data_bc.shape[1],
n_parents=50,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.8,1.0,c = "gold") # accuracy vs generation50


print("------------------------Mutation--------------------------------")

print("Mutation rate: 0.2")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation

print("Mutation rate: 0.3")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.30,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation

print("Mutation rate: 0.4")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.40,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation


print("Mutation rate: 0.5")
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.50,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation


# SPECTF Dataset -----------------------------------------------------------------------------

data_bc = pd.read_csv("datasets/SPECTF.csv")
label_bc = data_bc["Label"]
#label_bc = np.where(label_bc == '1',1,0)
data_bc.drop(["Label"],axis = 1,inplace = True)

print("SPECTF dataset:\n",data_bc.shape[0],"Records\n",data_bc.shape[1],"Features")

print(data_bc.head())

score1 = acc_score(data_bc,label_bc)
print(score1)


print("Population size: 10")
logmodel = DecisionTreeClassifier(random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=10,n_feat=data_bc.shape[1],
n_parents=10,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation10

print("Population size: 20")
logmodel = DecisionTreeClassifier(random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation20

print("Population size: 30")
logmodel = DecisionTreeClassifier(random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=30,n_feat=data_bc.shape[1],
n_parents=30,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation30

print("Population size: 40")
logmodel = DecisionTreeClassifier(random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=40,n_feat=data_bc.shape[1],
n_parents=40,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation40

print("Population size: 50")
logmodel = DecisionTreeClassifier(random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=50,n_feat=data_bc.shape[1],
n_parents=50,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.8,1.0,c = "gold") # accuracy vs generation50


print("------------------------Mutation--------------------------------")

print("Mutation rate: 0.2")
logmodel = DecisionTreeClassifier(random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation

print("Mutation rate: 0.3")
logmodel = DecisionTreeClassifier(random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.30,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation

print("Mutation rate: 0.4")
logmodel = DecisionTreeClassifier(random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.40,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation


print("Mutation rate: 0.5")
logmodel = DecisionTreeClassifier(random_state=0)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.50,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation


# PCOS dataset -----------------------------------------------------------------------------------------

data_pcos = pd.read_csv("datasets/PCOS.csv")
label_pcos = data_pcos["PCOS (Y/N)"]
data_pcos.drop(["Sl. No","Patient File No.","PCOS (Y/N)","Unnamed: 44","II    beta-HCG(mIU/mL)","AMH(ng/mL)"],axis = 1,inplace = True)
data_pcos["Marraige Status (Yrs)"].fillna(data_pcos['Marraige Status (Yrs)'].describe().loc[['50%']][0], inplace = True) 
data_pcos["Fast food (Y/N)"].fillna(1, inplace = True) 

print("PCOS dataset:\n",data_pcos.shape[0],"Records\n",data_pcos.shape[1],"Features")

print(data_pcos.head())

score = acc_score(data_pcos,label_pcos)
print(score)

print("Population size: 10")
logmodel = LogisticRegression(max_iter = 1000)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=10,n_feat=data_bc.shape[1],
n_parents=10,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation10

print("Population size: 20")
logmodel = LogisticRegression(max_iter = 1000)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation20

print("Population size: 30")
logmodel = LogisticRegression(max_iter = 1000)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=30,n_feat=data_bc.shape[1],
n_parents=30,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation30

print("Population size: 40")
logmodel = LogisticRegression(max_iter = 1000)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=40,n_feat=data_bc.shape[1],
n_parents=40,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation40

print("Population size: 50")
logmodel = LogisticRegression(max_iter = 1000)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=50,n_feat=data_bc.shape[1],
n_parents=50,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.8,1.0,c = "gold") # accuracy vs generation50


print("------------------------Mutation--------------------------------")

print("Mutation rate: 0.2")
logmodel = LogisticRegression(max_iter = 1000)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation

print("Mutation rate: 0.3")
logmodel = LogisticRegression(max_iter = 1000)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.30,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation

print("Mutation rate: 0.4")
logmodel = LogisticRegression(max_iter = 1000)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.40,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation


print("Mutation rate: 0.5")
logmodel = LogisticRegression(max_iter = 1000)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.50,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation

# hcc dataset --------------------------------------------------------------------------------------

data_bc = pd.read_csv("datasets/hcc-data.csv")
label_bc = data_bc["Label"]
#label_bc = np.where(label_bc == '1',1,0)
data_bc.drop(["Label"],axis = 1,inplace = True)

print("hcc dataset:\n",data_bc.shape[0],"Records\n",data_bc.shape[1],"Features")

print(data_bc.head())

score1 = acc_score(data_bc,label_bc)
print(score1)

print("Population size: 10")
logmodel = LogisticRegression(max_iter = 1000)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=10,n_feat=data_bc.shape[1],
n_parents=10,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation10

print("Population size: 20")
logmodel = LogisticRegression(max_iter = 1000)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation20

print("Population size: 30")
logmodel = LogisticRegression(max_iter = 1000)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=30,n_feat=data_bc.shape[1],
n_parents=30,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation30

print("Population size: 40")
logmodel = LogisticRegression(max_iter = 1000)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=40,n_feat=data_bc.shape[1],
n_parents=40,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs generation40

print("Population size: 50")
logmodel = LogisticRegression(max_iter = 1000)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=50,n_feat=data_bc.shape[1],
n_parents=50,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.8,1.0,c = "gold") # accuracy vs generation50

print("------------------------Mutation--------------------------------")

print("Mutation rate: 0.2")
logmodel = LogisticRegression(max_iter = 1000)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation

print("Mutation rate: 0.3")
logmodel = LogisticRegression(max_iter = 1000)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.30,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation

print("Mutation rate: 0.4")
logmodel = LogisticRegression(max_iter = 1000)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.40,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation


print("Mutation rate: 0.5")
logmodel = LogisticRegression(max_iter = 1000)
X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)
chromo_df_bc,score_bc=generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
n_parents=20,mutation_rate=0.50,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

print("Scores = ", score_bc)
# plot(score_bc,0.9,1.0,c = "gold") # accuracy vs mutation



# fin...









# pd - speech features dataset (L)(Too Big) --------------------------------------------------------------------------------

# data_bc = pd.read_csv("datasets/pd_speech_features.csv")
# label_bc = data_bc["gender"]
# #label_bc = np.where(label_bc == '1',1,0)
# data_bc.drop(["gender", "id"],axis = 1,inplace = True)

# print("pd speech dataset:\n",data_bc.shape[0],"Records\n",data_bc.shape[1],"Features")

# print(data_bc.head())

# score1 = acc_score(data_bc,label_bc)
# print(score1)


# logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
# X_train,X_test, Y_train, Y_test = split(data_bc,label_bc)

# chromo_df_bc,score_bc = generations(data_bc,label_bc,size=20,n_feat=data_bc.shape[1],
# n_parents=20,mutation_rate=0.20,n_gen=10, X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)

# plot(score_bc,0.7,1.0,c = "gold") # accuracy vs generation

# print(score_bc)