import pandas as pd
import numpy as np
import math


# small change here, all i do is consider cols that have at least 1 mut
data = pd.read_csv("mutations_1.csv")
data2 = data.copy()
data2 = data2.loc[:, (data2.sum(axis=0)!= 0)]

#a. Construct a bootstrap dataset from the full set of samples. 
#The bootstrap data set should contain the same number of samples as the original dataset.
def bootstrap(df):
    bootstrapdf = df.sample(frac=1, replace=True)

    outofbagdf = pd.concat([bootstrapdf,df]).drop_duplicates(keep=False)
    
    return bootstrapdf, outofbagdf

#for each node splitting decision, randomly select 
#sqrt(n) features to consider (where, n is the total number of genetic mutations)
def consider(df):
#     rows = df['class']
#     sample_this = df.iloc[:, 1:].copy()
#     cols = df.columns
#     coln = len(cols)
#     reduce_by = math.floor(math.sqrt(coln))
#     df_reduced = sample_this.sample(n=reduce_by, axis = 1)
#     df_reduced.insert(0, "class", rows)
    df_reduced = df
    return df_reduced

#for all node splitting decisions, use EITHER the phi function OR information gain 
#(choose one method and always use it)
def phi_function(df):
    #Get all the column names slice off 'class'
    cols = df.columns
    cols = cols[1:]
    
    #Counts for #of samples at node T for total, C and NC
    nt = 0
    nt_C = 0
    nt_NC = 0
    
    #Gives me an array where C = 1 and NC = 0 (samples)
    rows = df['class']
    samples = []
    for s in rows:
        if s.startswith('C'):
            samples.append(1)
            nt_C += 1
        else:
            samples.append(0)
            nt_NC += 1
    
    nt = nt_C + nt_NC
    
    idx = 0
    
    #Number of samples at each mutation having or not having the mutation
    nt_left = []
    nt_right = []
    
    #Number of samples at each mutation having or not having the mutation
    #specified by sample
    nt_left_C = []
    nt_left_NC = []
    nt_right_C = []
    nt_right_NC = []
    
    while idx < len(cols):
        #Makes array of each mutations occurrence across the samples
        #Array of 1 and 0 where it is present or not
        mut_occur = df[cols[idx]].to_numpy()
        
        #Gets the number of samples at each mutation
        l = 0
        r = 0
        lc = 0
        lnc = 0
        rc = 0
        rnc = 0
        idx2 = 0
        
        #Count L and R samples
        #Count L and R cancer and nocancer samples
        for m in mut_occur:
            if (m == 1):
                l += 1
                if (samples[idx2] == 1):
                    lc += 1
                else:
                    lnc += 1  
            else:
                r += 1
                if (samples[idx2] == 1):
                    rc += 1
                else:
                    rnc += 1
            idx2 += 1
            
        nt_left.append(l)
        nt_right.append(r)
        nt_left_C.append(lc)
        nt_left_NC.append(lnc)
        nt_right_C.append(rc)
        nt_right_NC.append(rnc)
        
        idx += 1
        
        
    percent_left = []
    percent_right = []
    percent_left_C = []
    percent_left_NC = []
    percent_right_NC = []
    percent_right_C = []
    pelipper = []
    Q = []
    phi = []
    
    #Calculates percentages for the Right and Left as well as C and NC for each
    #Calculates other info including phi as well
    idx3 = 0
    while idx3 < len(cols):
        percent_left.append(nt_left[idx3] / nt)
        percent_right.append(nt_right[idx3] / nt)
        
        if(nt_left[idx3] != 0):
            percent_left_C.append(nt_left_C[idx3] / (nt_left[idx3]))
            percent_left_NC.append(nt_left_NC[idx3] / (nt_left[idx3]))
        else:
            percent_left_C.append(0)
            percent_left_NC.append(0)
            
        if(nt_right[idx3] != 0):
            percent_right_C.append(nt_right_C[idx3] / (nt_right[idx3]))
            percent_right_NC.append(nt_right_NC[idx3] / (nt_right[idx3]))
        else:
            percent_right_C.append(0)
            percent_right_NC.append(0)
            
        pelipper.append(2*((percent_left[idx3]) * (percent_right[idx3])))
        
        Q.append(abs((percent_left_C[idx3])-(percent_right_C[idx3])) + abs((percent_left_NC[idx3])-(percent_right_NC[idx3])))
        
        phi.append(pelipper[idx3] * Q[idx3])
        
        idx3 += 1
    
    #Builds a dataframe of the results
    resultdf = df = pd.DataFrame(cols, columns = ['Genetic Mutation'])
    resultdf.insert(1, "n(t_L)", nt_left, True)
    resultdf.insert(2, "n(t_R)", nt_right, True)
    resultdf.insert(3, "n(t_L_C)", nt_left_C, True)
    resultdf.insert(4, "n(t_L_NC)", nt_left_NC, True)
    resultdf.insert(5, "n(t_R_C)", nt_right_C, True)
    resultdf.insert(6, "n(t_R_NC)", nt_right_NC, True)
    resultdf.insert(7, "P_L", percent_left, True)
    resultdf.insert(8, "P_R", percent_right, True)
    resultdf.insert(9, "P(C | t_L)", percent_left_C, True)
    resultdf.insert(10, "P(NC | t_L)", percent_left_NC, True)
    resultdf.insert(11, "P(C | t_R)", percent_right_C, True)
    resultdf.insert(12, "P(NC | t_R)", percent_right_NC, True)
    resultdf.insert(13, "2PLPR", pelipper, True)
    resultdf.insert(14, "Q", Q, True)
    resultdf.insert(15, "phi", phi, True)
    
    resultdf = resultdf.sort_values(by=['phi'], ascending= False)
    
    #Gets index of the top (only) row
    dfindex = resultdf.head(1).index.values.astype(int)[0]
    best_feature = cols[dfindex]
    
    return resultdf, best_feature

def groups (df, best_feature):
    
    #Makes a list of the given samples
    samples = df['class']
    samples = list(samples)
    
    #Creates dataframe and list for the left group
    left_list = []
    #drops rows where a col that has best_feature = 0
    left = df.drop(df.loc[df[best_feature]==0].index)
    samples_A = left['class']
    
    for x in samples_A:
        left_list.append(x)
        
    #Creates dataframe and list for the right group    
    right_list = []    
    #drops rows where a col that has best_feature = 1
    right = df.drop(df.loc[df[best_feature]!=0].index)
    samples_B = right['class']
    
    for y in samples_B:
        right_list.append(y)   
           
    return left, right, left_list, right_list

#Classify the leaf nodes
def classifyas (sample_list):
    left_weight = 0
    right_weight = 0
    classification = []
    
    #Count times C or NC appears in the leaf nodes
    for pred in sample_list:
        if pred.startswith('C'):
            left_weight += 1
        else:
            right_weight += 1
          
    #Return 1 or 0 if it should be classified as C or NC
    if left_weight >= right_weight: # >= used to make a FP prediction over a FN prediction
        return 1
    else:
        return 0

#Only use is to turn 1 and 0 to C and NC from classify
def classifyas_help(LA, LR, RA, RR):
    num_class = []
    num_class.append(LA)
    num_class.append(LR)
    num_class.append(RA)
    num_class.append(RR)
    name_class = []
    
    #Appends C and NC for 1 and 0
    for num in num_class:
        if (num == 1):
            name_class.append('C')
        else:
            name_class.append('NC')
    return name_class

def classify(sample, roots, left_nodes, right_nodes, classes):
    cols = sample.columns
    cols = cols[1:]
    
    #Gets index vals of incoming DF
    sample_index = list(sample.index.values)
    
    predict = []
    
    tree = len(roots)
    run = 0
    
    while(run < tree):
        if((sample[roots[run]].loc[sample.index[0]]) == 1):
            if((sample[left_nodes[run]].loc[sample.index[0]]) == 1):
                predict.append(classes[run][0])
            else:
                predict.append(classes[run][1])
        else:
            if((sample[right_nodes[run]].loc[sample.index[0]]) == 1):
                predict.append(classes[run][2])
            else:
                predict.append(classes[run][3])  
        run = run+1
    
    ccount = 0
    ncount = 0
    for x in predict:
        if(x == "C"):
            ccount = ccount + 1
        else:
            ncount = ncount + 1
            
    majority = ""
    
    if(ccount >= ncount):
        majority = "C"
    else:
        majority = "NC"
            
    return majority, ccount, ncount

def rfc(root, left, right, class_list, original):
    #     samples = testdf['class']
    cols = original.columns
    cols = cols[1:]
    
    #Gets index vals of incoming DF
    sample_index = list(original.index.values)
    
    predict = []
    
    original = original.drop_duplicates()
#    display(original)
    #Check "Cell" at (current index * mutation name)
    #Walk through tree and append the classification from training
    #Loops are looking at a cell at x (df-index) and col (mut)
    for x in sample_index:
        #print(original.loc[x][root])
        if ((original.loc[x][root]) == 1):
            if (original.loc[x][left] == 1):
                predict.append(class_list[0])#A1
            else:
                predict.append(class_list[1])#A2
        else:
            if (original.loc[x][right] == 1):
                predict.append(class_list[2])#B1
            else:
                predict.append(class_list[3])#B2

    return predict #list of sample has cancer or no cancer

def metrics(pred, test):
    #Gets a list of samples from the class col
    samples = test['class']
    actual = []
    #Makes C = 1 and NC = 0
    for s in samples:
        if s.startswith('C'):
            actual.append(1)
        else:
            actual.append(0)
    
    predicted = []
    #For the predictions makes C = 1 and NC = 0
    for p in pred:
        if p.startswith('C'):
            predicted.append(1)
        else:
            predicted.append(0)
            
    TP,FP,FN,TN = 0,0,0,0   
    
    #makes list of [(pred,actual), ...] == (i,j) and calculates TP,FP,FN,TN
    for i,j in zip(predicted, actual):
        if (i == 1) and (j == 1):
            TP += 1
        if (i == 1) and (j == 0):
            FP += 1
        if (i == 0) and (j == 1):
            FN += 1
        if (i == 0) and (j == 0):
            TN += 1
            
    accuracy = 0        
    sensitivity = 0
    FO = 0       
    specificity = 0
    precision = 0
    missrate = 0
    FD = 0
    
    #Metric calculations
    if((TP+TN+FP+FN) != 0):
        accuracy = (TP+TN)/(TP+TN+FP+FN)
    else:
        accuracy = 0
    if((TP+FN) != 0):
        sensitivity = TP/(TP+FN)
        FO = FN/(FN+TP)
    else: 
        sensitivity = 0
        FO = 0
    if((TN+FP) != 0):
        specificity = TN/(TN+FP)
    else:
        specificity = 0
    if((TP+FP) != 0):
        precision = TP/(TP+FP) 
        missrate = FN/(TP+FP)
    else:
        precision = 0
        missrate = 0
    if((FP+TP)!=0):
        FD = FP/(FP+TP)
    else:
        FD = 0
    
    return accuracy, sensitivity, FO, specificity, precision, missrate, FD

def main():
    copy = data2.copy()
    
    num_trees = 1
    
    roots = []
    lefts = []
    rights = []
    classes = []
    
    originals = []
    tests = []
    
    x = 0
    while(x < num_trees):
        boot, bag = bootstrap(copy)
        red_boot = consider(boot)
        
        node_df, node = phi_function(red_boot)
        group_A, group_B, list_A, list_B = groups(red_boot, node)
        
        left_df, left_node = phi_function(group_A)
        l_group_A, l_group_B, l_list_A, l_list_B = groups(group_A, left_node)
        
        right_df, right_node = phi_function(group_B)
        r_group_A, r_group_B, r_list_A, r_list_B = groups(group_B, right_node)
        
        l_A_class = classifyas (l_list_A)
        l_B_class = classifyas (l_list_B)
        r_A_class = classifyas (r_list_A)
        r_B_class = classifyas (r_list_B)
        
        class_all = classifyas_help(l_A_class, l_B_class, r_A_class, r_B_class)
        
        
        
        roots.append(node)
        lefts.append(left_node)
        rights.append(right_node)
        classes.append(class_all)
        tests.append(bag)
        originals.append(red_boot)
        
        x = x + 1
    
    base_results = []
    acc = []
    sen = []
    FOl = []
    spe = []
    pre = []
    mis = []
    FDl = []
    bag_acc = []
    bag_sen = []
    bag_FOl = []
    bag_spe = []
    bag_pre = []
    bag_mis = []
    bag_FDl = []
    
    treenames = []
    run = 0
    while(run < num_trees):
        treenames.append("Tree " + str(run + 1))
        preds = rfc(roots[run], lefts[run], rights[run], classes[run], originals[run])
        accuracy, sensitivity, FO, specificity, precision, missrate, FD = metrics(preds, originals[run])
        bag_preds = rfc(roots[run], lefts[run], rights[run], classes[run], tests[run])
        bag_accuracy, bag_sensitivity, bag_FO, bag_specificity, bag_precision, bag_missrate, bag_FD = metrics(bag_preds, tests[run])
        acc.append(accuracy)
        sen.append(sensitivity)
        FOl.append(FO)
        spe.append(specificity)
        pre.append(precision)
        mis.append(missrate)
        FDl.append(FD)
        
        bag_acc.append(bag_accuracy)
        bag_sen.append(bag_sensitivity)
        bag_FOl.append(bag_FO)
        bag_spe.append(bag_specificity)
        bag_pre.append(bag_precision)
        bag_mis.append(bag_missrate)
        bag_FDl.append(bag_FD)
        #base_results.append()
        run = run + 1
        
    overall_acc = (sum(acc)/len(acc))
    overall_sen = (sum(sen)/len(sen))
    overall_FOl = (sum(FOl)/len(FOl))
    overall_spe = (sum(spe)/len(spe))
    overall_pre = (sum(pre)/len(pre))
    overall_mis = (sum(mis)/len(mis))
    overall_FDl = (sum(FDl)/len(FDl))
    
    metricdf = pd.DataFrame(treenames, columns = ['Bootstrap Metrics'])
    metricdf.insert(1, "Accuracy", acc, True)
    metricdf.insert(2, "Sensitivity", sen, True)
    metricdf.insert(3, "Specificity", spe, True)
    metricdf.insert(4, "Precision", pre, True)
    metricdf.insert(5, "Missrate", mis, True)
    metricdf.insert(6, "False Detection", FDl, True)
    metricdf.insert(7, "False Omission", FOl, True)
    
    bag_metricdf = pd.DataFrame(treenames, columns = ['Out-Of-Bag Metrics'])
    bag_metricdf.insert(1, "Accuracy", bag_acc, True)
    bag_metricdf.insert(2, "Sensitivity", bag_sen, True)
    bag_metricdf.insert(3, "Specificity", bag_spe, True)
    bag_metricdf.insert(4, "Precision", bag_pre, True)
    bag_metricdf.insert(5, "Missrate", bag_mis, True)
    bag_metricdf.insert(6, "False Detection", bag_FDl, True)
    bag_metricdf.insert(7, "False Omission", bag_FOl, True)
    
    print("AVERAGE STATS FOR BOOTSTRAP DATA")
    print("Average Accuracy: ", overall_acc)
    print("Average Precision: ", overall_pre)
    print("Average Sensitivity: ", overall_sen)
    print("Average Specificity: ", overall_spe)
    print("Average Miss: ", overall_mis)
    print("Average FD: ", overall_FDl)
    print("Average FO: ", overall_FOl)
    
    display(metricdf)
    print("------------------------------------------")
    
    
    print("AVERAGE STATS FOR OUT-OF-BAG DATA")
    print("Average Accuracy: ", sum(bag_acc)/len(bag_acc))
    print("Average Precision: ", sum(bag_pre)/len(bag_pre))
    print("Average Sensitivity: ", sum(bag_sen)/len(bag_sen))
    print("Average Specificity: ", sum(bag_spe)/len(bag_spe))
    print("Average Miss: ", sum(bag_mis)/len(bag_mis))
    print("Average FD: ", sum(bag_FDl)/len(bag_FDl))
    print("Average FO: ", sum(bag_FOl)/len(bag_FOl))
    
    display(bag_metricdf)
    
    print("------------------------------------------")
    
    print("Run Accuracy: ", ((sum(acc)/len(acc))+(sum(bag_acc)/len(bag_acc)))/2   )
    
main()