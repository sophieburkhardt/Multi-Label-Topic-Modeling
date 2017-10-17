import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[1]),'liblinear/python/'))

from liblinearutil import *
import time
from evaluation import Evaluation

class BR:

    def __init__(self,labels=[],classifiers = [],pathname=None,weighted_c=False):
        self.pathname = pathname
        self.classifiers = classifiers
        self.labels = labels
        self.cs = [1 for l in self.labels]
        self.weighted_c = weighted_c

    def set_cs(self,cs):
        self.cs = cs

    def train_br(self,data,params,optimize,label_range=None,add_to_model=False):
        """
        label_range: range of labels to train, default: train all labels
        """
        if label_range is None:
            label_range = xrange(len(self.labels))
        c_values=[]
        c_options = [0.001,0.01,0.1,1,10,100]
        if params:
            param = parameter(params)
        else:
            param = parameter('')
        param.show()
        #for i,label in enumerate(self.labels):
        for i in label_range:
            print 'train model',i
            label_data,ratio = self.transform(data,i)
            if self.weighted_c:
                weightstring=' -w1 '+str(1./ratio)
            else:
                weightstring = ''
            print 'transformed'#,label_data[0]
            chosen_c = self.cs[i]
            if optimize:
                accs = []
                max_acc = 0
                for c in c_options:
                    crossparam = parameter('-v 3 -c '+str(c)+weightstring)
                    acc = train(problem(label_data[0],label_data[1]), crossparam)
                    accs.append(acc)
                    if acc>max_acc:
                        chosen_c = c
                        max_acc = acc
                    elif acc<max_acc:
                        #save some computation time by breaking off early if accuracy is going down
                        break
                    print 'c and accuracy',c,acc
                param = parameter('-c '+str(chosen_c)+weightstring)
                print 'chose',chosen_c
            c_values.append(chosen_c)
            self.cs[i]=chosen_c
            m = train(problem(label_data[0],label_data[1]), param)
            if add_to_model:
                self.classifiers.append(m)
            self.save_specific_model(self.pathname,i,m)
        return c_values
            
    def predict_br(self,instances,labelrange=None):
        transformed,ratio = self.transform(instances,0)
        if labelrange is None:
            labelrange = xrange(len(self.labels))
        toolbar_width = 40
        batch_size = len(labelrange)/toolbar_width
        if batch_size==0:
            batch_size=1
        # setup toolbar
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
        
        predicted=[[] for d in instances]
        for i in labelrange:
            if i>0 and i%batch_size==0:
                sys.stdout.write("-")
                sys.stdout.flush()
            if i<len(self.classifiers) and self.classifiers[i] is not None:
                prediction = predict(transformed[0],transformed[1],self.classifiers[i],options='-b 1')
                for j,p in enumerate(prediction[0]):
                    if p==1:
                        predicted[j].append(i)
        sys.stdout.write("\n")
        fm,p,r = Evaluation.get_f_measure(predicted,instances)
        print fm,p,r
        return predicted
        
    def predict_br_probs(self,instances,labelrange=None):
        transformed,ratio = self.transform(instances,0)
        if labelrange is None:
            labelrange = xrange(len(self.labels))
        toolbar_width = 40
        batch_size = len(labelrange)/toolbar_width
        if batch_size==0:
            batch_size=1
        # setup toolbar
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
        
        predicted=[[0.0 for i in labelrange if i<len(self.classifiers)] for d in instances]
        c=0
        for i in labelrange:
            if i>0 and i%batch_size==0:
                sys.stdout.write("-")
                sys.stdout.flush()
            if i<len(self.classifiers) and self.classifiers[i] is not None:
                ls = self.classifiers[i].get_labels()
                prediction = predict(transformed[0],transformed[1],self.classifiers[i],options='-b 1')
                try:
                    for j,p in enumerate(prediction[2]):
                        if len(p)==1:
                            #special case where the value is always true or false
                            if ls[0]==0:
                                predicted[j][c]=0.0
                            else:
                                predicted[j][c]=1.0
                        else:
                            if ls[0]==0:
                                predicted[j][c]=p[1]
                            else:
                                predicted[j][c]=p[0]
                except IndexError:
                    print p,j,c,len(predicted),len(predicted[j])
                    sys.exit(1)
            c+=1
        sys.stdout.write("\n")
        #fm,p,r = Evaluation.get_f_measure(predicted,instances)
        #print fm,p,r
        #print predicted
        return predicted
        
    def transform(self,data,label_index):
        """
        reduce to one label given by label_index
        data input format: [{labels:[true labels],text:[features]}]
        data output format: ([true label],[{feature:value}])
        """
        result = ([],[])
        count = 0
        for i,d in enumerate(data):
            doclabels = d['labels']
            docfeatures = d['text']
            if docfeatures:
                if label_index in doclabels:
                    result[0].append(1)
                    count+=1
                else:
                    result[0].append(0)
                doc = {}
                for f in docfeatures:
                    doc[int(f)+1]=docfeatures[f]
                result[1].append(doc)
        return result,(count/float(len(data)))
        
    def save_model(self,pathname):
        for i,label in enumerate(self.labels):
            print 'save model',i,(pathname+'model-'+str(i))
            save_model(pathname+'model-'+str(i), self.classifiers[i])
            
    def save_specific_model(self,pathname,num,model):
        print 'save model',i,(pathname+'model-'+str(num))
        save_model(pathname+'model-'+str(num), model)
        
    @staticmethod
    def load_model(pathname,labels,labelrange=None,weighted_c=True):
        if labelrange is None:
            labelrange = xrange(len(labels))
        classifiers = []
        for i,label in enumerate(labels):
            if i in labelrange:
                m = load_model(pathname+'model-'+str(i))
            else:
                m=None
            classifiers.append(m)
        return BR(labels,classifiers,weighted_c)
                
    def get_f_measure(self,predicted,instances):
        """
        label-based Micro-F-Measure
        Z: true values
        Y: predicted values
        return f-measure, precision, recall
        """
        tp = 0
        sum_all = 0
        sum_y=0
        for i,n in enumerate(predicted):
            trues = instances[i]['labels']
            for j,val in enumerate(n):
                #if val==1:
                sum_y+=1
                sum_all+=1
                if val in trues:
                    tp+=1
                    #sum_all+=1
            sum_all+=len(trues)
        print tp,'num labels',(sum_all-sum_y),'num predicted labels',sum_y
        return 2*tp/float(sum_all),(tp/float(sum_y)),(tp/float(sum_all-sum_y))
