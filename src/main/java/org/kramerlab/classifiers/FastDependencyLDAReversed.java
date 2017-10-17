package org.kramerlab.classifiers;

import org.kramerlab.interfaces.*;
import org.kramerlab.inferencers.*;

import cc.mallet.types.InstanceList;
import cc.mallet.types.Instance;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.LabelSequence;
import cc.mallet.topics.TopicAssignment;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.types.Alphabet;
import cc.mallet.util.Randoms;
import cc.mallet.types.IDSorter;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Stack;

public class FastDependencyLDAReversed implements TopicModel,Inferencer{

    boolean evaluate;
    public double beta;
    double betaSum;
    double[] gammaLabel;
    double testgammaLabel;
    public double gammaLabelSum;
    public double testgammaLabelSum;
    double gammaTopic;
    double gammaTopicSum;
    double alpha;
    public double alphaSum;
    public double testalpha;
    int[][] typeTopicCounts;
    int[] tokensPerTopic;
    int[][] typeTopicDCounts;
    int[] tokensPerDTopic;

    int numWords;
    int numTopics;
    int numLabels;

    int rejections;
    int accepts;

    LabelAlphabet topicAlphabet;
    LabelAlphabet topicDAlphabet;
    Alphabet alphabet;
    ArrayList<TopicAssignment> data;
    ArrayList<TopicAssignment> dataD;

    Randoms random;
    AliasInferencer aliasInferencer;

    /**reversed version where the supervised level is the upper level*/
    public FastDependencyLDAReversed(LabelAlphabet la,double alphasum,double beta,double gammasum,double testalphasum,double testgammasum,int numTopics){
        this.alphaSum = alphasum;
        this.evaluate = false;
        this.random = new Randoms();
        this.topicDAlphabet = la;
        this.topicAlphabet = newLabelAlphabet(numTopics);
        this.numTopics = la.size();
        this.numLabels = numTopics;
        this.alpha = alphasum/(double)numTopics;
        this.beta = beta;
        this.gammaLabelSum = gammasum;
        this.gammaLabel = new double[numLabels];
        for(int i = 0;i<gammaLabel.length;++i){
            gammaLabel[i] = gammasum/(double)numLabels;
        }
        this.testalpha = testalphasum/numTopics;
        this.testgammaLabelSum = testgammasum;
        this.testgammaLabel = testgammasum/numLabels;
        this.gammaTopic=testgammaLabel;
        this.gammaTopicSum = this.numLabels*this.gammaTopic;
        this.data = new ArrayList<TopicAssignment>();
        this.dataD = new ArrayList<TopicAssignment>();
        this.tokensPerTopic = new int[this.numLabels];
        this.tokensPerDTopic = new int[this.numTopics];
        this.typeTopicDCounts = new int[this.numLabels][this.numTopics];
    }

    public void setEvaluate(boolean evaluate){
        this.evaluate = evaluate;
        if(evaluate){
            this.rejections=0;
            this.accepts=0;
            double[] alphainf = new double[numTopics];
            for(int i = 0;i<alphainf.length;++i){
                alphainf[i]=testalpha;
            }
            this.aliasInferencer = new AliasInferencer(typeTopicDCounts,tokensPerDTopic,alphainf,gammaTopic);
        }
    }


    public void addInstances(InstanceList instances){
        this.alphabet = instances.getDataAlphabet();
        this.numWords = alphabet.size();
        this.betaSum = beta*numWords;
        this.typeTopicCounts = new int[numWords][numLabels];
        int totalNumTokens=0;
        for(Instance inst:instances){
            FeatureSequence tokens = (FeatureSequence) inst.getData();
            FeatureVector labelVector = (FeatureVector)inst.getTarget();
            int local_numLabels = labelVector.numLocations();
            if(local_numLabels>0){
                int docLength = tokens.size();
                totalNumTokens+=docLength;
                LabelSequence topicSequence =
                    new LabelSequence(topicAlphabet, new int[docLength]);
                LabelSequence dtopicSequence = new LabelSequence(this.topicDAlphabet,new int[docLength]);
                int[] topics = topicSequence.getFeatures();
                int[] dtopics = dtopicSequence.getFeatures();
                for (int position = 0; position < docLength; position++) {
                    //the topic can only come from the labels
                    int chosenrank = random.nextInt(local_numLabels);
                    int dtopic = labelVector.indexAtLocation(chosenrank);
                    int topic = random.nextInt(this.numLabels);
                    topics[position] = topic;
                    dtopics[position] = dtopic;
                }
                TopicAssignment t = new TopicAssignment (inst, topicSequence);
                TopicAssignment dt = new TopicAssignment(inst, dtopicSequence);
                data.add(t);
                dataD.add(dt);     
            }else{
                System.out.println("instance without label");
            }       
        }
        buildInitialTypeTopicCounts();
        buildInitialDTypeTopicCounts();
    }


    public void buildInitialDTypeTopicCounts(){
        int c = 0;
        for(TopicAssignment datum:data){
            FeatureSequence labelSequence = (FeatureSequence) datum.topicSequence; // the labels for each word
            FeatureSequence topicSequence = (FeatureSequence) dataD.get(c).topicSequence;
            if(labelSequence.size()!=topicSequence.size())System.out.println("different sizes");
            int[] topics = labelSequence.getFeatures();
            int[] dtopics = topicSequence.getFeatures();
            for (int position = 0; position < labelSequence.size(); position++) {
                int topic = topics[position];
                int dtopic = dtopics[position];
                tokensPerDTopic[dtopic]++;
                typeTopicDCounts[topic][dtopic]++;
            }
            c++;
        }

    }


    public void buildInitialTypeTopicCounts(){
        for(TopicAssignment datum:data){
            FeatureSequence tokens = (FeatureSequence) datum.instance.getData();
            FeatureSequence topicSequence = (FeatureSequence) datum.topicSequence;
            
            int[] topics = topicSequence.getFeatures();
            if(tokens.size()!=topicSequence.size())System.out.println("different sizes "+tokens.size()+" "+topics.length+" "+tokens);

            for (int position = 0; position < tokens.size(); position++) {
                int topic = topics[position];
                int type = tokens.getIndexAtPosition(position);

                tokensPerTopic[topic]++;
                typeTopicCounts[type][topic]++;
            }
        }
    }

    public double[] getSampledDistribution(Instance instance,int numIterations, int thinning, int burnin){
        double[] result = new double[numTopics];
        double norm = 0;
        FeatureSequence tokenSequence =
            (FeatureSequence) instance.getData();
        int docLength = tokenSequence.getLength();
        int[] topics = new int[docLength];
        int[] dtopics = new int[docLength];
        for(int position = 0;position<docLength;++position){
            topics[position]=random.nextInt(numLabels);
            dtopics[position]=random.nextInt(numTopics);
        }
        int[] allLabels = new int[numTopics];
        for(int i = 0;i<allLabels.length;++i){
            allLabels[i]=i;
        }
        FeatureVector labels = new FeatureVector(topicDAlphabet,allLabels);
        LabelSequence topicSequence =
            new LabelSequence(topicAlphabet, topics);
        LabelSequence dtopicSequence = new LabelSequence(this.topicDAlphabet,dtopics);
        topics = topicSequence.getFeatures();
        dtopics = dtopicSequence.getFeatures();
        for(int i = 0;i<numIterations;++i){            
            //this.aliasInferencer.getSampledTopics(topicSequence,dtopicSequence);
            this.sampleTopicsForOneDoc(topicSequence,dtopicSequence,labels);
            this.sampleLabelsForOneDoc(tokenSequence,topicSequence,dtopicSequence);
            if(i%thinning==0&&i>burnin){
                int[] localTokensPerDTopic = new int[numTopics];
                for(int position=0;position<docLength;++position){
                    localTokensPerDTopic[dtopics[position]]++;
                }
                //save a sample
                for(int label = 0;label<numTopics;++label){
                    double labelval=0;
                    for(int position = 0;position<docLength;++position){
                        int type = tokenSequence.getIndexAtPosition(position);
                        int topic = topics[position];
                        double val = (typeTopicCounts[type][topic]+beta)/(tokensPerTopic[topic]+betaSum)*(typeTopicDCounts[topic][label]+testgammaLabel)/(tokensPerDTopic[label]+testgammaLabelSum)*(localTokensPerDTopic[label]+testalpha);
                        labelval+=val;
                    }
                    if(docLength>0){
                        labelval/=docLength;
                    }else{
                        labelval = testalpha;
                    }
                    result[label]+=labelval;
                    norm+=labelval;
                }
                    

            }
        }
        //normalize
        for(int label = 0;label<numTopics;++label){
            result[label]/=norm;
        }
        return result;
    }

    public void sample(int iterations){
        for(int iteration = 0;iteration<iterations;++iteration){
            for(int doc = 0;doc<data.size();++doc){
                FeatureSequence tokenSequence =
                    (FeatureSequence) data.get(doc).instance.getData();
                LabelSequence topicSequence =
                    (LabelSequence) data.get(doc).topicSequence;
                FeatureVector labels =(FeatureVector) data.get(doc).instance.getTarget();
                FeatureSequence dtopicSequence = dataD.get(doc).topicSequence;
                sampleTopicsForOneDoc(topicSequence,dtopicSequence,labels);
                sampleLabelsForOneDoc(tokenSequence,topicSequence,dtopicSequence);
            }
        }
    }



    public void sampleTopicsForOneDoc(LabelSequence labelSequence,FeatureSequence topicSequence,FeatureVector labelVector){
        int local_numLabels = labelVector.numLocations();
        int[] oneDocTopics = labelSequence.getFeatures();
        int[] oneDocDTopics = topicSequence.getFeatures();
        int[] localTokensPerDTopic = new int[numTopics];
        for(int position = 0;position<oneDocTopics.length;++position){
            localTokensPerDTopic[oneDocDTopics[position]]++;
        }
        int docLength = topicSequence.getLength();
        for(int position = 0;position<docLength;++position){
            int topic = oneDocTopics[position];
            int dtopic = oneDocDTopics[position];
            //decrease counts
            if(!evaluate){    
                tokensPerDTopic[dtopic]--;
                typeTopicDCounts[topic][dtopic]--;
                if(tokensPerDTopic[dtopic]<0)System.out.println("decrease tokensPerDTopics"+" " +Arrays.toString(oneDocTopics));
                if(typeTopicDCounts[topic][dtopic]<0)System.out.println("typeDTopicCounts "+topic+" "+dtopic+" "+typeTopicDCounts[topic][dtopic]+" "+Arrays.toString(oneDocDTopics)+" "+Arrays.toString(oneDocTopics));
            }
            localTokensPerDTopic[dtopic]--;
            //sample new topic
            double[] probs = new double[local_numLabels];
            double mass = 0;

            for(int rank = 0;rank<local_numLabels;++rank){
                int top = labelVector.indexAtLocation(rank);
                if(!evaluate){
                    probs[rank] = (typeTopicDCounts[topic][top]+gammaTopic)/(tokensPerDTopic[top]+gammaTopicSum);//*(localTokensPerDTopic[top]+alpha);//
                }else{
                    probs[rank] = (typeTopicDCounts[topic][top]+gammaTopic)/(tokensPerDTopic[top]+gammaTopicSum)*(localTokensPerDTopic[top]+testalpha);
                }
                mass+=probs[rank];
            }
            int newTopic = -1;
            double rand = random.nextUniform()*mass;
            int c = 0;
            while(rand>0){
                newTopic++;
                rand-=probs[c];
                c++;
            }
            newTopic = labelVector.indexAtLocation(newTopic);
            //increase counts
            if(!evaluate){
                tokensPerDTopic[newTopic]++;
                typeTopicDCounts[topic][newTopic]++;
            }
            localTokensPerDTopic[newTopic]++;
            oneDocDTopics[position] = newTopic;
        }

    }

    public void sampleLabelsForOneDoc(FeatureSequence tokenSequence,LabelSequence labelSequence,FeatureSequence topicSequence){
        int[] oneDocTopics = labelSequence.getFeatures();
        int[] oneDocDTopics = topicSequence.getFeatures();
        int[] localTokensPerDTopic = new int[numTopics];
        for(int position = 0;position<oneDocTopics.length;++position){
            localTokensPerDTopic[oneDocDTopics[position]]++;
        }
        int docLength = tokenSequence.getLength();
        for(int position = 0;position<docLength;++position){
            int type = tokenSequence.getIndexAtPosition(position);
            int topic = oneDocTopics[position];
            int dtopic = oneDocDTopics[position];
            //decrease counts
            if(!evaluate){
                tokensPerTopic[topic]--;
                typeTopicCounts[type][topic]--;
                typeTopicDCounts[topic][dtopic]--;
            }
            //sample new topic
            double[] probs = new double[numLabels];
            double mass = 0;
            for(int top = 0;top<this.numLabels;++top){
                int label = top;
                if(!evaluate){
                    probs[top] = (typeTopicCounts[type][label]+beta)/(tokensPerTopic[label]+betaSum)*(typeTopicDCounts[label][dtopic]+gammaLabel[label]);
                }else{
                    probs[top] = (typeTopicCounts[type][label]+beta)/(tokensPerTopic[label]+betaSum)*(typeTopicDCounts[label][dtopic]+testgammaLabel);
                }
                mass+=probs[top];
            }
            int newTopic = -1;
            double rand = random.nextUniform()*mass;
            int c = 0;
            while(rand>0){
                newTopic++;
                rand-=probs[c];
                c++;
            }
            int newLabel = newTopic;
            //increase counts
            if(!evaluate){
                tokensPerTopic[newLabel]++;
                typeTopicCounts[type][newLabel]++;
                typeTopicDCounts[newLabel][dtopic]++;
            }
            oneDocTopics[position] = newLabel;
        }
    }

    public IDSorter[] topTopics(int label){
	IDSorter[] sortedTopics = new IDSorter[numTopics];
	for (int topic = 0; topic < numTopics; topic++) {
	    sortedTopics[topic] = new IDSorter(topic, typeTopicDCounts[topic][label]);
	}
	Arrays.sort(sortedTopics);
	return sortedTopics;
    }

    public IDSorter[] topLabels(int topic){
	IDSorter[] sortedLabels = new IDSorter[numTopics];
	for (int label = 0; label < numTopics; label++) {
	    sortedLabels[label] = new IDSorter(label, typeTopicDCounts[topic][label]);
	}
	Arrays.sort(sortedLabels);
	return sortedLabels;
    }


    //returns top words for specific topic
    public IDSorter[] topWords(int topic) {
	IDSorter[] sortedWords = new IDSorter[numWords];
	for (int type = 0; type < numWords; type++) {
	    sortedWords[type] = new IDSorter(type, typeTopicCounts[type][topic]);
	}
	Arrays.sort(sortedWords);
	return sortedWords;
    }



    public void printTopWords(String filename,int numTopWords,int threshold){
        try{
            BufferedWriter bw = new BufferedWriter(new FileWriter(filename));
            for(int topic = 0;topic<numLabels;++topic){
                IDSorter[] topWords = topWords(topic);
                bw.write("Topic "+topic+"&\\\\\n");
                bw.write("\\hline\n");
                bw.write("Word&Count\\\\\n");
                bw.write("\\hline\n");
                for(int wordInd = 0;wordInd<10;++wordInd){
                    String word = (String)alphabet.lookupObject(topWords[wordInd].getID());
                    double prob = topWords[wordInd].getWeight();
                    bw.write(word+"&"+prob+"\\\\\n");
                }
                bw.write("\\hline\n");
                bw.write("Label&Count\\\\\n");
                bw.write("\\hline\n");
                IDSorter[] topLabels = topLabels(topic);
                for(int labelInd = 0;labelInd<10;++labelInd){
                    String label = (String)topicDAlphabet.lookupObject(topLabels[labelInd].getID());
                    label = label.replace("TAG_","");
                    double prob = topLabels[labelInd].getWeight();
                    bw.write(label+"&"+prob+"\\\\\n");
                }
                bw.write("\\hline\n");
            }
            bw.flush();
            bw.close();
        }catch(Exception e){
            e.printStackTrace();
        }

    }

    public Inferencer getInferencer(){
        return this;
    }

    public static LabelAlphabet newLabelAlphabet (int numTopics) {
        LabelAlphabet ret = new LabelAlphabet();
        for (int i = 0; i < numTopics; i++)
            ret.lookupIndex("topic"+i);
        return ret;
    }


}
