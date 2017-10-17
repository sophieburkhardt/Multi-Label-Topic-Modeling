package org.kramerlab.classifiers;

import org.kramerlab.interfaces.*;
import org.kramerlab.inferencers.*;

import cc.mallet.types.IDSorter;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Instance;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.LabelSequence;
import cc.mallet.topics.TopicAssignment;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.types.Alphabet;
import cc.mallet.util.Randoms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Stack;
import java.io.BufferedWriter;
import java.io.FileWriter;

public class FastDependencyLDA implements TopicModel,Inferencer{

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

    LabelAlphabet topicAlphabet;
    LabelAlphabet topicDAlphabet;
    Alphabet alphabet;
    ArrayList<TopicAssignment> data;
    ArrayList<TopicAssignment> dataD;

    Randoms random;
    AliasInferencer aliasInferencer;
    
    int[] localTokensPerDTopic;
    int[] localTokensPerTopic;
    double[] labelDist;
    
    boolean s2;

    public FastDependencyLDA(LabelAlphabet la,double alphasum,double beta,double gammasum,double testalphasum,double testgammasum,int numTopics){
        this.alphaSum = alphasum;
        this.evaluate = false;
        this.random = new Randoms();
        this.topicAlphabet = la;
        this.topicDAlphabet = newLabelAlphabet(numTopics);
        this.numTopics = numTopics;
        this.numLabels = la.size();
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
        this.gammaTopicSum = numLabels*this.gammaTopic;
        this.data = new ArrayList<TopicAssignment>();
        this.dataD = new ArrayList<TopicAssignment>();
        this.tokensPerTopic = new int[numLabels];
        this.tokensPerDTopic = new int[numTopics];
        this.typeTopicDCounts = new int[numLabels][numTopics];
        this.localTokensPerDTopic = new int[numTopics];
        this.localTokensPerTopic = new int[numLabels];
        this.labelDist = new double[numLabels];
    }

    public void setEvaluate(boolean evaluate){
        this.evaluate = evaluate;
        if(evaluate){
            double[] alphainf = new double[numTopics];
            for(int i = 0;i<alphainf.length;++i){
                alphainf[i]=testalpha;
            }
            this.aliasInferencer = new AliasInferencer(typeTopicDCounts,tokensPerDTopic,alphainf,gammaTopic);
        }
    }

    public void setEvalStrategy(boolean s2){
        this.s2=s2;
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
                    int topic = labelVector.indexAtLocation(chosenrank);
                    topics[position] = topic;
                    int dtopic = random.nextInt(this.numTopics);
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
        //System.out.println("sample distribution "+evaluate);
        double[] result = new double[numLabels];
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
        int[] allLabels = new int[numLabels];
        for(int i = 0;i<allLabels.length;++i){
            allLabels[i]=i;
        }
        FeatureVector labels = new FeatureVector(topicAlphabet,allLabels);
        LabelSequence topicSequence =
            new LabelSequence(topicAlphabet, topics);
        LabelSequence dtopicSequence = new LabelSequence(this.topicDAlphabet,dtopics);
        topics = topicSequence.getFeatures();
        dtopics = dtopicSequence.getFeatures();
        for(int i = 0;i<numIterations;++i){            
        
            this.aliasInferencer.getSampledTopics(topicSequence,dtopicSequence);            
            //this.sampleTopicsForOneDoc(topicSequence,dtopicSequence);
            this.sampleLabelsForOneDoc(tokenSequence,topicSequence,labels,dtopicSequence);

            if(i%thinning==0&&i>burnin){
                Arrays.fill(localTokensPerDTopic,0);
                Arrays.fill(localTokensPerTopic,0);
                for(int position=0;position<docLength;++position){
                    localTokensPerDTopic[dtopics[position]]++;
                    localTokensPerTopic[topics[position]]++;
                }
                if(s2){
                    Arrays.fill(labelDist,0);
                    double labelSum=0.0;
                    for(int label = 0;label<numLabels;++label){
                        for(int topic = 0;topic<numTopics;++topic){
                            double val = (typeTopicDCounts[label][topic]+testgammaLabel)/(tokensPerDTopic[topic]+testgammaLabelSum)*(localTokensPerDTopic[topic]+testalpha);
                            labelDist[label]+=val;
                            labelSum+=val;
                        }
                    }
                    //save a sample
                    for(int label = 0;label<numLabels;++label){
                        double labelval=0;
                        for(int position = 0;position<docLength;++position){
                            int type = tokenSequence.getIndexAtPosition(position);
                            //   int topic = topics[position];
                            int dtopic = dtopics[position];
                            double val = 0.0;
                            //val = (typeTopicCounts[type][label]+beta)/(tokensPerTopic[label]+betaSum)*(typeTopicDCounts[label][dtopic]+testgammaLabel)/(tokensPerDTopic[dtopic]+testgammaLabelSum)*(localTokensPerDTopic[dtopic]+testalpha)/(docLength+testalpha*docLength);
                            val = (typeTopicCounts[type][label]+beta)/(tokensPerTopic[label]+betaSum)*(localTokensPerTopic[label]+docLength*labelDist[label]/labelSum);
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
                }else{
                    //save a sample
                    for(int label = 0;label<numLabels;++label){
                        double labelval=0;
                        for(int position = 0;position<docLength;++position){
                            int type = tokenSequence.getIndexAtPosition(position);
                            //   int topic = topics[position];
                            int dtopic = dtopics[position];
                            double val = (typeTopicCounts[type][label]+beta)/(tokensPerTopic[label]+betaSum)*(typeTopicDCounts[label][dtopic]+testgammaLabel)/(tokensPerDTopic[dtopic]+testgammaLabelSum)*(localTokensPerDTopic[dtopic]+testalpha);
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
        }
        //normalize
        for(int label = 0;label<numLabels;++label){
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
                sampleTopicsForOneDoc(topicSequence,dtopicSequence);
                sampleLabelsForOneDoc(tokenSequence,topicSequence,labels,dtopicSequence);
            }
        }
    }




    public void sampleTopicsForOneDoc(LabelSequence labelSequence,FeatureSequence topicSequence){
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
            double[] probs = new double[numTopics];
            double mass = 0;

            for(int top = 0;top<numTopics;++top){
                if(!evaluate){
                    probs[top] = (typeTopicDCounts[topic][top]+gammaTopic)/(tokensPerDTopic[top]+gammaTopicSum)*(localTokensPerDTopic[top]+alpha);//
                }else{
                    probs[top] = (typeTopicDCounts[topic][top]+gammaTopic)/(tokensPerDTopic[top]+gammaTopicSum)*(localTokensPerDTopic[top]+testalpha);
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
            //increase counts
            if(!evaluate){
                tokensPerDTopic[newTopic]++;
                typeTopicDCounts[topic][newTopic]++;
            }
            localTokensPerDTopic[newTopic]++;
            oneDocDTopics[position] = newTopic;
        }

    }

    public void sampleLabelsForOneDoc(FeatureSequence tokenSequence,LabelSequence labelSequence,FeatureVector labelVector,FeatureSequence topicSequence){
        int local_numLabels = labelVector.numLocations();
        int[] labelIndices=new int[local_numLabels];
        for(int i=0;i<local_numLabels;++i){
            labelIndices[i]=labelVector.indexAtLocation(i);
        }
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
            double[] probs = new double[local_numLabels];
            double mass = 0;
            for(int rank = 0;rank<local_numLabels;++rank){
                int label = labelIndices[rank];
                if(!evaluate){
                    probs[rank] = (typeTopicCounts[type][label]+beta)/(tokensPerTopic[label]+betaSum);
                }else{
                    probs[rank] = (typeTopicCounts[type][label]+beta)/(tokensPerTopic[label]+betaSum)*(typeTopicDCounts[label][dtopic]+testgammaLabel);
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
            int newLabel = labelIndices[newTopic];
            //increase counts
            if(!evaluate){
                tokensPerTopic[newLabel]++;
                typeTopicCounts[type][newLabel]++;
                typeTopicDCounts[newLabel][dtopic]++;
            }
            oneDocTopics[position] = newLabel;
        }
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


    public void printTopWords(String filename,int numWords, int threshold){
        try{
            BufferedWriter bw = new BufferedWriter(new FileWriter(filename));
            for(int topic = 0;topic<numLabels;++topic){
                IDSorter[] topWords = topWords(topic);
                bw.write("Label: "+this.topicAlphabet.lookupObject(topic)+"\n");
                for(int word = 0;word<topWords.length;++word){
                    if(word<numWords&&topWords[word].getWeight()>threshold){
                        String wordString = (String) alphabet.lookupObject(topWords[word].getID());
                        bw.write(wordString + " ");      
                    }
                }
                bw.write("\n");
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
