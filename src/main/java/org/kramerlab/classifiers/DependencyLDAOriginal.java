package org.kramerlab.classifiers;

import org.kramerlab.inferencers.*;
import org.kramerlab.interfaces.*;

import cc.mallet.util.Randoms;
import cc.mallet.topics.TopicAssignment;
//import cc.mallet.topics.SimpleLDA;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.LabelSequence;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.IDSorter;

import java.util.*;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;



public class DependencyLDAOriginal implements TopicModel{

    double gamma;
    protected ArrayList<TopicAssignment> data; //the label assignments 
    protected ArrayList<TopicAssignment> data_dep; //the assignments of the next hierarchy level
    protected final int T; //the number of topic distributions
    final int UNASSIGNED_TOPIC = -1;
    int numLabels;
    int numTypes;
    // the alphabet for the topics
    protected Alphabet alphabet;//words
    protected LabelAlphabet topicAlphabet; //labels
    protected LabelAlphabet topicDAlphabet; //topics
    double beta;
    double betaSum;
    double[] alpha_array;
    double alpha;
    double alphaSum;
    protected double gammasum;
    protected double beta_c_sum;
    public double eta;
    protected double beta_c;
    protected double testeta;
    protected double testgamma;
    public double testalpha;
    // Statistics needed for sampling.
    int[][] typeTopicCounts;
    int[] tokensPerTopic;
    protected int[][] topicDTopicCounts; // indexed by <feature index, topic index>
    protected int[] tokensPerDTopic; // indexed by <topic index>

    protected int[] labelCounts;

    protected int[] oneDocDTopicCounts;

    protected int innerIterations;

    Randoms random;

    AliasInferencer aliasInferencer;

    public DependencyLDAOriginal(){T=0;}

    public DependencyLDAOriginal(LabelAlphabet la, double alphasum, double beta, Randoms random, double eta, double beta_c,double gamma,int T,double testalpha,double testeta, double testgamma,int innerIterations){
        this.random = new Randoms();
        this.beta = beta;
        this.T=T;
        this.topicAlphabet = la;
        this.topicDAlphabet = newLabelAlphabet(T);
        this.gammasum = gamma;
        this.gamma=gammasum/this.T;
        this.beta_c_sum=T*beta_c;
        this.tokensPerDTopic = new int[T];
        this.oneDocDTopicCounts=new int[T];
        this.eta=eta;
        this.beta_c=beta_c;
        this.testeta=testeta;
        this.testgamma=testgamma/this.T;
        this.numLabels = la.size();
        this.testalpha = testalpha/numLabels;
        this.alpha_array = new double[numLabels];
        this.innerIterations=innerIterations;
    }

    public double getBeta(){
        return this.beta;
    }

    public double getAlphaSum(){
        return this.alphaSum;
    }
    
    public void addInstances(InstanceList training){
        this.data = new ArrayList<TopicAssignment>();
        this.data_dep = new ArrayList<TopicAssignment>();
        alphabet = training.getDataAlphabet();
        numTypes = alphabet.size();
        betaSum = beta * numTypes;
        typeTopicCounts = new int[numTypes][this.numLabels];
        tokensPerTopic = new int[this.numLabels];
        topicDTopicCounts = new int[this.numLabels][this.T];
        tokensPerDTopic = new int[this.T];

        int doc=0;
        for (Instance instance : training) {
	    //the number of labels for this document
            int local_numLabels = ((FeatureVector)instance.getTarget()).numLocations();
            //skip instances that have no labels
            if(local_numLabels>0){
                doc++;
                FeatureSequence tokens = (FeatureSequence) instance.getData();
                LabelSequence topicSequence =
                    new LabelSequence(topicAlphabet, new int[ tokens.size() ]);
                LabelSequence dtopicSequence = new LabelSequence(this.topicDAlphabet,new int[local_numLabels]);
                int[] topics = topicSequence.getFeatures();
                int[] dtopics = dtopicSequence.getFeatures();
                for(int l = 0;l<local_numLabels;++l){
                    int dtopic = random.nextInt(this.T);
                    dtopics[l] = dtopic;
                    tokensPerDTopic[dtopic]++;
                }
                for (int position = 0; position < tokens.size(); position++) {
                    //the topic can only come from the labels
                    int chosenrank = random.nextInt(local_numLabels);
                    int topic = ((FeatureVector)instance.getTarget()).indexAtLocation(chosenrank);
                    topics[position] = topic;
                    tokensPerTopic[topic]++;
                }
                TopicAssignment t = new TopicAssignment (instance, topicSequence);
                TopicAssignment dt = new TopicAssignment(instance, dtopicSequence);
                data.add (t);
                data_dep.add(dt);
            }
            else{
                System.out.println("instance with no label ... skipped");
            }
        }
        buildInitialTypeTopicCounts();
        buildInitialDTypeTopicCounts();
    
    }



    public void buildInitialTypeTopicCounts(){
        Arrays.fill(tokensPerTopic, 0);
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



    public void buildInitialDTypeTopicCounts () {
        // Clear the topic totals
        Arrays.fill(tokensPerDTopic, 0);
        // Clear the type/topic counts
        for (int type = 0; type < this.numLabels; type++) {
            int[] topicCounts = topicDTopicCounts[type];
            for(int topic = 0;topic<this.T;++topic){
                topicCounts[topic] = 0;
            }
        }
        int c=0;
        for (TopicAssignment document : data_dep) {
            FeatureSequence tokens = (FeatureSequence) data.get(c).topicSequence; // the labels for each word
            FeatureVector ls = (FeatureVector)document.instance.getTarget();
            c++;
            FeatureSequence topicSequence = (FeatureSequence) document.topicSequence;
            int[] topics = topicSequence.getFeatures();
            for (int position = 0; position < topicSequence.size(); position++) {
                int topic = topics[position];
                if (topic == UNASSIGNED_TOPIC) { continue; }
                tokensPerDTopic[topic]++;
                int type = ls.indexAtLocation(position);
                int[] currentTypeTopicCounts = topicDTopicCounts[type];
                currentTypeTopicCounts[topic]++;
            }
    
        }
    }


    public double[] getAlphaArray(FeatureSequence tokenSequence, FeatureSequence topicSequence,FeatureVector labels){
        //during training we only sample the topics, alpha stays fixed
        this.sampleLabelsForDocument(tokenSequence,topicSequence,labels);
        int local_numLabels = labels.numLocations();
        this.alphaSum = 0;
        for(int i = 0;i<numLabels;++i){
            this.alpha_array[i]=1.0/local_numLabels*eta+this.alpha;
            this.alphaSum+=this.alpha_array[i];
        }
        return alpha_array;
    }




    public Inferencer getInferencer(){
        return new FastDependencyTopicInferencer(aliasInferencer,this.typeTopicCounts,this.tokensPerTopic,
                                             this.topicAlphabet,alpha_array,
                                             this.beta,this.betaSum,this.topicDTopicCounts,
                                             this.tokensPerDTopic,this.T,this.testgamma,
                                             this.beta_c,this.testalpha,
                                             this.topicDAlphabet,this.testeta);
    }


    
    //increases count of label-topic pair by one
    protected void increaseTopicDTopicCounts(int label,int topic){
        int[] currentTypeTopicCounts=this.topicDTopicCounts[label];
        currentTypeTopicCounts[topic]++;    
    }



    //decreases count of label-topic pair by one
    protected void decreaseTopicDTopicCounts(int label,int topic){
        int[] currentTypeTopicCounts=this.topicDTopicCounts[label];
        currentTypeTopicCounts[topic]--;           
    }




    //analog for sampleTopicsForOneDoc
    //tokenSequence: Sequence with the assigned labels for this document
    //topicSequence: Sequence with assigned topics for this document
    //this method should actually be called sampleDTopicsForDocument
    protected int[] sampleLabelsForDocument(FeatureSequence tokenSequence, FeatureSequence topicSequence, FeatureVector labels){
        //System.out.println("new doc");
        int[] oneDocTopics = topicSequence.getFeatures();
        int[] currentTypeTopicCounts;
        int type, oldTopic, newTopic;
        double topicWeightsSum;
        int docLength = topicSequence.getLength();
        int[] localTopicCounts = new int[this.T];
        // populate topic counts
        //System.out.println(numTopics+" "+this.T);
        for (int position = 0; position < docLength; position++) {
            localTopicCounts[oneDocTopics[position]]++;
        }
        double score, sum;
        double[] topicTermScores = new double[this.T];
        // Iterate over the positions (words) in the document
        for (int position = 0; position < docLength; position++) {
            //System.out.println("position "+position);
            //type = tokenSequence.getIndexAtPosition(position);//0516
            type = labels.indexAtLocation(position);
            oldTopic = oneDocTopics[position];
            // Grab the relevant row from our two-dimensional array
            currentTypeTopicCounts = topicDTopicCounts[type];
            // Remove this token from all counts.
            localTopicCounts[oldTopic]--;
            tokensPerDTopic[oldTopic]--;
            assert(tokensPerDTopic[oldTopic] >= 0) : "old Topic " + oldTopic + " below 0";
            this.decreaseTopicDTopicCounts(type,oldTopic);
    
            // Now calculate and add up the scores for each topic for this word
            sum = 0.0;
            // Here's where the math happens! Note that overall performance is
            // dominated by what you do in this loop.
            for (int topic = 0; topic < T; topic++) {
                int current_count = currentTypeTopicCounts[topic];
                //System.out.println(current_count+" "+localTopicCounts[topic]+" "+tokensPerDTopic[topic]);
                score =
                    (gamma + localTopicCounts[topic]) *
                    ((beta_c + current_count) / 
                     (beta_c_sum + tokensPerDTopic[topic]));
                //if(score<0||current_count>tokensPerDTopic[topic])//System.out.println(topic+" "+current_count+" "+localTopicCounts[topic]+" "+tokensPerDTopic[topic]);
                sum += score;
                topicTermScores[topic] = score;
            }
            // Choose a random point between 0 and the sum of all topic scores
            double sample = random.nextUniform() * sum;
            //System.out.println(sample);
            // Figure out which topic contains that point
            newTopic = -1;
            while (sample > 0.0) {
                newTopic++;
                sample -= topicTermScores[newTopic];
            }
            // Make sure we actually sampled a topic
            if (newTopic == -1) {
                throw new IllegalStateException ("DependencyLDAOriginal: New topic not sampled.");
            }
            // Put that new topic into the counts
            this.increaseTopicDTopicCounts(type,newTopic);
            //System.out.println("label "+type+" new Topic: "+newTopic);
            oneDocTopics[position] = newTopic;
            localTopicCounts[newTopic]++;
            tokensPerDTopic[newTopic]++;
            
        }
        return oneDocTopics;
    }



    public void sample (int iterations) throws IOException {

		for (int iteration = 1; iteration <= iterations; iteration++) {
                    long iterationStart = System.currentTimeMillis();        
                    // Loop over every document in the corpus
                    for (int doc = 0; doc < data.size(); doc++) {
                        //System.out.println("doc: "+doc+" of "+data.size());
                        FeatureSequence tokenSequence =
                            (FeatureSequence) data.get(doc).instance.getData();
                        LabelSequence topicSequence =
                            (LabelSequence) data.get(doc).topicSequence;
                        FeatureVector labels =(FeatureVector) data.get(doc).instance.getTarget();
                        FeatureSequence dtopicSequence = data_dep.get(doc).topicSequence;
                        try{
                            sampleTopicsForOneDoc (tokenSequence, topicSequence, labels, dtopicSequence);
                        }catch(Exception e){
                            e.printStackTrace();
                        }
                    }		
                    long elapsedMillis = System.currentTimeMillis() - iterationStart;
		}
	}


    protected void sampleTopicsForOneDoc (FeatureSequence tokenSequence,
                                          FeatureSequence topicSequence,FeatureVector labels,FeatureSequence dtopicSequence){
        FeatureSequence tokens = tokenSequence;
        int docLength = tokens.getLength();
        int[] topics = topicSequence.getFeatures();
        int[] dtopics = dtopicSequence.getFeatures();;
        int local_numLabels = labels.numLocations();
        int[] localTopicCounts = new int[numLabels];
        for (int position = 0; position < docLength; position++) {
            localTopicCounts[topics[position]]++;
        }
    
        int type;
        int[] currentTypeTopicCounts;
        int[] currentTopicDTopicCounts;


        int i;
        double score;

        int oldTopic, newTopic;


        //at this point, first sample labels from topics for the alpha array
        this.getAlphaArray(topicSequence,dtopicSequence,labels);
        double[] docSum = new double[numLabels];

        //  Iterate over the positions (words) in the document                                                        
        for (int position = 0; position < docLength; position++) {
                
                
            type = tokens.getIndexAtPosition(position);
            
            // ignore out-of-vocabulary terms
            if (type >= numTypes || typeTopicCounts[type].length == 0) { continue; }

            oldTopic = topics[position];
            currentTypeTopicCounts = typeTopicCounts[type];
            localTopicCounts[oldTopic]--;
            typeTopicCounts[type][oldTopic]--;
            tokensPerTopic[oldTopic]--;
        
            double[] topicTermScores = new double[local_numLabels];
            double topicTermMass = 0;
            for(int rank=0;rank<local_numLabels;++rank){
                int currentTopic = labels.indexAtLocation(rank);
                int currentValue = currentTypeTopicCounts[currentTopic];
                    
                score = getScore(currentValue,beta,tokensPerTopic[currentTopic],betaSum,localTopicCounts[currentTopic],alpha_array[currentTopic]);

                topicTermMass += score;
                topicTermScores[rank] = score;
                    
            }

            double sample = random.nextUniform() * topicTermMass;
            double origSample = sample;

            //  Make sure it actually gets set                                                                        
            newTopic = -1;

            i = -1;
            while (sample > 0) {
                i++;
                sample -= topicTermScores[i];
            }

            newTopic = i;
            newTopic = labels.indexAtLocation(newTopic);
            //increase counts
            topics[position] = newTopic;
            localTopicCounts[newTopic]++;
            typeTopicCounts[type][newTopic]++;
            tokensPerTopic[newTopic]++;
        }
    }

    public double getScore(int currentValue,double beta,int tokensPerTopic,double betaSum,int localTopicCount,double alpha){
        return (currentValue+beta)/(tokensPerTopic+betaSum)*(localTopicCount+alpha);
    }


    //returns top words for specific topic
    public IDSorter[] topWords(int topic) {
	IDSorter[] sortedWords = new IDSorter[numTypes];
	for (int type = 0; type < numTypes; type++) {
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


    public static LabelAlphabet newLabelAlphabet (int numTopics) {
        LabelAlphabet ret = new LabelAlphabet();
        for (int i = 0; i < numTopics; i++)
            ret.lookupIndex("topic"+i);
        return ret;
    }

}
