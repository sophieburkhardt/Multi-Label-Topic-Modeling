package org.kramerlab.classifiers;

import java.util.ArrayList;
import java.util.Arrays;

import cc.mallet.types.InstanceList;
import cc.mallet.types.Instance;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.LabelSequence;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.topics.TopicAssignment;
import cc.mallet.util.Randoms;

public class DependencyLDA12 extends DependencyLDAOriginal{


    public DependencyLDA12(LabelAlphabet la, double alphasum, double beta, Randoms random, double eta, double beta_c,double gamma,int T,double testalpha,double testeta, double testgamma,int innerIterations){
        super(la,alphasum,beta,random,eta,beta_c,gamma,T,testalpha,testeta,testgamma,innerIterations);
    }


    @Override
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
                LabelSequence dtopicSequence = new LabelSequence(this.topicDAlphabet,new int[tokens.size()]);
                int[] topics = topicSequence.getFeatures();
                int[] dtopics = dtopicSequence.getFeatures();

                for (int position = 0; position < tokens.size(); position++) {
                    //the topic can only come from the labels
                    int chosenrank = random.nextInt(local_numLabels);
                    int topic = ((FeatureVector)instance.getTarget()).indexAtLocation(chosenrank);
                    topics[position] = topic;
                    tokensPerTopic[topic]++;
                    int dtopic = random.nextInt(this.T);
                    dtopics[position] = dtopic;
                    tokensPerDTopic[dtopic]++;
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
            FeatureSequence tokens = (FeatureSequence) data.get(c).topicSequence; 
            c++;
            FeatureSequence topicSequence = (FeatureSequence) document.topicSequence;
            int[] topics = topicSequence.getFeatures();
            int[] labels = tokens.getFeatures();
            for (int position = 0; position < topicSequence.size(); position++) {
                int topic = topics[position];
                if (topic == UNASSIGNED_TOPIC) { continue; }
                tokensPerDTopic[topic]++;
                int type = labels[position];
                int[] currentTypeTopicCounts = topicDTopicCounts[type];
                currentTypeTopicCounts[topic]++;
            }
    
        }
    }



    protected void sampleTopicsForOneDoc (FeatureSequence tokenSequence,
                                          FeatureSequence topicSequence,FeatureVector labels,FeatureSequence dtopicSequence){
        FeatureSequence tokens = tokenSequence;
        int docLength = tokens.getLength();
        int[] topics = topicSequence.getFeatures();
        int[] dtopics = dtopicSequence.getFeatures();
        int local_numLabels = labels.numLocations();
        int[] localTopicCounts = new int[numLabels];
        
        int type;
        int[] currentTypeTopicCounts;
        int[] currentTopicDTopicCounts;
        for (int position = 0; position < docLength; position++) {
            localTopicCounts[topics[position]]++;
        }

        int i;
        double score;

        int oldTopic, newTopic, dtopic;


        //at this point, first sample labels from topics for the alpha array
        this.getAlphaArray(topicSequence,dtopicSequence,labels);
        double[] docSum = new double[numLabels];

        //  Iterate over the positions (words) in the document                                                        
        for (int position = 0; position < docLength; position++) {
            type = tokens.getIndexAtPosition(position);
            dtopic = dtopics[position];
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
            this.decreaseTopicDTopicCounts(oldTopic,dtopic);
            this.increaseTopicDTopicCounts(newTopic,dtopic);
        }
    }

    /***/
    @Override
    public double getScore(int currentValue,double beta,int tokensPerTopic,double betaSum,int localTopicCount,double alpha){
        return (currentValue+beta)/(tokensPerTopic+betaSum);
    }

    @Override
    protected int[] sampleLabelsForDocument(FeatureSequence tokenSequence, FeatureSequence topicSequence,FeatureVector labels){
        int[] oneDocTopics = topicSequence.getFeatures();
        int[] currentTypeTopicCounts;
        int type, oldTopic, newTopic;
        double topicWeightsSum;
        int docLength = tokenSequence.getLength();
        int[] localTopicCounts = new int[this.T];
        // populate topic counts
        for (int position = 0; position < docLength; position++) {
            localTopicCounts[oneDocTopics[position]]++;
        }
        double score, sum;
        double[] topicTermScores = new double[T];
        // Iterate over the positions (words) in the document
        for (int position = 0; position < docLength; position++) {
            type = tokenSequence.getIndexAtPosition(position);
            oldTopic = oneDocTopics[position];
            // Grab the relevant row from our two-dimensional array
            currentTypeTopicCounts = topicDTopicCounts[type];
            // Remove this token from all counts.
            localTopicCounts[oldTopic]--;
            currentTypeTopicCounts[oldTopic]--;
            tokensPerDTopic[oldTopic]--;
            // Now calculate and add up the scores for each topic for this word
            sum = 0.0;
            for (int topic = 0; topic < this.T; topic++) {
                int current_count = currentTypeTopicCounts[topic];
                score =
                    (gamma + localTopicCounts[topic]) *
                    ((beta_c + current_count) / 
                     (beta_c_sum + tokensPerDTopic[topic]));
                sum += score;
                topicTermScores[topic] = score;
            }
            // Choose a random point between 0 and the sum of all topic scores
            double sample = random.nextUniform() * sum;
            newTopic = -1;
            while (sample > 0.0) {
                newTopic++;
                sample -= topicTermScores[newTopic];
            }
            // Make sure we actually sampled a topic
            if (newTopic == -1) {
                throw new IllegalStateException ("DependencyLDA12: New topic not sampled.");
            }
            // Put that new topic into the counts
            oneDocTopics[position] = newTopic;
            localTopicCounts[newTopic]++;
            currentTypeTopicCounts[newTopic]++;
            tokensPerDTopic[newTopic]++;
        }
        return oneDocTopics;
    }

}
