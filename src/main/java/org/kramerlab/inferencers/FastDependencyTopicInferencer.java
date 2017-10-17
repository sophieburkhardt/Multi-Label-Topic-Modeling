package org.kramerlab.inferencers;

import org.kramerlab.interfaces.Inferencer;
import cc.mallet.types.Instance;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.Alphabet;
import cc.mallet.util.Randoms;

public class FastDependencyTopicInferencer implements Inferencer{

    protected Alphabet topicalphabet; //label alphabet
    protected Alphabet topicDalphabet; //topic alphabet
    protected double[] alpha_array;

    protected int[][] typeTopicCounts;
    protected int[][] topicDTopicCounts;
    protected int[] tokensPerDTopic;
    protected int[] tokensPerTopic;
    protected int[] newTokensPerDTopic;
    protected int numDTopics;
    protected double gamma;
    protected double gammasum;
    protected double beta_c;
    protected double beta_c_sum;
    protected double eta;
    protected double alpha;
    protected double beta;
    protected double betaSum;

    Randoms random;
    int numLabels;
    int numTypes;
    int[] labelCounts;

    AliasInferencer aliasInferencer;

    public FastDependencyTopicInferencer(AliasInferencer inferencer,int[][] typeTopicCounts, int[] tokensPerTopic,
                                     Alphabet topicalphabet, double[] alpha, double beta, double betaSum,int[][] topicDTopicCounts, int[] tokensPerDTopic,int numDTopics,double gamma,double beta_c,double justalpha,Alphabet topicDalphabet,double eta){
        this.typeTopicCounts = typeTopicCounts;
        this.tokensPerTopic = tokensPerTopic;
        this.numTypes = typeTopicCounts.length;
        this.alpha_array = alpha;
        this.alpha = justalpha;
        this.beta = beta;
        this.betaSum = betaSum;
        this.aliasInferencer = inferencer;
        this.topicDTopicCounts=topicDTopicCounts;
        this.tokensPerDTopic=tokensPerDTopic;
        this.numDTopics=numDTopics;
        this.gamma=gamma;
        this.gammasum=numDTopics*gamma;
        this.beta_c=beta_c;
        this.beta_c_sum=beta_c*numDTopics;
        this.alpha_array=alpha;
        this.alpha=justalpha;
        this.topicalphabet=topicalphabet;
        this.topicDalphabet=topicDalphabet;
        this.numLabels=topicalphabet.size();
        this.eta=eta;
        this.random = new Randoms();    
        this.aliasInferencer = new AliasInferencer(topicDTopicCounts,tokensPerDTopic,alpha_array,gamma);
    }





    /** 
     *  Use Gibbs sampling to infer a topic distribution.
     *  Topics are initialized to the (or a) most probable topic
     *   for each token. Using zero iterations returns exactly this
     *   initial topic distribution.<p/>
     *  This code does not adjust type-topic counts: P(w|t) is clamped.
     */
    public double[] getSampledDistribution(Instance instance, int numIterations,
                                           int thinning, int burnIn) {
        //System.out.println("get");
        FeatureSequence tokens = (FeatureSequence) instance.getData();
        int docLength = tokens.getLength();
        int[] topics = new int[docLength];
        int[] dtopics = new int[docLength];

        int[] localTopicCounts = new int[numLabels];
        
        int type;
        int[] currentTypeTopicCounts;
        int[] currentTopicDTopicCounts;

        // Initialize all positions to a random topic

        for (int position = 0; position < docLength; position++) {
            type = tokens.getIndexAtPosition(position);

            // Ignore out of vocabulary terms
            if (type < numTypes && typeTopicCounts[type].length != 0) { 

                currentTypeTopicCounts = typeTopicCounts[type];

                topics[position] = random.nextInt(numLabels);
                dtopics[position] = random.nextInt(numDTopics);

                localTopicCounts[topics[position]]++;
            }
        }
        int i;
        double score;

        int oldTopic, newTopic;

        double[] result = new double[numLabels];
        double sum = 0.0;

        for (int iteration = 1; iteration <= numIterations; iteration++) {
            //at this point, first sample labels from topics for the alpha array
                FeatureSequence f1 = new FeatureSequence(topicalphabet,topics);
                FeatureSequence f2 = new FeatureSequence(topicDalphabet,dtopics);
                //System.out.println("hier");
                this.getAlphaArray(f1,f2);
                double[] docSum = new double[numLabels];

            //  Iterate over the positions (words) in the document                                                        
            for (int position = 0; position < docLength; position++) {
                
                
                type = tokens.getIndexAtPosition(position);
                
                // ignore out-of-vocabulary terms
                if (type >= numTypes || typeTopicCounts[type].length == 0) { continue; }

                oldTopic = topics[position];
                currentTypeTopicCounts = typeTopicCounts[type];
                localTopicCounts[oldTopic]--;
                double[] topicTermScores = new double[numLabels];
                double topicTermMass = 0;
                for(int currentTopic=0;currentTopic<numLabels;++currentTopic){
                    double currentValue = currentTypeTopicCounts[currentTopic];
                    
                    score = (currentValue+beta)/(tokensPerTopic[currentTopic]+betaSum)*(localTopicCounts[currentTopic]+alpha_array[currentTopic]);
                    topicTermMass += score;
                    topicTermScores[currentTopic] = score;
                    
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
                
                

                topics[position] = newTopic;

                
                if (iteration > burnIn &&
                    (iteration - burnIn) % thinning == 0) {

                    // Save a sample
                    for (int topic=0; topic < numLabels; topic++) {
                        int topiccount = currentTypeTopicCounts[topic];
                        int topicindex = topic;
                        double val = ((topiccount+beta)/(tokensPerTopic[topic]+betaSum))*(alpha_array[topic] + localTopicCounts[topic]);
                        docSum[topic]+=val;
                    }

                }



                localTopicCounts[newTopic]++;


            }

            for(int topic = 0;topic<numLabels;++topic){
                if(docLength>0){
                    docSum[topic]/=docLength;
                    result[topic] += docSum[topic];
                    sum += docSum[topic];
                }
            }

        }

        if (sum == 0.0) {
            // Save at least one sample
            for (int topic=0; topic < numLabels; topic++) {
                result[topic] = (alpha_array[topic] + localTopicCounts[topic]);
                sum += result[topic];
            }
        }

        // Normalize
        if(sum>0){
            for (int topic=0; topic < numLabels; topic++) {
                result[topic] /= sum;
            }
        }
        return result;
    }


     public double[] getAlphaArray(FeatureSequence tokenSequence, FeatureSequence topicSequence){
        this.aliasInferencer.getSampledTopics(tokenSequence,topicSequence);
        //this.sampleLabelsForDocument(tokenSequence,topicSequence);
        //initialize alpha array with alpha
        double[] new_alpha_array = new double[this.numLabels];
        for(int i=0;i<this.numLabels;++i){
            new_alpha_array[i]=alpha;
        }

        int[] topics = topicSequence.getFeatures();
        int docLen=topicSequence.getLength();
        int[] topics_counts = new int[this.numDTopics];
        for(int i=0;i<docLen;++i){
            topics_counts[topics[i]]++;
        }

        double alpha_array_sum = 0;
        //iterate over phi'
        for(int i=0;i<this.numLabels;++i){
            int[] currentCounts = this.topicDTopicCounts[i];
            for(int topic = 0;topic<numDTopics;++topic){
                int count = currentCounts[topic];
                new_alpha_array[i]+=this.eta*(topics_counts[topic]+gamma)/(double)(docLen+gammasum)*(count+this.beta_c)/(this.tokensPerDTopic[topic]+this.beta_c_sum);
            }
            alpha_array_sum+=new_alpha_array[i];
        }
        //scale alpha according to document length
        if(alpha_array_sum>0&&docLen>0){
            for(int i=0;i<this.numLabels;++i){
                new_alpha_array[i]*=docLen;
                new_alpha_array[i]/=alpha_array_sum;
            }
        }
        this.alpha_array=new_alpha_array;
        return new_alpha_array;
    }

    
    
    /**analog for sampleTopicsForOneDoc
       tokenSequence: Sequence with the assigned labels for this document
       topicSequence: Sequence with assigned topics for this document*/
    protected int[] sampleLabelsForDocument(FeatureSequence tokenSequence, FeatureSequence topicSequence){
        int[] oneDocTopics = topicSequence.getFeatures();
        int[] currentTypeTopicCounts;
        int type, oldTopic, newTopic;
        double topicWeightsSum;
        int docLength = tokenSequence.getLength();
        int[] localTopicCounts = new int[this.numDTopics];
        // populate topic counts
        for (int position = 0; position < docLength; position++) {
            localTopicCounts[oneDocTopics[position]]++;
        }
        double score, sum;
        double[] topicTermScores = new double[numDTopics];
        // Iterate over the positions (words) in the document
        for (int position = 0; position < docLength; position++) {
            type = tokenSequence.getIndexAtPosition(position);
            oldTopic = oneDocTopics[position];
            // Grab the relevant row from our two-dimensional array
            currentTypeTopicCounts = topicDTopicCounts[type];
            // Remove this token from all counts.
            localTopicCounts[oldTopic]--;
            // Now calculate and add up the scores for each topic for this word
            sum = 0.0;
            for (int topic = 0; topic < numDTopics; topic++) {
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
                throw new IllegalStateException ("FastDependencyTopicInferencer: New topic not sampled.");
            }
            // Put that new topic into the counts
            oneDocTopics[position] = newTopic;
            localTopicCounts[newTopic]++;
        }
        return oneDocTopics;
    }


    
}
