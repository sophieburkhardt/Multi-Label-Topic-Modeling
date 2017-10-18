/**
 * AliasInferencer.java
 * 
 * Copyright (C) 2017 Sophie Burkhardt
 *
 * This file is part of Multi-Label-Topic-Modeling.
 * 
 * Multi-Label-Topic-Modeling is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * Multi-Label-Topic-Modeling is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301 USA
 *
 */

package org.kramerlab.inferencers;

import org.kramerlab.interfaces.*;

import cc.mallet.types.Instance;
import java.util.Stack;
import cc.mallet.util.Randoms;
import cc.mallet.types.FeatureSequence;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.Iterator;
import java.util.EmptyStackException;

public class AliasInferencer implements Inferencer{

    int numTopics;
    int numWords;
    int[][] typeTopicCounts;
    int[] tokensPerTopic;
    int k;//number of samples to store
    Stack[] s_w; //stored samples for each word
    double[] qw_norm;//normalization constants per word
    double[][] qw;
    double[] pdw_norm;
    double[][] pdw;
    double[] alpha;
    double beta;
    double betaSum;
    Randoms random;
    int rejections;


    public AliasInferencer(int[][] typeTopicCounts, int[] tokensPerTopic, double[] alpha, double beta, int k){
        this.typeTopicCounts=typeTopicCounts;
        this.tokensPerTopic = tokensPerTopic;
        this.k=k;
        this.numTopics = tokensPerTopic.length;
        this.numWords = typeTopicCounts.length;
        this.s_w=new Stack[numWords];
        this.qw_norm = new double[numWords];
        this.qw = new double[numWords][];
        this.pdw_norm = new double[numWords];
        this.pdw = new double[numWords][numTopics];
        this.alpha = alpha;
        this.beta = beta;
        this.betaSum = beta * numWords;
        this.random = new Randoms();
            
    }


    public AliasInferencer(int[][] typeTopicCounts, int[] tokensPerTopic, double[] alpha, double beta){
        this(typeTopicCounts,tokensPerTopic,alpha,beta,500);
    }


    public double[][] generateAlias(double[] p){
        double epsilon = 0.000001;
        double[][] a = new double[numTopics][3];
        double average = 1/(double) numTopics;
        Stack<Integer> l_i = new Stack<Integer>();
        Stack<Double> l_p = new Stack<Double>();
        Stack<Integer> h_i = new Stack<Integer>();
        Stack<Double> h_p = new Stack<Double>();
        for(int i = 0;i<numTopics;++i){
            if(p[i]>0+epsilon){
                if(p[i]<=average){
                    l_i.push(i);
                    l_p.push(p[i]);
                }else{

                    h_i.push(i);
                    h_p.push(p[i]);
                }
            }
        }
        int counter = 0;
        
        while(!l_i.empty()||!h_i.empty()){

            if(h_i.empty()){

                int l_ind = l_i.pop();
                double l_prob = l_p.pop();
                

                a[counter][0]=l_ind;
                a[counter][1]=l_ind;
                a[counter][2]=l_prob;
                counter++;
            }else if(l_i.empty()){
                int h_ind;
                double h_prob;
               
                h_ind = h_i.pop();
                h_prob = h_p.pop();
                a[counter][0]=h_ind;
                a[counter][1]=h_ind;
                a[counter][2]=h_prob;
                double val = h_prob-average;

                if(val>average+epsilon){
                    h_i.push(h_ind);
                    h_p.push(val);
                }else{
                    l_i.push(h_ind);
                    l_p.push(val);
                }

                
            }else{
		int l_ind=-1;
                double l_prob=-1;
                int h_ind;
                double h_prob;
                            
                h_ind = h_i.pop();
                h_prob = h_p.pop();
                

		l_ind = l_i.pop();
		l_prob = l_p.pop();

                a[counter][0]=l_ind;
                a[counter][1]=h_ind;
                a[counter][2]=l_prob;
                
                counter++;
                double val = h_prob-(average-l_prob);
                
                if(val>average+epsilon){
                    h_i.push(h_ind);
                    h_p.push(val);
                }else{
                    l_i.push(h_ind);
                    l_p.push(val);
                }
            }
        }
        return a;
    }

    public int sampleAlias(double[][] a, double sample){
        int bin = (int)(sample*a.length);
        double[] entry = a[bin];
        double coinflip = random.nextUniform();
        if(coinflip<numTopics*entry[2]){
            return (int)entry[0];
        }else{
            return (int)entry[1];
        }
    }
    /**

     */
    public boolean stationaryMetropolisHastings(int[] local_tokensPerTopic, int s, int t, int type){
        //compute pi
        double pi = 0;
        pi=(local_tokensPerTopic[t]+alpha[t])/(double)(local_tokensPerTopic[s]+alpha[s]);

        pi*=(typeTopicCounts[type][t]+beta)/(double)(typeTopicCounts[type][s]+beta);
        pi*=(tokensPerTopic[s]+betaSum)/(double)(tokensPerTopic[t]+betaSum);
        pi*=(pdw_norm[type]*pdw[type][s]+qw_norm[type]*qw[type][s])/(double)(pdw_norm[type]*pdw[type][t]+qw_norm[type]*qw[type][t]);
        //check whether to accept or not
        double sample = random.nextUniform();
        return (sample<=pi);
    }


    
    public double[] getSampledDistribution(Instance instance, int numIterations, int thinning, int burnin){

        HashSet<Integer> uniqueTopics=new HashSet<Integer>();
        double[] distribution = new double[numTopics];
        FeatureSequence tokens = (FeatureSequence) instance.getData();
        int docLength = tokens.size();
        int[] topics = new int[docLength];
        double sum = 0;
        //initialize topics
        int[] local_tokensPerTopic = new int[numTopics];
        //do some kind of intelligent initialization
        for(int position = 0; position < docLength; ++position){
            int type = tokens.getIndexAtPosition(position);
            int maxTopic=-1;
            int maxCount=0;
            for(int t = 0;t<numTopics;++t){
                if(typeTopicCounts[type][t]>maxCount){
                    maxTopic=t;
                    maxCount = typeTopicCounts[type][t];
                }
            }
            if(maxCount>0){
                topics[position]=maxTopic;
                uniqueTopics.add(maxTopic);
                }else{
                topics[position] = random.nextInt(numTopics);
                uniqueTopics.add(topics[position]);
                }
                local_tokensPerTopic[topics[position]]++;
        }

        for(int i=0;i<numIterations;++i){


            for(int position = 0; position < docLength; ++position){
                int type = tokens.getIndexAtPosition(position);
                int oldTopic = topics[position];
                //decrement counts
                local_tokensPerTopic[oldTopic]--;
                if(local_tokensPerTopic[oldTopic]==0){
                    uniqueTopics.remove(oldTopic);
                }
                computeP(type, local_tokensPerTopic, uniqueTopics);
                
                if(s_w[type]==null){
                    s_w[type]=new Stack();
                }
                //check if another sample is available
                //if no, generate new Alias for current word, generate samples, then use MH to accept or reject
                if(s_w[type].empty()){

                    createMoreSamples(k, type);

                }
                boolean accept=false;
                int newTopic=-1;

                while(!accept){ 
                    rejections++;
                    if(s_w[type].empty()){
                        System.out.println("create more samples for type "+type);
                        createMoreSamples(k,type);
                    }
                    double randomNumber = random.nextUniform();
                    
                    double psum = pdw_norm[type]/(pdw_norm[type]+qw_norm[type]);    
                    if(randomNumber<psum){
                        //first bucket
                        randomNumber/=pdw_norm[type];
                        randomNumber*=(pdw_norm[type]+qw_norm[type]);
                        int counter = -1;
                        Iterator<Integer> uniqueTopicsIterator = uniqueTopics.iterator();
                        int currentTopic=-1;
                        while(randomNumber>0){
                            currentTopic = uniqueTopicsIterator.next();
                            if(currentTopic != oldTopic || local_tokensPerTopic[currentTopic] > 0){
                            
                                counter++;
                                randomNumber-=pdw[type][currentTopic];
                            }
                        }
                        newTopic = currentTopic;
                    }else{
                        //second bucket
                        newTopic = (Integer)s_w[type].pop();
                    }
                    //since the topic distributions dont change, we can always accept
                    //accept = stationaryMetropolisHastings(local_tokensPerTopic, topics[position],(Integer)newTopic,type);
                    accept = true;
                }
                rejections--; //you have to go through loop once
                topics[position] = newTopic;
                uniqueTopics.add(newTopic);
                //increase counts
                
                local_tokensPerTopic[newTopic]++;        
                                
                if(i >= burnin && i % thinning == 0){
                    distribution[newTopic]++;
                    sum+=1;
                }

            }        
        }
        double intermediateSum = 0;
        for(int r = 0;r<numTopics;++r){
            distribution[r]+=sum*alpha[r];
            intermediateSum+=sum*alpha[r];
        }
        sum+=intermediateSum;
        //normalize
        for(int i=0;i<distribution.length;++i){
            distribution[i]/=sum;
        }
        return distribution;
    }


    public void getSampledTopics(FeatureSequence tokenSequence,FeatureSequence topicSequence){
        PForDoc pcalc = new PForDoc();
        HashSet<Integer> uniqueTopics=new HashSet<Integer>();
        int docLength = tokenSequence.getLength();
        int[] topics = topicSequence.getFeatures();
        double sum = 0;
        
        int[] local_tokensPerTopic = new int[numTopics];
        for(int position = 0; position < docLength; ++position){
            int type = tokenSequence.getIndexAtPosition(position);
            local_tokensPerTopic[topics[position]]++;
            uniqueTopics.add(topics[position]);
        }

        for(int position = 0; position < docLength; ++position){
            int type = tokenSequence.getIndexAtPosition(position);
            int oldTopic = topics[position];
            //decrement counts
            local_tokensPerTopic[oldTopic]--;
            if(local_tokensPerTopic[oldTopic]==0){
                uniqueTopics.remove(oldTopic);
            }
            pcalc.computeP(type, local_tokensPerTopic, uniqueTopics);
                
            if(s_w[type]==null){
                s_w[type]=new Stack();
            }
            //check if another sample is available
            //if no, generate new Alias for current word, generate samples, then use MH to accept or reject
            if(s_w[type].empty()){

                createMoreSamples(k, type);

            }
            boolean accept=false;
            int newTopic=-1;

            while(!accept){ 

                if(s_w[type].size()<1){
                    //System.out.println("create more samples for type "+type);
                    createMoreSamples(k,type);
                }
                double randomNumber = random.nextUniform();
                    
                double psum = pcalc.pdw_norm_doc[type]/(pcalc.pdw_norm_doc[type]+qw_norm[type]);    
                if(randomNumber<psum){
                    //first bucket
                    randomNumber/=pcalc.pdw_norm_doc[type];
                    randomNumber*=(pcalc.pdw_norm_doc[type]+qw_norm[type]);
                    int counter = -1;
                    Iterator<Integer> uniqueTopicsIterator = uniqueTopics.iterator();
                    int currentTopic=-1;
                    while(randomNumber>0){
                        currentTopic = uniqueTopicsIterator.next();
                        if(currentTopic != oldTopic || local_tokensPerTopic[currentTopic] > 0){
                            
                            counter++;
                            randomNumber-=pcalc.pdw_doc[type][currentTopic];
                        }
                    }
                    newTopic = currentTopic;
                }else{
                    //second bucket
                    newTopic = (Integer)s_w[type].pop();
                }
                //since the topic distributions dont change, we can always accept
                //accept = stationaryMetropolisHastings(local_tokensPerTopic, topics[position],(Integer)newTopic,type);
                accept = true;
            }
            topics[position] = newTopic;
            uniqueTopics.add(newTopic);
            //increase counts
                
            local_tokensPerTopic[newTopic]++;        
                                
                
        }        
    }



    public void createMoreSamples(int numSamples, int type){
        if(qw[type]==null){
            qw[type]=new double[numTopics];
            //Q_w
            for(int t = 0; t<numTopics;++t){
                qw[type][t]=alpha[t]*(typeTopicCounts[type][t]+beta)/(tokensPerTopic[t]+betaSum);
                qw_norm[type]+=qw[type][t];
            }
            //normalize q_w
            for(int t = 0; t<numTopics; ++t){
                qw[type][t]/=qw_norm[type];
            }
        }
        //generate alias table
        double[][] a = generateAlias(qw[type]);
        //draw k samples from alias table
        double[] counts = new double[numTopics];
        for(int i=0;i<numSamples;++i){
            double sample = random.nextUniform();
            int newTopic = sampleAlias(a,sample);
            s_w[type].push(newTopic);
            counts[newTopic]+=1;
        }
        
    }

    public void computeP(int type, int[] local_tokensPerTopic, HashSet<Integer> topics){

            pdw[type]=new double[numTopics];
            pdw_norm[type]=0;
            //P_dw
            for(Integer topic: topics){
                pdw[type][topic] = local_tokensPerTopic[topic] * (typeTopicCounts[type][topic] + beta) / (double)(tokensPerTopic[topic] + betaSum);
                pdw_norm[type] += pdw[type][topic];
            }
            if(pdw_norm[type]<0){
                System.out.println("problem at initialization");
            }
            //normalize p_dw 
            for(Integer topic: topics){
                pdw[type][topic] /= pdw_norm[type];
            }

    }




    public class PForDoc{

        double[][] pdw_doc;
        double[] pdw_norm_doc;

        public void computeP(int type, int[] local_tokensPerTopic, HashSet<Integer> topics){
            if(pdw_doc==null){
                pdw_doc = new double[numWords][];
                pdw_norm_doc = new double[numWords];
            }
            pdw_doc[type]=new double[numTopics];
            pdw_norm_doc[type]=0;
            //P_dw
            for(Integer topic: topics){
                pdw_doc[type][topic] = local_tokensPerTopic[topic] * (typeTopicCounts[type][topic] + beta) / (double)(tokensPerTopic[topic] + betaSum);
                pdw_norm_doc[type] += pdw_doc[type][topic];
            }
            if(pdw_norm_doc[type]<0){
                System.out.println("problem at initialization");
            }
            //normalize p_dw 
            for(Integer topic: topics){
                pdw_doc[type][topic] /= pdw_norm_doc[type];
            }

        }


    }


    //Just get the topic for one word
    public int getSampledTopic(int type,int[] local_tokensPerTopic,int oldTopic,int[] topics){
        PForDoc pcalc = new PForDoc();
        HashSet<Integer> uniqueTopics=new HashSet<Integer>();
        int docLength = topics.length;
        for(int position = 0; position < docLength; ++position){
            uniqueTopics.add(topics[position]);
        }
        //decrement counts
        local_tokensPerTopic[oldTopic]--;
        if(local_tokensPerTopic[oldTopic]==0){
            uniqueTopics.remove(oldTopic);
        }
        pcalc.computeP(type, local_tokensPerTopic, uniqueTopics);
                
        if(s_w[type]==null){
            s_w[type]=new Stack();
        }
        //check if another sample is available
        //if no, generate new Alias for current word, generate samples, then use MH to accept or reject
        if(s_w[type].size()<1){

            createMoreSamples(k, type);

        }
        int newTopic=-1;

        rejections++;
        if(s_w[type].size()<1){
            System.out.println("create more samples for type "+type);
            createMoreSamples(k,type);
        }
        double randomNumber = random.nextUniform();
                    
        double psum = pcalc.pdw_norm_doc[type]/(pcalc.pdw_norm_doc[type]+qw_norm[type]);    
        if(randomNumber<psum){
            //first bucket
            randomNumber/=pcalc.pdw_norm_doc[type];
            randomNumber*=(pcalc.pdw_norm_doc[type]+qw_norm[type]);
            int counter = -1;
            Iterator<Integer> uniqueTopicsIterator = uniqueTopics.iterator();
            int currentTopic=-1;
            while(randomNumber>0){
                currentTopic = uniqueTopicsIterator.next();
                if(currentTopic != oldTopic || local_tokensPerTopic[currentTopic] > 0){
                            
                    counter++;
                    randomNumber-=pcalc.pdw_doc[type][currentTopic];
                }
            }
            newTopic = currentTopic;
        }else{
            //second bucket
            newTopic = (Integer)s_w[type].pop();
        }
        //since the topic distributions dont change, we can always accept

        return newTopic;
    }


}
