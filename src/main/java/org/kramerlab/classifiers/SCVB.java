/**
 * SCVB.java
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

package org.kramerlab.classifiers;


import org.kramerlab.interfaces.*;

import cc.mallet.types.InstanceList;
import cc.mallet.types.Instance;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Alphabet;

import java.util.Arrays;
import java.io.FileWriter;
import java.io.IOException;
import java.io.BufferedWriter;
import cc.mallet.types.IDSorter;
import cc.mallet.util.Randoms;


public class SCVB implements Inferencer,OnlineModel{

    int numBatches;
    int numTopics;
    int numWords;

    double eta; //prior for words
    double etaSum;
    double alpha; //prior for topics
    double rho; //for the documents
    double rhophi; //for the overall distributions
    double[] nz;
    double[][] nphi;
    boolean rc;
    int c;
    InstanceList data;
    Alphabet alphabet;
    Alphabet topicAlphabet;
    int batchsize;
    Randoms random;


    public SCVB(int numTopics, int numWords,double eta,double alpha,double rho,double rhophi,boolean rc){
        this.numBatches = 0;
        this.rc = rc;
        this.numTopics = numTopics;
        this.numWords = numWords;
        this.eta = eta;
        this.etaSum = eta*numWords;
        this.alpha = alpha;
        this.rho = rho;
        this.rhophi = rhophi;
        this.initialize();
    }

    public SCVB(int numTopics, int numWords,double eta,double alpha,double rho,double rhophi,boolean rc,int batchsize){
        this(numTopics,numWords,eta,alpha,rho,rhophi,rc);
        this.batchsize = batchsize;
    }


    public double getEta(){
        return this.eta;
    }

    public int getNumTopics(){
        return this.numTopics;
    }

    public double getAlpha(){
        return this.alpha;
    }

    public int getBatchsize(){
        return this.batchsize;
    }

    public double getRho(){
        return this.rho;
    }

    public double getRhophi(){
        return this.rhophi;
    }

    public void sample(int numIterations)throws IOException{
        for(int i = 0;i<numIterations;++i){
            this.numBatches = 0;
            this.c = 0;
            int doc = 0;
            InstanceList batch=null;
            while(doc<data.size()){
                if(data.size()-doc>batchsize){
                    batch = data.subList(doc,doc+batchsize);
                }else{
                    batch = data.subList(doc,data.size());
                }
                updateBatch(batch);
                doc+=batchsize;
            }
        }

    }

    public void addInstances(InstanceList instances){
        this.data = instances;
        this.alphabet = instances.getDataAlphabet();
        this.topicAlphabet= instances.getTargetAlphabet();
    }

    public Inferencer getInferencer(){
        return this;
    }

    //returns top words for specific topic
    public IDSorter[] topWords(int topic) {
	IDSorter[] sortedWords = new IDSorter[numWords];
	for (int type = 0; type < numWords; type++) {
	    sortedWords[type] = new IDSorter(type, nphi[type][topic]);
	}
	Arrays.sort(sortedWords);
	return sortedWords;
    }


    public void printTopWords(String filename,int numTopWords,int wordThreshold){
        try{
            BufferedWriter bw = new BufferedWriter(new FileWriter(filename));
            for(int topic = 0;topic<numTopics;++topic){
                String label = (String)this.topicAlphabet.lookupObject(topic);
                bw.write("Topic " +label);
                bw.write("\t");
                IDSorter[] words = topWords(topic);
                for(int word = 0;word<numTopWords;++word){
                    int id = words[word].getID();
                    String wordStr = (String)this.alphabet.lookupObject(id);
                    bw.write(wordStr);
                    bw.write(" ");
                }
                bw.write("\n");
            }
            bw.flush();
            bw.close();
        }catch(Exception e){
            e.printStackTrace();
        }
    
    }

    public void initialize(){
        this.random = new Randoms();
        this.nz = new double[numTopics];
        this.nphi = new double[numWords][numTopics];
        for(int i = 0;i<nphi.length;++i){
            for(int j = 0;j<nphi[i].length;++j){
                nphi[i][j]=random.nextInt(numTopics*numWords)/(double)(numTopics*numWords);
                nz[j]+=nphi[i][j];
            }
        }
    }


    public void updateBatch(InstanceList instances){
        this.alphabet = instances.getDataAlphabet();
        this.topicAlphabet= instances.getTargetAlphabet();
    
        double[] nz_estimate = new double[numTopics];
        double[][] nphi_estimate = new double[numWords][numTopics];
        double[][] gamma = new double[numWords][numTopics];
        int miniBatchLength=0;
        for(Instance inst: instances){
            this.c+=((FeatureSequence)inst.getData()).getFeatures().length;
           miniBatchLength+=((FeatureSequence)inst.getData()).getFeatures().length;
        }
        this.c+=miniBatchLength;
        for(Instance instance: instances){
            this.update(instance,nz_estimate,nphi_estimate,gamma,miniBatchLength);
        }
        //update nphi and nz
        double realrho = 10.0/Math.pow(3000+numBatches,rhophi);;
        for(int topic = 0;topic<numTopics;++topic){
            nz[topic] = (1-realrho)*nz[topic]+realrho*nz_estimate[topic];
            for(int word = 0;word<numWords;++word){
                nphi[word][topic] = (1-realrho)*nphi[word][topic]+realrho*nphi_estimate[word][topic];
            }
        }
        this.numBatches++;
    }


    public void update(Instance instance,double[] nz_estimate, double[][] nphi_estimate,double[][] gamma,int minibatchLength){
        FeatureSequence tokenSequence =
            (FeatureSequence) instance.getData();
        FeatureVector labels =(FeatureVector) instance.getTarget();
        int numLabels = labels.numLocations();
        double[] n_theta = new double[numTopics];
        int docLength = tokenSequence.getLength();

        double div = c/(double)minibatchLength;

        for(int i = 0;i<2;++i){
            for(int pos = 0;pos<docLength;++pos){
                double realrho = 1.0/Math.pow(10+pos,rho);
                int word = tokenSequence.getIndexAtPosition(pos);
                double sum = 0;
                for(int rank = 0;rank<numLabels;++rank){
                    int topic = labels.indexAtLocation(rank);
                    //update gamma
                    gamma[word][topic] = (nphi[word][topic]+eta)/(double)(nz[topic]+etaSum);//*(n_theta[topic]+alpha);
                    sum+=gamma[word][topic];
                }
                
                for(int rank = 0;rank<numLabels;++rank){
                    int topic = labels.indexAtLocation(rank);
                    gamma[word][topic]/=sum;
                    //update n_theta
                    n_theta[topic] = (1-realrho)*n_theta[topic]+realrho*docLength*gamma[word][topic];
                    if(i>0){//after burnin
                        //update nphi_estimate
                        nphi_estimate[word][topic] = nphi_estimate[word][topic]+div*gamma[word][topic];
                        //update nz_estimate
                        nz_estimate[topic] = nz_estimate[topic]+div*gamma[word][topic];
                    }
                }
           
            }
        }
    }


    public double[] getSampledDistribution(Instance instance,int numIterations,int thinning,int burnin){
        double[] nz_estimate = new double[numTopics];
        double[][] nphi_estimate = new double[numWords][numTopics];

        double sum = this.updatePredict(instance,nz_estimate,nphi_estimate);

        //normalize nz_estimate
        for(int i = 0;i<nz_estimate.length;++i){
            nz_estimate[i]/=sum;
        }
        return nz_estimate;
    }


    public double updatePredict(Instance instance,double[] nz_estimate, double[][] nphi_estimate){
        FeatureSequence tokenSequence =
            (FeatureSequence) instance.getData();
        double nz_sum = 0;
        double[] n_theta = new double[numTopics];
        int docLength = tokenSequence.getLength();
        double[][] gamma_doc = new double[numWords][numTopics];
        double div = numWords;

        for(int i = 0;i<2;++i){
            for(int pos = 0;pos<docLength;++pos){
                double realrho = 1.0/Math.pow(10+pos,rho);
                int word = tokenSequence.getIndexAtPosition(pos);
                double sum = 0;
                for(int topic = 0;topic<numTopics;++topic){
                    //update gamma
                    gamma_doc[word][topic] = (nphi[word][topic]+eta)/(double)(nz[topic]+etaSum)*(n_theta[topic]+alpha);
                    sum+=gamma_doc[word][topic];
                }
                for(int topic = 0;topic<numTopics;++topic){
                    gamma_doc[word][topic]/=sum;
                    //update n_theta
                    n_theta[topic] = (1-realrho)*n_theta[topic]+realrho*docLength*gamma_doc[word][topic];
                    if(i>0){
                        //update nphi_estimate
                        nphi_estimate[word][topic] = nphi_estimate[word][topic]+div*gamma_doc[word][topic];
                        //update nz_estimate
                        nz_sum-=nz_estimate[topic];
                        nz_estimate[topic] = nz_estimate[topic]+div*gamma_doc[word][topic];
                        nz_sum+=nz_estimate[topic];
                    }
                }
           
            }
        }
        return nz_sum;
    }


}
