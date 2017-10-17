package org.kramerlab.classifiers;


import cc.mallet.types.InstanceList;
import cc.mallet.types.Instance;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Alphabet;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.types.IDSorter;
import cc.mallet.util.Randoms;

import java.util.Arrays;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

/**like Fast-DependencyLDA, but stochastic variational Bayes*/
public class SCVBDependency extends SCVB{
    
    int cLabels;

    Randoms random;
    int numDepTopics;

    double etaDep; //prior for Dependency words
    double etaDepSum;
    double alphaDep; //prior for Dependency Topics

    double[] nzDep;
    double[][] nphiDep;
    double[] alpha;
    double alphaSum;

    Alphabet topicAlphabet;//label alphabet
    int iterations;

    public SCVBDependency(int numTopics, int numWords,double eta,double alpha,double rho,double rhophi,int T,double alphaDep,double etaDep,boolean rc){
        super(numTopics,numWords,eta,alpha,rho,rhophi,rc);
        this.numDepTopics = T;
        this.alphaDep = alphaDep;
        this.etaDep = etaDep;
        this.etaDepSum = etaDep*this.numTopics;
        this.alpha = new double[this.numDepTopics];
        for(int i = 0;i<this.alpha.length;++i){
            this.alpha[i]=alpha;
        }
        this.random = new Randoms();
        this.initializeDep();
    }

    public SCVBDependency(int numTopics, int numWords,double eta,double alpha,double rho,double rhophi,int T,double alphaDep,double etaDep,boolean rc,int batchsize){
        super(numTopics,numWords,eta,alpha,rho,rhophi,rc,batchsize);
        this.numDepTopics = T;
        System.out.println("HALLO!!! "+this.numTopics+" "+this.numDepTopics+" "+numTopics+" "+numWords);
        this.alphaDep = alphaDep;
        this.etaDep = etaDep;
        this.etaDepSum = etaDep*this.numTopics;
        this.alpha = new double[this.numDepTopics];
        for(int i = 0;i<this.alpha.length;++i){
            this.alpha[i]=alpha;
        }
        this.random = new Randoms();
        this.initializeDep();
        
    }

    public int getNumDepTopics(){
        return this.numDepTopics;
    }
    
    public void initializeDep(){
        this.nphiDep = new double[numTopics][numDepTopics];
        this.nzDep = new double[numDepTopics];
        for(int i = 0;i<nphiDep.length;++i){
            for(int j = 0;j<nphiDep[i].length;++j){
                nphiDep[i][j]=random.nextInt(numTopics*numDepTopics)/(double)(numTopics*numDepTopics);
                nzDep[j]+=nphiDep[i][j];
            }
        }
    }


    public void sample(int numIterations)throws IOException{
        this.cLabels=0;
        super.sample(numIterations);
    }


    public void updateBatch(InstanceList instances){
        this.alphabet = instances.getDataAlphabet();
        this.topicAlphabet = instances.getTargetAlphabet();
        double[] nz_estimate = new double[numTopics];
        double[][] nphi_estimate = new double[numWords][numTopics];
        double[] nzDep_estimate = new double[numDepTopics];
        double[][] nphiDep_estimate = new double[numTopics][numDepTopics];
        double[][] gamma = new double[numWords][numTopics];
        double[][] gammaD = new double[numTopics][numDepTopics];
        int miniBatchLength=0;
        int miniBatchLabelsLength=0;
        for(Instance inst: instances){
           miniBatchLength+=((FeatureSequence)inst.getData()).getFeatures().length;
           miniBatchLabelsLength+=((FeatureVector)inst.getTarget()).numLocations();
        }
        this.c+=miniBatchLength;
        this.cLabels+=miniBatchLabelsLength;
    
        for(Instance instance: instances){
            this.update(instance,nz_estimate,nphi_estimate,nzDep_estimate,nphiDep_estimate,gamma,gammaD,miniBatchLength,miniBatchLabelsLength);
            
        }
        //update nphi and nz
        double realrho = 10.0/Math.pow(2000+numBatches,rhophi);
        for(int topic = 0;topic<numTopics;++topic){
            nz[topic] = (1-realrho)*nz[topic]+realrho*nz_estimate[topic];
            for(int word = 0;word<numWords;++word){
                nphi[word][topic] = (1-realrho)*nphi[word][topic]+realrho*nphi_estimate[word][topic];
            }
        }
        //update nphiDep and nzDep
            realrho = 10.0/Math.pow(3000+numBatches,rhophi);
            for(int topic = 0;topic<numDepTopics;++topic){
                nzDep[topic] = (1-realrho)*nzDep[topic]+realrho*nzDep_estimate[topic];
                for(int word = 0;word<numTopics;++word){
                    nphiDep[word][topic] = (1-realrho)*nphiDep[word][topic]+realrho*nphiDep_estimate[word][topic];
                }
            }
    
        this.numBatches++;
    }

            

    public IDSorter[] toplabels(int topic){
	IDSorter[] sortedLabels = new IDSorter[numTopics];
	for (int type = 0; type < numTopics; type++) {
	    sortedLabels[type] = new IDSorter(type, (nphiDep[type][topic]/nzDep[topic]));
	}
	Arrays.sort(sortedLabels);
	return sortedLabels;
    }


    @Override
    public void printTopWords(String filename,int numTopWords,int threshold){
        try{
            BufferedWriter bw = new BufferedWriter(new FileWriter(filename));
            for(int label = 0;label<numTopics;++label){
                bw.write("Label "+this.topicAlphabet.lookupObject(label)+"\n");

                IDSorter[] topwords = topWords(label);
                for(int word = 0;word<numTopWords;++word){
                    int id = topwords[word].getID();
                    String wordStr = (String)this.alphabet.lookupObject(id);
                    bw.write(wordStr);
                    bw.write(" ");
                }
                bw.write("\n");
            }
            for(int topic = 0;topic<numDepTopics;++topic){
                bw.write("Topic "+topic+"\n");
                IDSorter[] toplabels = toplabels(topic);
                for(int label = 0;label<5;++label){
                    int id = toplabels[label].getID();
                    String labelStr = (String)this.topicAlphabet.lookupObject(id);
                    bw.write(labelStr);
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



    public void update(Instance instance,double[] nz_estimate, double[][] nphi_estimate,double[] nzDep_estimate, double[][] nphiDep_estimate,double[][] gamma, double[][] gammaD,int minibatchLength,int minibatchLabelsLength){
        FeatureSequence tokenSequence =
            (FeatureSequence) instance.getData();
        FeatureVector labels =(FeatureVector) instance.getTarget();
        int numLabels = labels.numLocations();
        double[] n_theta = new double[numDepTopics];
        int docLength = tokenSequence.getLength();
    
        double div = c/(double)minibatchLength;
        double divLabels = cLabels*c/(double)minibatchLabelsLength/(double)minibatchLength;

        for(int i = 0;i<2;++i){

            //learn label-topic model
            for(int rank = 0;rank<numLabels;++rank){
                int label = labels.indexAtLocation(rank);
                double realrho = 1.0/Math.pow(10,rho);    
                double sumLabel = 0;
                for(int topic = 0;topic<numDepTopics;++topic){
                    gammaD[label][topic] = (nphiDep[label][topic]+etaDep)/(nzDep[topic]+etaDepSum)*(n_theta[topic]+alpha[topic]);
                    sumLabel+=gammaD[label][topic];
                }
                for(int topic = 0;topic<numDepTopics;++topic){
                    double gammaVal = gammaD[label][topic]/sumLabel;
                    n_theta[topic] = (1-realrho)*n_theta[topic]+realrho*docLength*numLabels*gammaVal;
                    if(i>0){//after burnin
                            //update nphi_estimate
                        nphiDep_estimate[label][topic] += div*docLength*gammaVal;
                        nzDep_estimate[topic] += div*docLength*gammaVal;
                    }
                }
                }


            //learn word-label model
            for(int pos = 0;pos<docLength;++pos){
                int word = tokenSequence.getIndexAtPosition(pos);
                double[] labelSums = new double[numTopics];
                double[] wordSums = new double[numTopics];
                double sumWord = 0;
                double sum = 0;
                for(int rank = 0;rank<numLabels;++rank){
                    int label = labels.indexAtLocation(rank);
                    gamma[word][label] = (nphi[word][label]+eta)/(double)(nz[label]+etaSum);
                    sumWord+=gamma[word][label];
                }
                for(int rank = 0;rank<numLabels;++rank){
                    int label = labels.indexAtLocation(rank);
                    double gammaVal = gamma[word][label]/sumWord;
                    if(i>0){//after burnin
                            //update nphi_estimate
                        nphi_estimate[word][label] += div*gammaVal;
                        //update nz_estimate
                        nz_estimate[label] += div*gammaVal;
                    }
                }
            }
            
        }
    }

    public double[] getSampledDistribution(Instance instance,int numIterations,int thinning,int burnin){
        double[] nz_estimate = new double[numTopics];
        double[][] nphi_estimate = new double[numWords][numTopics];
        double[][] gamma = new double[numWords][numTopics];
        double[][] gammaD = new double[numTopics][numDepTopics];
        double[] nzDep_estimate = new double[numDepTopics];
        double[][] nphiDep_estimate = new double[numTopics][numDepTopics];

        double[] n_theta = this.updatePredict(instance,nz_estimate,nphi_estimate,nzDep_estimate,nphiDep_estimate,gamma,gammaD,1,numTopics);

        double[] prediction = new double[numTopics];

        FeatureSequence tokenSequence =
            (FeatureSequence) instance.getData();
       
        //normalize nz_estimate
        double sum = 0;
        for(int i = 0;i<nz_estimate.length;++i){
            sum+=nz_estimate[i];
        }
        for(int i = 0;i<nz_estimate.length;++i){
            nz_estimate[i]/=sum;
        }
        return nz_estimate;
    }


    public double[] updatePredict(Instance instance,double[] nz_estimate, double[][] nphi_estimate,double[] nzDep_estimate,double[][] nphiDep_estimate,double[][] gamma,double[][] gammaD,int minibatchLength,int minibatchLabelsLength){
        FeatureSequence tokenSequence =
            (FeatureSequence) instance.getData();
        double nz_sum = 0;
        double[] n_theta = new double[numDepTopics];
        int docLength = tokenSequence.getLength();
        double div = c/(double)minibatchLength;
        double divLabels = cLabels/(double)minibatchLabelsLength;
        for(int i = 0;i<2;++i){
            for(int pos = 0;pos<docLength;++pos){
                double wordSum=0;
                double realrho = 1.0/Math.pow(10+pos,rho);
                int word = tokenSequence.getIndexAtPosition(pos);
                double sum = 0;
                double[] labelSums = new double[numTopics];
                for(int label = 0;label<numTopics;++label){    
                    double labelSum=0;
                    for(int topic = 0;topic<numDepTopics;++topic){
                        gamma[word][label] = (nphi[word][label]+eta)/(double)(nz[label]+etaSum);//*(nphiDep[label][topic]+etaDep);
                        gammaD[label][topic] = (nphiDep[label][topic]+etaDep)/(nzDep[topic]+etaDepSum)*(n_theta[topic]+this.alpha[topic]);
                        wordSum+=gamma[word][label]*(nphiDep[label][topic]+etaDep);
                        sum+=gamma[word][label]*gammaD[label][topic];
                    }
                }

                for(int topic = 0;topic<numDepTopics;++topic){
                    double thsum = 0;
                    for(int label = 0;label<numTopics;++label){    
                        double gammaVal = gamma[word][label]*gammaD[label][topic]/sum;//labelSums[label];
                        thsum+=gammaVal;
                        if(i>0){
                            //update nphi_estimate
                            nphi_estimate[word][label] += div*gammaVal;
                            //update nz_estimate
                            nz_sum-=nz_estimate[label];
                            nz_estimate[label] += div*gammaVal;
                            nz_sum+=nz_estimate[label];
                        }
                    }
                    n_theta[topic] = (1-realrho)*n_theta[topic]+realrho*docLength*thsum;
                }
           
            }
        }
        return n_theta;
    }


}

