package org.kramerlab.evaluation;

import org.kramerlab.interfaces.*;
import org.kramerlab.classifiers.*;


import meka.core.Metrics;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;

import cc.mallet.types.InstanceList;
import cc.mallet.types.Instance;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.IDSorter;

import java.util.ArrayList;
import java.util.Arrays;
import java.io.FileWriter;
import java.io.BufferedWriter;

public class Evaluation{


    public static Double[] evaluate(TopicModel model,InstanceList testingData,String outputDirectory,boolean writeDistributions,boolean parallelEvaluation){
        Double[] results = new Double[2]; //for storing micro- and macro-averaged AUC
        int[][] trueVals = new int[testingData.size()][];
        double[][] prediction = new double[testingData.size()][];
        try{
            BufferedWriter distWriter = new BufferedWriter(new FileWriter(outputDirectory+"/distributions.txt"));
            distWriter.write("\n");
            distWriter.flush();
            distWriter.close();
        }catch(Exception e){
            e.printStackTrace();
        }

        final int averageLabelNumber = findAverageLabelNumber(testingData);
        int numLabels =testingData.getTargetAlphabet().size();
        double[] distributionSum = new double[numLabels];
        
        int originalK = -1;

        if(model instanceof FastDependencyLDA){
            ((FastDependencyLDA)model).setEvaluate(true);
        }else if(model instanceof FastDependencyLDAReversed){
            ((FastDependencyLDAReversed)model).setEvaluate(true);
        }

        
        long start = System.currentTimeMillis();
        Inferencer ti=model.getInferencer();
        for(int i=0;i<testingData.size();++i){
            Instance testinst = testingData.get(i);
            double[] dist=ti.getSampledDistribution(testinst,1100,10,100);
            int[] truth = getTruth(testinst);
            trueVals[i]=truth;
            prediction[i]=dist;
            //System.out.println(Arrays.toString(dist));
            if(writeDistributions){
                try{
                    BufferedWriter distWriter = new BufferedWriter(new FileWriter(outputDirectory+"/distributions.txt",true));
                    distWriter.write(Arrays.toString(dist)+"\n");
                    distWriter.flush();
                    distWriter.close();
                }catch(Exception e){
                    e.printStackTrace();
                }
            }

        }
        long end = System.currentTimeMillis();
        try{
            BufferedWriter timeWriter = new BufferedWriter(new FileWriter(outputDirectory+"/evalTime.txt"));
            timeWriter.write(end-start+"\n");
            timeWriter.flush();
            timeWriter.close();
        }catch(Exception e){
            e.printStackTrace();
        }
        


        if(model instanceof FastDependencyLDA){
            ((FastDependencyLDA)model).setEvaluate(false);
        }else if(model instanceof FastDependencyLDAReversed){
            ((FastDependencyLDAReversed)model).setEvaluate(false);
        }

        //micro AUC
        Instances resinst = Metrics.curveDataMicroAveraged(trueVals,prediction);
        double microAUC = ThresholdCurve.getROCArea(resinst);

        results[0]=microAUC;
        //macro AUC
        double macroAUC = P_macroAUROC(trueVals,prediction);
        results[1]=macroAUC;

        System.out.println("Micro-averaged AUC "+results[0]);
        System.out.println("Macro-averaged AUC "+results[1]);

        return results;
        
    }



    public static int[] getTruth(Instance instance){
        int numlabels = instance.getTargetAlphabet().size();
        int[] truth_vals = new int[numlabels];
        FeatureVector labels = ((FeatureVector)instance.getTarget());
        int[] labelIndices = labels.getIndices();

        for(int j=0;j<labelIndices.length;++j){
            truth_vals[labelIndices[j]]=1;
        }
        return truth_vals;   
    }
    public static int findAverageLabelNumber(InstanceList data){
        int average = 0;
        for(Instance inst:data){
            average+=((FeatureVector)inst.getTarget()).numLocations();
        }
        if(data.size()>0) average/=data.size();
        return average;
    }




    public static boolean[] getBipartition(double[] distribution,int number){
        boolean[] bipartition = new boolean[distribution.length];
        IDSorter[] labelProbs = new IDSorter[distribution.length];
        for(int label = 0;label<labelProbs.length;++label){
            labelProbs[label] = new IDSorter(label,distribution[label]);
        }
        Arrays.sort(labelProbs);
        for(int num = 0;num<number;++num){
            bipartition[labelProbs[num].getID()]=true;
        }
        return bipartition;
    }

    /** Calculate AUROC: Area Under the ROC curve. */
    public static double P_macroAUROC(int Y[][], double P[][]) {
        // works with missing
        int L = Y[0].length;
        int nonZeroLabels = L;
        double AUC[] = new double[L];
        for(int j = 0; j < L; j++) {
            int[] yj = getCol(Y, j);
            if(allZeroOrMissing(yj)){
                nonZeroLabels--;
                continue;
            }
            ThresholdCurve curve = new ThresholdCurve();
            Instances result = curve.getCurve(meka.core.MLUtils.toWekaPredictions(yj, getCol(P, j)));
            AUC[j] = ThresholdCurve.getROCArea(result);

        }
        double sumAll = 0;
        for(int i = 0;i<AUC.length;++i){
            sumAll+=AUC[i];
        }
        sumAll/=nonZeroLabels;
        return sumAll;
    }

    /**
     * To check if all values for this label are zero or missing
     * 
     * @return If all labels are either missing or zero
     */
    public static boolean allZeroOrMissing(int[] real) {
        for (int i = 0; i < real.length; i++) {
            if (real[i] != 0 && real[i]!=-1) {
                return false;
            }
        }
        return true;
    }



    /**
     * from Meka MatrixUtils
     * GetCol - return the k-th column of M (as a vector).
     */
    public static double[] getCol(double[][] M, int k) {
        double[] col_k = new double[M.length];
        for (int i = 0; i < M.length; i++) {
            col_k[i] = M[i][k];
        }
        return col_k;
    }

    public static int[] getCol(int[][] M, int k) {
        int[] col_k = new int[M.length];
        for (int i = 0; i < M.length; i++) {
            col_k[i] = M[i][k];
        }
        return col_k;
    }


}
