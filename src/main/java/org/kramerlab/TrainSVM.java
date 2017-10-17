package org.kramerlab;

import org.kramerlab.classifiers.*;

import java.util.ArrayList;
import mulan.classifier.MultiLabelOutput;
import meka.core.MLUtils;
import meka.classifiers.multilabel.Evaluation;
//import meka.classifiers.multilabel.BR;
import meka.core.Result;
import meka.core.Metrics;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.functions.LibLINEAR;

import weka.core.Instances;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.io.File;

import weka.classifiers.bayes.NaiveBayesUpdateable;



public class TrainSVM 
{



    /**reads parameters from config.txt, specify the required line as command line parameter
     path to training dataset, path to testing dataset, output directory for saving results, and percentage of the training set to use for validation*/
    public static void main( String[] args )
    {
        String configFile = "config.txt";
        int lineNumber = Integer.valueOf(args[0]);

        BufferedReader breader=null;
        String line=null;
        int count = 0;
        try{
            breader = new BufferedReader(new FileReader(configFile));
            while(count<lineNumber){
                line=breader.readLine();
                count++;
            }
        }catch(FileNotFoundException e1){
            e1.printStackTrace();
        }catch(IOException e2){
            e2.printStackTrace();
        }
        String[] pars = line.split(" ");

        String traindatasetFilename = pars[0];
        String testdatasetFilename = pars[1];
        String outputDir = pars[2];
        double validationPerc = Double.valueOf(pars[3]);
        outputDir+="/validation-"+validationPerc+"/";
        createDirectory(outputDir);
        //load a data set for testing
        Instances traindata = null;
        Instances validationtraindata = null;
        Instances validationtestdata = null;
        Instances testdata = null;

        try{
            traindata = new Instances(new BufferedReader(new FileReader(traindatasetFilename)));
            MLUtils.prepareData(traindata);
            System.out.println(traindata.classIndex());
            for(int i = 0;i<101;++i){
                System.out.println(i+" "+traindata.attribute(i).name()+" "+traindata.attribute(i).isNumeric()+" "+traindata.attribute(i).isNominal()+"\n");
            }
            System.out.println("last attribute: "+traindata.attribute(traindata.numAttributes()-1)+"\n");
            testdata = new Instances(new BufferedReader(new FileReader(testdatasetFilename)));
            MLUtils.prepareData(testdata);
            int split = (int)(traindata.numInstances()*(1-validationPerc));
            int toCopy = traindata.numInstances()-split;
            validationtestdata = new Instances(traindata,split,toCopy);
            validationtraindata = new Instances(traindata,0,split);
        }catch(Exception e){
            e.printStackTrace();
        }


        //Binary Relevance with LibLinear as base classifier
        LibLINEAR svm = new LibLINEAR();
        svm.setProbabilityEstimates(true);
        String[] possibleCs = new String[7];
        possibleCs[0]=String.valueOf(1.0/1000.0);
        possibleCs[1]=String.valueOf(1.0/100.0);
        possibleCs[2]=String.valueOf(1.0/10.0);
        possibleCs[3]=String.valueOf(1.0);
        possibleCs[4]=String.valueOf(10.0);
        possibleCs[5]=String.valueOf(100.0);
        possibleCs[6]=String.valueOf(1000.0);
        //select best C on validation set
        long start = System.currentTimeMillis();
        int bestC = selectC(validationtraindata,validationtestdata,possibleCs);
        svm.setCost(Double.valueOf(possibleCs[bestC]));
        BR br = new BR();
        br.setClassifier(svm);
        try{
            long end = System.currentTimeMillis();
            printEvalTime(outputDir+"/runtime.txt",start,end,-1);
            printEvalTime(outputDir+"/runtime.txt",start,end,0);
            //evaluate
            Evaluation eval = new Evaluation();
            Result result = eval.evaluateModel(br, traindata,testdata);
            int[][] trueVals = result.allTrueValues();
            double[][] preds = result.allPredictions();
            double macroAUC =  P_macroAUROC(trueVals,preds);
            Instances resinst = Metrics.curveDataMicroAveraged(trueVals,preds);
            double microAUC = ThresholdCurve.getROCArea(resinst);
            //write result to file
            BufferedWriter bw = new BufferedWriter(new FileWriter(outputDir+"/Micro-averaged AUC.txt"));
            bw.write(String.valueOf(microAUC));
            bw.flush();
            bw.close();
            bw = new BufferedWriter(new FileWriter(outputDir+"/Macro-averaged AUC.txt"));
            bw.write(String.valueOf(macroAUC));
            bw.flush();
            bw.close();

        }catch(Exception e){
            e.printStackTrace();
        }

    }

    //selects based on the first measure in the provided list
    public static int selectC(Instances train,Instances test, String[] cs){
        double[] results = new double[cs.length];
        double maxVal = -1;
        int maxInd = -1;
        for(int cInd = 0;cInd<cs.length;++cInd){
            LibLINEAR svm = new LibLINEAR();
            svm.setProbabilityEstimates(true);
            svm.setCost(Double.valueOf(cs[cInd]));
            BR br = new BR();
            br.setClassifier(svm);

            try{
                //evaluate
                Evaluation eval = new Evaluation();
                Result result = eval.evaluateModel(br, train, test);
                int[][] trueVals = result.allTrueValues();
                System.out.println("results dimensions "+trueVals.length+" "+trueVals[0].length);
                double[][] preds = result.allPredictions();
                Instances resinst = Metrics.curveDataMicroAveraged(trueVals,preds);
                double microAUC = ThresholdCurve.getROCArea(resinst);

                double myMacroAUC = P_macroAUROC(trueVals,preds);
                System.out.println("macro AUC "+myMacroAUC);
                results[cInd] = microAUC;
                if(maxInd == -1||results[cInd]>results[maxInd]){
                    maxInd = cInd;
                    maxVal = results[cInd];
                }
            }catch(Exception e){
                e.printStackTrace();
            }
            System.out.println("C: "+cs[cInd]+" result: "+results[cInd]);
        }
        return maxInd;
    }

    public static void createDirectory(String directory){
        File dirfile=new File(directory);
        if(!dirfile.exists()){
            try{
                Files.createDirectories(Paths.get(directory));
            }catch(IOException e){
                e.printStackTrace();
            }
        }
        System.out.println("directory: "+directory);
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
        System.out.println("average /L "+sumAll);
        //System.out.println(L+" "+java.util.Arrays.toString(AUC));
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
    public static int[] getCol(int[][] M, int k) {
        int[] col_k = new int[M.length];
        for (int i = 0; i < M.length; i++) {
            col_k[i] = M[i][k];
        }
        return col_k;
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

    public static void printEvalTime(String filename,long start,long end,int iter){
        try{
            if(iter==-1){
                BufferedWriter bwruntime = new BufferedWriter(new FileWriter(filename));
                bwruntime.close();
            }else{
                long runtime = end-start;
                BufferedWriter bwruntime = new BufferedWriter(new FileWriter(filename,true));
                bwruntime.write(iter+" "+String.valueOf(runtime));
                bwruntime.flush();
                bwruntime.close();
            }
        }catch(IOException e){
            e.printStackTrace();
        }    
        
    }


}
