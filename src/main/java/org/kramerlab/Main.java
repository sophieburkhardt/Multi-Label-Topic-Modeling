package org.kramerlab;

import org.kramerlab.classifiers.*;
import org.kramerlab.interfaces.*;
import org.kramerlab.evaluation.*;
import org.kramerlab.util.*;

import org.apache.commons.cli.*;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.io.File;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Iterator;

import cc.mallet.types.FeatureVector;
import cc.mallet.types.Alphabet;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Instance;
import cc.mallet.util.Randoms;
import cc.mallet.types.IDSorter;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.util.concurrent.Executors;
import java.lang.InterruptedException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.Collection;

public class Main{

    public enum Algorithm{
        DEPENDENCY,FASTDEPENDENCY,SCVB,SCVBDEPENDENCY
    }

    public static void main(String[] args){
        try{
            String name = "config.txt";
            String lineN = args[0];
            String[] toParse = new String[4];
            toParse[0] = "-f";
            toParse[1] = name;
            toParse[2] = "-l";
            toParse[3] = lineN;
            //read filename and linenumber where parameters are stored
            Options mainOptions = new Options().addOption("f",true,"filename where parameters are stored").addOption("l",true,"line number where parameters are in the file");
            CommandLineParser mainParser = new GnuParser();
            CommandLine mainLine = mainParser.parse(mainOptions, toParse);
            String fileName = mainLine.getOptionValue("f");
            int lineNumber = Integer.valueOf(mainLine.getOptionValue("l"));
            String[] parLine = readLine(fileName,lineNumber);
            //read in the selected algorithm from retrieved line
            OptionGroup algorithmTypeGroup = new OptionGroup().addOption(new Option("dependency",false,"DependencyLDA")).addOption(new Option("fastdependency",false,"Fast-DependencyLDA")).addOption(new Option("SCVB",false,"SCVB")).addOption(new Option("SCVBDependency",false,"SCVB Dependency"));
            algorithmTypeGroup.setRequired(true);
            Option supervisedOption = new Option("supervised",true,"whether or not the algorithm is supervised");
            Options methodOptions = new Options().addOptionGroup(algorithmTypeGroup).addOption(supervisedOption);
            CommandLineParser algParser = new ExtendedGnuParser(true);
            CommandLine algLine = algParser.parse(methodOptions, parLine);
            String selectedAlgorithm = algorithmTypeGroup.getSelected();
            String superv = algLine.getOptionValue("supervised");
            boolean supervised = Boolean.valueOf(superv);
            //get the options for the selected algorithm and parse them
            Algorithm alg = intValue(selectedAlgorithm);
            Options options = constructOptions(alg, supervised);
            CommandLineParser paraParser = new ExtendedGnuParser(true);
            //System.out.println(Arrays.toString(parLine)+" "+options.toString());
            System.out.println("now parse main options");
            CommandLine paraLine = paraParser.parse(options, parLine);
            runAlgorithm(alg, paraLine, supervised);
        }catch(ParseException e){
            e.printStackTrace();
        }/*catch(FileNotFoundException e){
            e.printStackTrace();
            }*/
    }


    public static void runAlgorithm(Algorithm alg, CommandLine options, boolean supervised){
        //read in data
        String dataFilename = options.getOptionValue("dataFilename");
        String datasetName = options.getOptionValue("datasetName");
        String testDataFilename = options.getOptionValue("testDataFilename");
        System.out.println("filename "+dataFilename+" dataset "+datasetName+" ");
	String inputType = options.getOptionValue("inputType");
        ImportExample ie = new ImportExample();
        InstanceList data = readData(ie,dataFilename, supervised,inputType);
        InstanceList trainingData = null;
        InstanceList testingData = null;
        if(testDataFilename!=null){
            trainingData = data;
            testingData = readData(ie,testDataFilename,supervised,inputType);
        }

        TopicModel model = null;
        String dirBaseString="allresults/"+datasetName+"/";
        String dirEndString=null;
        int numIterations;


       int numChains = 1;
       if(options.getOptionValue("numChains")!=null){
           numChains = Integer.valueOf(options.getOptionValue("numChains"));
       }

       boolean rc=false;
        if(options.getOptionValue("readExisting")!=null){
            //read existing model
            System.out.println("reading existing model...");
            String readFilename = options.getOptionValue("modelFilename");
            try{
                model = read(new File(readFilename));
            }catch(Exception e){
                e.printStackTrace();
            }
        }else{
            switch(alg){
            case DEPENDENCY: 
                dirBaseString+="DependencyLDA-Original/";
                model = dependencyInit(options,data,supervised);
                numIterations = Integer.valueOf(options.getOptionValue("numIterations"));  
                DependencyLDAOriginal mod = (DependencyLDAOriginal) model;
                dirEndString = mod.getAlphaSum()+"-"+mod.getBeta()+"-"+mod.testalpha+"/"+mod.eta+"/"; 
                dirEndString+=numIterations+"/"; 

                break;
            case FASTDEPENDENCY: 
                boolean reversed = Boolean.valueOf(options.getOptionValue("reversed"));  
                if(reversed) dirBaseString+="Fast-DependencyLDA-Reversed/";
                else dirBaseString+="Fast-DependencyLDA/";
                model = mydependencyInit(options,data,supervised,reversed);
                numIterations = Integer.valueOf(options.getOptionValue("numIterations"));  

                if(reversed){
                    FastDependencyLDAReversed mod2 = (FastDependencyLDAReversed) model;
                    dirEndString = mod2.alphaSum+"-"+mod2.beta+"-"+mod2.testalpha+"/"+mod2.gammaLabelSum+"-"+mod2.testgammaLabelSum+"/"; 
                    dirEndString+=numIterations+"/"; 
                }else{
                    FastDependencyLDA mod3 = (FastDependencyLDA) model;
                    dirEndString = mod3.alphaSum+"-"+mod3.beta+"-"+mod3.testalpha+"/"+mod3.gammaLabelSum+"-"+mod3.testgammaLabelSum+"/"; 
                    dirEndString+=numIterations+"/"; 
                }

                break;
            case SCVB:
                dirBaseString+="SCVB/"; 
                numIterations = Integer.valueOf(options.getOptionValue("numIterations"));  
                model = CVBInit(options,data,supervised);
                SCVB mod5 = (SCVB) model;
                dirEndString = mod5.getEta()+"-"+mod5.getAlpha()+"/"+mod5.getRho()+"-"+mod5.getRhophi()+"/"+mod5.getBatchsize()+"-"+mod5.getNumTopics()+"/"+numIterations+"/"; 
                break;
            case SCVBDEPENDENCY:
                dirBaseString+="SCVBDependency/"; 
                numIterations = Integer.valueOf(options.getOptionValue("numIterations"));  
                model = CVBDepInit(options,data);
                SCVBDependency scvb = (SCVBDependency)model;
                dirEndString = scvb.getEta()+"-"+scvb.getAlpha()+"/"+scvb.getRho()+"-"+scvb.getRhophi()+"/"+scvb.getBatchsize()+"-"+scvb.getNumDepTopics()+"/"+numIterations+"/"; 
                
                break;
            }
        }

        if(options.getOptionValue("run")!=null){
            Integer run = Integer.valueOf(options.getOptionValue("run"));
            if(run>1)data.shuffle(new Randoms());
            dirEndString = dirEndString+"run-"+run+"/";
        }

        //batch, incremental, incremental batch ?
        if(options.getOptionValue("batch")!=null){
            String outputDirectory = dirBaseString+"justbatch/"+dirEndString;
            createDirectory(outputDirectory);
            if(supervised){
                //divide into training and testing data
                if(options.getOptionValue("trainingPercentage")!=null&&testingData==null){
                    double trainingPercentage = Double.valueOf(options.getOptionValue("trainingPercentage"));
                    int split = (int)(data.size()*trainingPercentage);
                    trainingData = data.subList(0,split);
                    testingData = data.subList(split,data.size());
                }
            }else{
                trainingData = data;
            }
            System.out.println(trainingData.size()+" "+testingData.size());
            trainBatch(model,options,trainingData,testingData,outputDirectory,supervised);
        }else if(options.getOptionValue("incremental")!=null){
            //divide into training and testing data
            if(options.getOptionValue("trainingPercentage")!=null&&testingData==null){
                double trainingPercentage = Double.valueOf(options.getOptionValue("trainingPercentage"));
                int split = (int)(data.size()*trainingPercentage);
                trainingData = data.subList(0,split);
                testingData = data.subList(split,data.size());
            }
            String outputDirectory = dirBaseString+"online/"+dirEndString;
            createDirectory(outputDirectory);
            int batchsize = Integer.valueOf(options.getOptionValue("batchSize"));
            trainOnline((OnlineModel)model,options,data,batchsize,outputDirectory,supervised);
            //trainOnlineFixedTestset((OnlineModel)model,options,trainingData,testingData,batchsize,outputDirectory,supervised);
        }
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


    public static TopicModel read (File f) throws Exception {

        TopicModel topicModel = null;

        ObjectInputStream ois = new ObjectInputStream (new FileInputStream(f));
        topicModel = (TopicModel) ois.readObject();
        ois.close();

        return topicModel;
    }


    public static void save(TopicModel model, String filename){
        try {
            ObjectOutputStream oos = new ObjectOutputStream (new FileOutputStream(filename));
            oos.writeObject(model);
            oos.close();
        } catch (IOException e) {
            System.err.println("Problem serializing Model to file " +
                               filename + ": " + e);
        }
    }


    public static void trainOnline(OnlineModel model,CommandLine options,InstanceList data,int batchsize,String outputDirectory,boolean supervised){
        ArrayList<String> measureNames = new ArrayList<String>();
        measureNames.add("Micro-averaged AUC");
        measureNames.add("Macro-averaged AUC");

        printEvalTime(outputDirectory+"runtime.txt",0,0,-1);    
        int numTopWords = Integer.valueOf(options.getOptionValue("numTopWords"));
        int wordThreshold = Integer.valueOf(options.getOptionValue("wordThreshold"));
        boolean print = Boolean.valueOf(options.getOptionValue("print"));
        boolean parallel = Boolean.valueOf(options.getOptionValue("parallelEvaluation"));
        boolean writeDistributions = Boolean.valueOf(options.getOptionValue("writeDistributions"));
        String saveFilename = null;
        boolean saveModel = Boolean.valueOf(options.getOptionValue("saveModel"));
        if(saveModel) saveFilename = options.getOptionValue("saveFilename");
        int instanceCounter = 0;
        boolean started = false;
        while(instanceCounter<data.size()-(2*batchsize)){            
            System.out.println(instanceCounter);
            long start = System.currentTimeMillis();
            InstanceList batch = data.subList(instanceCounter,instanceCounter+batchsize);
            if(started){
                model.updateBatch(batch);
            }else{
                model.addInstances(batch);
                try{
                    model.sample(100);
                }catch(IOException e){
                    e.printStackTrace();
                }
            }
            long end = System.currentTimeMillis();
            printEvalTime(outputDirectory+"runtime.txt",start,end,(instanceCounter+batchsize));    
            InstanceList testingData = data.subList(instanceCounter+batchsize,instanceCounter+(2*batchsize));
            Double[] results = Evaluation.evaluate(model,testingData,outputDirectory,writeDistributions,parallel);
            if(print) model.printTopWords(outputDirectory+"topics.txt",numTopWords,wordThreshold);
            if(saveModel) save(model,saveFilename);
            if(!started){
                //delete previous file contents
                for(String name: measureNames){
                    try{
                        BufferedWriter bw = new BufferedWriter(new FileWriter(outputDirectory+name+".txt"));
                        bw.write("");
                        bw.flush();
                        bw.close();
                    }catch(IOException e){
                        e.printStackTrace();
                    }
                }

            }
            //write results to file
            int measureCount = 0;
            for(String name: measureNames){
                try{
                    BufferedWriter bw = new BufferedWriter(new FileWriter(outputDirectory+name+".txt",true));
                    bw.write((instanceCounter+batchsize) +" "+results[measureCount]);
                    bw.write("\n");
                    bw.flush();
                    bw.close();
                }catch(IOException e){
                    e.printStackTrace();
                }
            }
            started = true;
            instanceCounter+=batchsize;
        }
    }



    public static void trainOnlineFixedTestset(OnlineModel model,CommandLine options,InstanceList data,InstanceList testdata,int batchsize,String outputDirectory,boolean supervised){
        System.out.println("train with fixed test set");
        int numIterations = Integer.valueOf(options.getOptionValue("numIterations"));
        printEvalTime(outputDirectory+"runtime.txt",0,0,-1);    
        int numTopWords = Integer.valueOf(options.getOptionValue("numTopWords"));
        int wordThreshold = Integer.valueOf(options.getOptionValue("wordThreshold"));
        boolean print = Boolean.valueOf(options.getOptionValue("print"));
        boolean parallel = Boolean.valueOf(options.getOptionValue("parallelEvaluation"));
        boolean writeDistributions = Boolean.valueOf(options.getOptionValue("writeDistributions"));
        String saveFilename = null;
        boolean saveModel = Boolean.valueOf(options.getOptionValue("saveModel"));
        if(saveModel) saveFilename = options.getOptionValue("saveFilename");
        ArrayList<String> measureNames = new ArrayList<String>();
        measureNames.add("Micro-averaged AUC");
        measureNames.add("Macro-averaged AUC");
        boolean started = false;
        for(int iter=0;iter<numIterations;++iter){
        int instanceCounter = 0;
        while(instanceCounter<data.size()-(2*batchsize)){            
            System.out.println(instanceCounter);
    
            InstanceList batch = data.subList(instanceCounter,instanceCounter+batchsize);
            long start = System.currentTimeMillis();    
            model.updateBatch(batch);
            long end = System.currentTimeMillis();
            printEvalTime(outputDirectory+"runtime.txt",start,end,(instanceCounter+batchsize));    
            Double[] results =  Evaluation.evaluate(model,testdata,outputDirectory,writeDistributions,parallel);
            if(print) model.printTopWords(outputDirectory+"topics.txt",numTopWords,wordThreshold);
            if(saveModel) save(model,saveFilename);
            if(!started){
                //delete previous file contents
                for(String name: measureNames){
                    try{
                        BufferedWriter bw = new BufferedWriter(new FileWriter(outputDirectory+name+".txt"));
                        bw.write("");
                        bw.flush();
                        bw.close();
                    }catch(IOException e){
                        e.printStackTrace();
                    }
                }

            }
            //write results to file
            int measureCount = 0;
            for(String name: measureNames){
                try{
                    BufferedWriter bw = new BufferedWriter(new FileWriter(outputDirectory+name+".txt",true));
                    bw.write((instanceCounter+batchsize) +" "+results[measureCount]);
                    bw.write("\n");
                    bw.flush();
                    bw.close();
                }catch(IOException e){
                    e.printStackTrace();
                }
                measureCount++;
            }
            started = true;
            instanceCounter+=batchsize;
        }
        }
    }


    public static void trainBatch(TopicModel model,CommandLine options,InstanceList trainingData,InstanceList testingData,String outputDirectory,boolean supervised){
        ArrayList<String> measureNames = new ArrayList<String>();
        measureNames.add("Micro-averaged AUC");
        measureNames.add("Macro-averaged AUC");

        try{
            printEvalTime(outputDirectory+"runtime.txt",0,0,-1);    
            int numIterations = Integer.valueOf(options.getOptionValue("numIterations"));
            int numTopWords = Integer.valueOf(options.getOptionValue("numTopWords"));
            int wordThreshold = Integer.valueOf(options.getOptionValue("wordThreshold"));
            int evalInterval = -1;
            if(supervised) evalInterval = Integer.valueOf(options.getOptionValue("evaluationInterval"));
            boolean print = Boolean.valueOf(options.getOptionValue("print"));
            boolean parallel = Boolean.valueOf(options.getOptionValue("parallelEvaluation"));
            boolean writeDistributions = Boolean.valueOf(options.getOptionValue("writeDistributions"));
            String saveFilename = null;
            boolean saveModel = Boolean.valueOf(options.getOptionValue("saveModel"));
            if(saveModel) saveFilename = options.getOptionValue("saveFilename");
            boolean started = false;
            model.addInstances(trainingData);
            for(int iter = 0;iter<numIterations;++iter){
                System.out.println("iteration "+iter);
                long start = System.currentTimeMillis();
                model.sample(1);
                long end = System.currentTimeMillis();
                printEvalTime(outputDirectory+"runtime.txt",start,end,iter);    
                if(print) model.printTopWords(outputDirectory+"topics.txt",numTopWords,wordThreshold);
                if(saveModel) save(model,saveFilename);
                if(supervised&&(iter+1)%evalInterval==0){
                    //evaluate model here
                    Double[] results = Evaluation.evaluate(model,testingData,outputDirectory,writeDistributions,parallel);
                    if(!started){
                        //delete previous file contents
                        for(String name: measureNames){
                            try{
                                BufferedWriter bw = new BufferedWriter(new FileWriter(outputDirectory+name+".txt"));
                                bw.write("");
                                bw.flush();
                                bw.close();
                            }catch(IOException e){
                                e.printStackTrace();
                            }
                        }

                    }
                    //write results to file
                    int measureCount = 0;
                    for(String name: measureNames){
                        try{
                            BufferedWriter bw = new BufferedWriter(new FileWriter(outputDirectory+name+".txt",true));
                            bw.write(iter +" "+results[measureCount]);
                            bw.write("\n");
                            bw.flush();
                            bw.close();
                        }catch(IOException e){
                            e.printStackTrace();
                        }
                        measureCount++;
                    }
                    started = true;
                }
            }
    
        }catch(IOException e){
            e.printStackTrace();
        }
    }



    
    public static void printEvalTime(String filename,long start,long end,int iter){
        try{
            if(iter==-1){
                BufferedWriter bwruntime = new BufferedWriter(new FileWriter(filename));
                bwruntime.close();
            }else{
                long runtime = end-start;
                BufferedWriter bwruntime = new BufferedWriter(new FileWriter(filename,true));
                bwruntime.write(iter+" "+String.valueOf(runtime)+"\n");
                bwruntime.flush();
                bwruntime.close();
            }
        }catch(IOException e){
            e.printStackTrace();
        }    
        
    }

    public static InstanceList readData(ImportExample ie,String filename, boolean supervised,String inputType){
        InstanceList il = null;
        il = ie.readFromFile(filename);
        System.out.println("number of features: "+il.getDataAlphabet().size());
        return il;
    }

    public static TopicModel CVBDepInit(CommandLine options, InstanceList data){
        TopicModel model = null;
        int numWords = data.getDataAlphabet().size();
        double eta = Double.valueOf(options.getOptionValue("eta"));
        double alpha = Double.valueOf(options.getOptionValue("alpha"));
        double rho = Double.valueOf(options.getOptionValue("rho"));
        double rhophi = Double.valueOf(options.getOptionValue("rhophi"));
        int numTopics = numTopics = data.getTargetAlphabet().size();
        int batchsize = Integer.valueOf(options.getOptionValue("batchSize"));

        double etaDep = Double.valueOf(options.getOptionValue("etaDep"));
        double alphaDep = Double.valueOf(options.getOptionValue("alphaDep"));
        Integer t = Integer.valueOf(options.getOptionValue("t"));
        boolean rc = Boolean.valueOf(options.getOptionValue("rc"));
    

        model = new SCVBDependency(numTopics,numWords,eta,alpha,rho,rhophi,t,alphaDep,etaDep,rc,batchsize);

        return model;
    }

    public static TopicModel CVBInit(CommandLine options, InstanceList data, boolean supervised){
        TopicModel model = null;
        int numWords = data.getDataAlphabet().size();
        double eta = Double.valueOf(options.getOptionValue("eta"));
        double alpha = Double.valueOf(options.getOptionValue("alpha"));
        double rho = Double.valueOf(options.getOptionValue("rho"));
        double rhophi = Double.valueOf(options.getOptionValue("rhophi"));
        int numTopics = Integer.valueOf(options.getOptionValue("topicNumber"));
        int batchsize = Integer.valueOf(options.getOptionValue("batchSize"));
    
        model = new SCVB(data.getTargetAlphabet().size(),numWords,eta,alpha,rho,rhophi,true,batchsize);
                
        return model;
    }

    
    /**function to get a label Alphabet from a normal Alphabet*/
    public static LabelAlphabet makeLabelAlphabet(Alphabet alphabet){
        LabelAlphabet la = new LabelAlphabet();
        for(int i=0;i<alphabet.size();++i){
            la.lookupIndex(alphabet.lookupObject(i),true);
        }
        return la;
    }
    

    public static Options dependencyOptions(boolean supervised){
        Options options = new Options();
        Option[] opts = new Option[11];
        opts[0] = new Option("alphaSum",true,"alphaSum, prior for document-label distributions");
        opts[1] = new Option("testAlphaSum",true,"alphaSum for testing, prior for document-label distributions");
        opts[2] = new Option("beta",true,"beta, prior for topic-word distributions");
        opts[3] = new Option("eta",true,"eta");
        opts[4] = new Option("beta_c",true,"beta_c");
        opts[5] = new Option("testEta",true,"eta for testing");
        opts[6] = new Option("gamma",true,"gamma, prior for document-topic distributions");
        opts[7] = new Option("testGamma",true,"gamma for testing");
        opts[8] = new Option("innerIterations",true,"number of inner iterations");
        opts[9] = new Option("t",true,"number of latent topics");
        opts[10] = new Option("original",true,"whether to use the original DependencyLDA with the labels as tokens");

        for(int i = 0;i<opts.length;++i){
            options.addOption(opts[i]);
        }

        return options;
    }


    public static Options mydependencyOptions(boolean supervised){
        Options options = new Options();
        Option[] opts = new Option[8];
        opts[0] = new Option("alphaSum",true,"alphaSum, prior for document-label distributions");
        opts[1] = new Option("testAlphaSum",true,"alphaSum for testing, prior for document-topic distributions");
        opts[2] = new Option("beta",true,"beta, prior for topic-word distributions");
        opts[3] = new Option("gamma",true,"gamma, prior for label-topic distributions");
        opts[4] = new Option("testGamma",true,"gamma for testing");
        opts[5] = new Option("t",true,"number of latent topics");
        opts[6] = new Option("reversed",true,"whether to reverse topics and labels");
        opts[7] = new Option("s2",true,"whether to use evaluation strategy S2");
        for(int i = 0;i<opts.length;++i){
            options.addOption(opts[i]);
        }

        return options;
    }


    public static TopicModel dependencyInit(CommandLine options,InstanceList data, boolean supervised){
        double alphaSum = Double.valueOf(options.getOptionValue("alphaSum"));    
        double testAlphaSum = Double.valueOf(options.getOptionValue("testAlphaSum"));    
        double beta = Double.valueOf(options.getOptionValue("beta"));    
        double eta = Double.valueOf(options.getOptionValue("eta"));    
        double beta_c = Double.valueOf(options.getOptionValue("beta_c"));    
        double testEta = Double.valueOf(options.getOptionValue("testEta"));    
        double gamma = Double.valueOf(options.getOptionValue("gamma"));    
        double testGamma = Double.valueOf(options.getOptionValue("testGamma"));    
        int innerIterations = Integer.valueOf(options.getOptionValue("innerIterations"));    
        int T = Integer.valueOf(options.getOptionValue("t"));    
        int numChains = 1;
        if(options.getOptionValue("numChains")!=null){
            numChains = Integer.valueOf(options.getOptionValue("numChains"));
        }
        TopicModel model = null;

        model = new DependencyLDAOriginal(makeLabelAlphabet(data.getTargetAlphabet()),alphaSum,beta,new Randoms(),eta,beta_c,gamma,T,testAlphaSum,testEta,testGamma,innerIterations);

        return model;
    }

    public static TopicModel mydependencyInit(CommandLine options,InstanceList data, boolean supervised,boolean reversed){
        double alphaSum = Double.valueOf(options.getOptionValue("alphaSum"));    
        double testAlphaSum = Double.valueOf(options.getOptionValue("testAlphaSum"));    
        double beta = Double.valueOf(options.getOptionValue("beta"));    
        double gamma = Double.valueOf(options.getOptionValue("gamma"));    
        double testGamma = Double.valueOf(options.getOptionValue("testGamma"));    
        boolean s2 = Boolean.valueOf(options.getOptionValue("s2"));    
        int T = Integer.valueOf(options.getOptionValue("t"));    
        int numChains = 1;
        if(options.getOptionValue("numChains")!=null){
            numChains = Integer.valueOf(options.getOptionValue("numChains"));
        }
        TopicModel model = null;

        if(reversed){
            model = new FastDependencyLDAReversed(makeLabelAlphabet(data.getTargetAlphabet()),alphaSum,beta,gamma,testAlphaSum,testGamma,T);
        }else{
            model = new FastDependencyLDA(makeLabelAlphabet(data.getTargetAlphabet()),alphaSum,beta,gamma,testAlphaSum,testGamma,T);
            ((FastDependencyLDA)model).setEvalStrategy(s2);
        }

        return model;
    }


        
    public static Options generalOptions(){
        Options options = new Options();
        Option parallelOption = new Option("parallelEvaluation",true,"do the evaluation in parallel");
        Option writeOption = new Option("writeDistributions",true,"write distributions to file");
        Option dataFileOption = new Option("dataFilename",true,"filename for data");
        Option testdataFileOption = new Option("testDataFilename",true,"filename for testingdata");
        Option inputTypeOption = new Option("inputType",true,"input type, e.g. json or txt");
        Option datasetOption = new Option("datasetName",true,"name of dataset");
        Option percOption = new Option("trainingPercentage",true,"percentage of data set to use for training in case of batch training");
        Option numIterOption = new Option("numIterations",true,"number of iterations for training");
        Option numTopOption = new Option("numTopWords",true,"number of words to display per topic");
        Option evalInterval = new Option("evaluationInterval",true,"after how many training steps to evaluate");
        Option wordThresholdOption = new Option("wordThreshold",true,"only display words that occur more often than this");
	Option printOption = new Option("print",true,"whether or not to print the resulting topics to file");    
	Option saveOption = new Option("saveModel",true,"whether or not to save the model to file");    
	Option saveFilenameOption = new Option("saveFilename",true,"filename where to save the model, only necessary when saveModel is true");    
	Option readFilenameOption = new Option("modelFilename",true,"filename to read existing model from");    
	Option removeLabelOption = new Option("removeLabel",true,"index of a label to remove");    
        Option numChainsOption = new Option("numChains",true,"number of chains to run");
        Option runOption = new Option("run",true,"run number, if larger than 1, dataset will be randomized");
        Option activeNumLabelsOption = new Option("activeNumLabels",true,"the number of labels to suggest for active learning");
        Option activeRefinementOption = new Option("activeRefinement", true, "strategy for refining classifier, 0: use true labels, 1: use the correct guesses, 2: use correct and incorrect guesses");
        Option batchsizeOption = new Option("batchSize", true, "batch size for online learning");
        options.addOption(batchsizeOption);
        options.addOption(writeOption);
        options.addOption(parallelOption);
        options.addOption(activeNumLabelsOption);
        options.addOption(activeRefinementOption);
        options.addOption(testdataFileOption);
        options.addOption(runOption);
        options.addOption(numChainsOption);    
        options.addOption(dataFileOption);
	options.addOption(inputTypeOption);
        options.addOption(datasetOption);
        options.addOption(percOption);
        options.addOption(numIterOption);
        options.addOption(numTopOption);
        options.addOption(evalInterval);
        options.addOption(wordThresholdOption);
	options.addOption(printOption);
        options.addOption(saveOption);
        options.addOption(saveFilenameOption);
        options.addOption(readFilenameOption);
        options.addOption(removeLabelOption);
        OptionGroup trainTypeGroup = new OptionGroup().addOption(new Option("batch",true,"to train model in batch mode")).addOption(new Option("incremental",true,"train model incrementally")).addOption(new Option("incrementalBatch",true,"train in batch mode with incrementally increasing size of training data")).addOption(new Option("incrementalActive",true,"train incrementally with active learning")).addOption(new Option("readExisting",true,"read existing model from file"));   
        trainTypeGroup.setRequired(true); 
        options.addOptionGroup(trainTypeGroup);
        return options;
    }

    public static Algorithm intValue(String alg){
        if(alg.equals("dependency")){
            return Algorithm.DEPENDENCY;
        }else if(alg.equals("fastdependency")){
            return Algorithm.FASTDEPENDENCY;
        }else if(alg.equals("SCVB")){
            return Algorithm.SCVB;
        }else if(alg.equals("SCVBDependency")){
            return Algorithm.SCVBDEPENDENCY;
        }
        return null;
    }


    public static Options constructOptions(Algorithm alg,boolean supervised){
        Options options = null;
        //get algorithm specific options
        switch(alg){
        case DEPENDENCY: options = dependencyOptions(supervised); break; 
        case FASTDEPENDENCY: options = mydependencyOptions(supervised); break; 
        case SCVB: options = SCVBOptions(supervised); break;
        case SCVBDEPENDENCY: options = SCVBDepOptions(supervised); break;
        }
        //add general options
        Options generalOptions = generalOptions();
        for(Option opt: generalOptions.getOptions()){
            options.addOption(opt);
        }
        return options;
    }

    public static Options SCVBOptions(boolean supervised){
        Options options = new Options();
        Option etaOption = new Option("eta",true,"prior for words");
        Option alphaOption = new Option("alpha",true,"prior for topics");
        Option rhoOption = new Option("rho",true,"for scaling document updates");
        Option rhophiOption = new Option("rhophi",true,"for scaling global updates");
        Option numberOption = new Option("topicNumber",true,"number of topics");
        //    Option batchsizeOption = new Option("batchSize",true,"batch size");
        Option rcOption = new Option("rc",true,"whether or not to use relaxed constraints");
        options.addOption(etaOption).addOption(alphaOption).addOption(rhoOption).addOption(rhophiOption).addOption(numberOption);//.addOption(batchsizeOption);
        if(supervised) options.addOption(rcOption);
        return options;
    }

    
    public static Options SCVBDepOptions(boolean supervised){
        Options options = SCVBOptions(supervised);
        Option etaDepOption = new Option("etaDep",true,"dependency prior for words");
        Option alphaDepOption = new Option("alphaDep",true,"dependency prior for topics");
        Option topicOption = new Option("t",true,"number of dependency topics");
        Option reversedOption = new Option("reversed",true,"whether to reverse topics and labels");
        options.addOption(etaDepOption).addOption(alphaDepOption).addOption(topicOption).addOption(reversedOption);

        return options;
    }

    public static String[] readLine(String filename,int lineNumber){
        BufferedReader br=null;
        String line=null;
        int count = 0;
        try{
            br = new BufferedReader(new FileReader(filename));
            while(count<lineNumber){
                line=br.readLine();
                count++;
            }
        }catch(FileNotFoundException e1){
            e1.printStackTrace();
        }catch(IOException e2){
            e2.printStackTrace();
        }
        String[] pars = line.split(" ");

        return pars;
    }


    
}
