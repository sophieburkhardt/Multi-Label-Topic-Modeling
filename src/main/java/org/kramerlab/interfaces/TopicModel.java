package org.kramerlab.interfaces;

import cc.mallet.types.InstanceList;
import java.io.IOException;



public interface TopicModel{



    public Inferencer getInferencer();
    public void sample(int numIterations) throws IOException;
    public void addInstances(InstanceList instances);
    public void printTopWords(String filename,int numTopWords,int wordThreshold);

}
