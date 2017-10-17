package org.kramerlab.interfaces;

import cc.mallet.types.Instance;


public interface Inferencer{




    public double[] getSampledDistribution(Instance instance,int numIterations,int thinning,int burnin);





}
