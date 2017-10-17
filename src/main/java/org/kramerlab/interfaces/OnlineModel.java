package org.kramerlab.interfaces;

import cc.mallet.types.InstanceList;

public interface OnlineModel extends TopicModel{
    public void updateBatch(InstanceList data);
}
