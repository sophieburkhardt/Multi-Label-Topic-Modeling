/**
 * Inferencer.java
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

package org.kramerlab.interfaces;

import cc.mallet.types.Instance;


public interface Inferencer{




    public double[] getSampledDistribution(Instance instance,int numIterations,int thinning,int burnin);





}
